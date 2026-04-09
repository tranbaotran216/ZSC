from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from gdcount.groundingdino.groundingdino.util.misc import NestedTensor
from gdcount.groundingdino.groundingdino.util.inference import load_model
from gdcount.groundingdino.groundingdino.models.GroundingDINO.transformer import VisionCrossFusion
from T2ICount.models.reg_model import UNetWrapper, load_model_from_config
import types

    
class DGDProjection(nn.Module):
    def __init__(self, unet_channels: List[int], hidden_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.GELU()
            ) for in_ch in unet_channels
        ])

    def forward(self, unet_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(unet_feats) == len(self.projs), "Mismatch between features and projection layers"
        return [proj(feat) for proj, feat in zip(self.projs, unet_feats)]
    
class TransformerWrapper(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.last_hs = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        if isinstance(outputs, tuple) and len(outputs) > 0:
            self.last_hs = outputs[0]
        else:
            self.last_hs = None
        return outputs

class CountHead(nn.Module):
    def __init__(self, d_model: int, threshold: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.threshold = float(threshold)

    def forward(self, hs):
        if isinstance(hs, (list, tuple)):
            if len(hs) == 0 or not isinstance(hs[0], torch.Tensor):
                raise ValueError(f"Unexpected hs list: {type(hs)}, first={type(hs[0]) if hs else None}")
            hs = torch.stack(hs, dim=0)  

        if hs.dim() == 4:
            hs_last = hs[-1]  
        elif hs.dim() == 3:
            hs_last = hs
        else:
            raise ValueError(f"Unexpected hs shape: {tuple(hs.shape)}")

        hs_last = self.norm(hs_last)
        logits = self.fc(hs_last).squeeze(-1)     

        soft = torch.relu(logits).sum(dim=1)      
        hard = (logits > self.threshold).sum(dim=1).to(torch.int64)
        return logits, soft, hard

class DiffusionGroundingDINOCounter(nn.Module):
    def __init__(self, 
                sd_config_path: str,
                sd_ckpt_path: str,
                gd_base_model: nn.Module,
                unet_channels: List[int] = [320, 640, 1280, 1280], 
                gd_hidden_dim: int = 256,
                unet_config: dict = dict(),
                gd_threshold: float = 0.23,
                clip_text_dim: int=768,
                fusion_option: str = "base",
                ):
        super().__init__()

        self.hidden_dim = gd_hidden_dim
        self.fusion_option = fusion_option
        assert self.fusion_option in ["base", "attn"], "fusion option must be 'base' or 'attn'"

        sd_config_obj = OmegaConf.load(sd_config_path)
        sd_model = load_model_from_config(sd_config_obj, sd_ckpt_path)
        
        self.vae = sd_model.first_stage_model
        self.vae.eval()
        self.vae.float() # Ép VAE luôn chạy ở float32
        
        self.unet = UNetWrapper(sd_model.model, **unet_config)
        self.unet.eval()
        
        self.clip = sd_model.cond_stage_model
        self.clip.eval()
        
        self.unet_feat_proj = DGDProjection(unet_channels=unet_channels, hidden_dim=self.hidden_dim)
        
        self.gd_model = gd_base_model
        self.gd_model.transformer = TransformerWrapper(self.gd_model.transformer)

        if self.gd_model.num_feature_levels != len(unet_channels):
            raise ValueError(f"Mismatch: swin - {self.gd_model.num_feature_levels} vs unet - {len(unet_channels)}")
            
        if self.fusion_option == "base":
            self.fusion_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, self.hidden_dim),
                    nn.GELU(),
                    # Zero-initalization
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
                ) for _ in range(self.gd_model.num_feature_levels) 
            ])

            for m in self.fusion_convs:
                nn.init.constant_(m[-1].weight, 0.0)
                nn.init.constant_(m[-1].bias, 0.0)

        elif self.fusion_option == "attn":
            encoder = self.gd_model.transformer.encoder
            if not hasattr(encoder, 'vv_fusion_layers') or encoder.vv_fusion_layers is None:
                encoder.vision_fusion = 'attn'
                encoder.vv_fusion_layers = nn.ModuleList([
                    VisionCrossFusion(d_model=self.hidden_dim, nhead=8)
                    for _ in range(encoder.num_layers)
                ]).to(next(self.gd_model.parameters()).device)
                
        self.count_head = CountHead(int(self.hidden_dim), threshold=gd_threshold)
        self.set_frozen_parameters()

    def set_frozen_parameters(self):
        for param in self.vae.parameters(): param.requires_grad = False
        for param in self.clip.parameters(): param.requires_grad = False
        for param in self.unet.parameters(): param.requires_grad= False
        if hasattr(self.gd_model, 'backbone'):
            for param in self.gd_model.backbone.parameters(): param.requires_grad=False
        if hasattr(self.gd_model, 'bert'):
            for param in self.gd_model.bert.parameters(): param.requires_grad=False

    def extract_unet_features(self, img: torch.Tensor, prompts: List[str]) -> Tuple[Tuple, torch.Tensor]:
        with torch.no_grad():
            # Tắt AMP (Mixed Precision) khi encode VAE
            with torch.cuda.amp.autocast(enabled=False):
                latents = self.vae.encode(img.float())
                if hasattr(latents, 'mode'):
                    latents = latents.mode().detach()
                else:
                    latents = latents.latent_dist.mode().detach()
                
                # Chốt chặn NaN/Inf cho latents của VAE
                latents = torch.nan_to_num(latents, nan=0.0, posinf=50.0, neginf=-50.0)

            c_crossattn = self.clip.encode(prompts) 

        t = torch.zeros((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
    
        safe_outs = []
        for o in outs:
            if isinstance(o, torch.Tensor):
                safe_outs.append(torch.nan_to_num(o, nan=0.0))
            elif isinstance(o, (list, tuple)):
                safe_outs.append(type(o)(torch.nan_to_num(x, nan=0.0) for x in o))
            else:
                safe_outs.append(o)
                
        return tuple(safe_outs), c_crossattn
    


    def get_text_dict(self, prompts: List[str], device: torch.device) -> dict:
        captions = [p.lower().strip() for p in prompts]
        
        max_text_len = getattr(self.gd_model, 'max_text_len', 256)
        
        tokenized = self.gd_model.tokenizer(
            captions, 
            padding="max_length", 
            max_length=max_text_len, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        try:
            from gdcount.groundingdino.groundingdino.models.GroundingDINO.utils import generate_masks_with_special_tokens_and_transfer_map
            special_tokens = getattr(self.gd_model, 'specical_tokens', getattr(self.gd_model, 'special_tokens', None))
            (
                text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, special_tokens, self.gd_model.tokenizer
            )
        except Exception:
            text_token_mask = tokenized.attention_mask.bool()
            bs, max_len = text_token_mask.shape
            text_self_attention_masks = text_token_mask.unsqueeze(1) & text_token_mask.unsqueeze(2)
            idx = torch.arange(max_len, device=device)
            text_self_attention_masks[:, idx, idx] = True
            position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).repeat(bs, 1)
            
        # Tắt AMP cho BERT để tránh tràn số FP16
        with torch.cuda.amp.autocast(enabled=False):
            bert_output = self.gd_model.bert(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
            )
            raw_text_features = bert_output["last_hidden_state"].float()
            
            # Project từ 768 -> 256 để khớp với Swin
            if hasattr(self.gd_model, 'feat_map'):
                encoded_text = self.gd_model.feat_map(raw_text_features)
            elif hasattr(self, 'feat_map'):
                encoded_text = self.feat_map(raw_text_features)
            else:
                raise AttributeError("Cannot find BERT projection layer to convert 768 to 256 dim")
            
        encoded_text = torch.nan_to_num(encoded_text, nan=0.0)
        text_token_mask = tokenized.attention_mask.bool()
        
        return {
            "encoded_text": encoded_text,                  
            "text_token_mask": text_token_mask,             # [B, 256]
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
            "tokenized": tokenized                          
        }

    def forward(self, x: torch.Tensor, prompts: List[str], images_swin: torch.Tensor = None, targets: Any = None, exemplars: Any = None, labels: Any = None, **kwargs):
        
        # 1. SD dùng tensor ảnh [-1, 1]
        outs, clip_text_feat = self.extract_unet_features(img=x, prompts=prompts)
        unet_img_feats, attn12, attn24, attn48 = outs
        proj_unet_feats = self.unet_feat_proj(unet_img_feats) 
        
        # 2. Nếu images_swin chưa được nạp, auto fallback chuẩn hóa ImageNet
        if images_swin is None:
            images_swin = (x + 1.0) / 2.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            images_swin = (images_swin - mean) / std

        masks = torch.zeros((images_swin.shape[0], images_swin.shape[2], images_swin.shape[3]), dtype=torch.bool, device=images_swin.device)
        
        nested_x = NestedTensor(images_swin, masks)
        swin_features, pos_embeds = self.gd_model.backbone(nested_x)
        
        srcs = []
        swin_masks = []
        for l, feat in enumerate(swin_features):
            src, mask = feat.decompose()
            src_proj = self.gd_model.input_proj[l](src)
            src_proj = torch.nan_to_num(src_proj, nan=0.0)
            srcs.append(src_proj)
            swin_masks.append(mask)

        if self.gd_model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l_idx in range(_len_srcs, self.gd_model.num_feature_levels):
                if l_idx == _len_srcs:
                    src_l = self.gd_model.input_proj[l_idx](swin_features[-1].tensors)
                else:
                    src_l = self.gd_model.input_proj[l_idx](srcs[-1])
                    
                src_l = torch.nan_to_num(src_l, nan=0.0)
                
                m = swin_masks[-1]
                mask_l = F.interpolate(m[None].float(), size=src_l.shape[-2:]).to(torch.bool)[0]
                pos_l = self.gd_model.backbone[1](NestedTensor(src_l, mask_l)).to(src_l.dtype)
                srcs.append(src_l)
                swin_masks.append(mask_l)
                pos_embeds.append(pos_l)

        # 3. Lấy text_dict chuẩn đã được pad về đúng 256 từ hàm get_text_dict
        text_dict = self.get_text_dict(prompts, x.device)

        unet_f_resized_list = []
        for l in range(len(srcs)): 
            if proj_unet_feats[l].shape[-2:] != srcs[l].shape[-2:]:
                unet_f_resized = F.interpolate(proj_unet_feats[l], size=srcs[l].shape[-2:], mode='bilinear', align_corners=False)
            else:
                unet_f_resized = proj_unet_feats[l]
            unet_f_resized_list.append(unet_f_resized)

        # 4. Fusion 
        if self.fusion_option == "base":
            fused_srcs = []
            for l in range(len(srcs)): 
                concat_feat = torch.cat([srcs[l], unet_f_resized_list[l]], dim=1) 
                delta = self.fusion_convs[l](concat_feat)  
                fused = srcs[l] + torch.nan_to_num(delta, nan=0.0)                
                fused_srcs.append(fused)
            final_srcs = fused_srcs
            final_srcs2 = None  
        elif self.fusion_option == 'attn':
            final_srcs = srcs
            final_srcs2 = unet_f_resized_list

        # 5. Transformer
        hs, references, hs_enc, ref_enc, init_box_proposal = self.gd_model.transformer(
            srcs=final_srcs,
            masks=swin_masks,
            refpoint_embed=None,
            pos_embeds=pos_embeds,
            tgt=None,
            attn_mask=None,
            text_dict=text_dict,
            srcs2=final_srcs2  
        )
        
        hs_last_layer = hs[-1]
        
        # 6. Lấy Outputs
        outputs_coord = self.gd_model.bbox_embed[-1](hs_last_layer).sigmoid()
        
        # pred_logits nhận thẳng text_dict (đã 256 channels)
        pred_logits = self.gd_model.class_embed[-1](hs_last_layer, text_dict)
        query_logits, soft_counts, hard_counts = self.count_head(hs_last_layer)
        
        outputs = {
            "pred_boxes": outputs_coord,                
            "pred_logits": pred_logits,                 
            "token": text_dict["tokenized"],            
            "text_mask": text_dict["text_token_mask"],  
            "query_logits": query_logits,               
            "soft_counts": soft_counts,
            "hard_counts": hard_counts,
            "caption": prompts,
        }
        return outputs
    
    

@dataclass
class GDinoConfig:
    threshold: float = 0.0 
    hidden_dim: int = 256

def build_dgd_model(
    gd_config_path: str,
    gd_ckpt_path:str,
    sd_config_path: str,
    sd_ckpt_path: str,
    unet_config: dict,
    gdconfig: GDinoConfig,
    device:str,
    fusion_option: str="base"
) -> DiffusionGroundingDINOCounter:
    base_dino = load_model(
        model_config_path=gd_config_path,
        model_checkpoint_path=gd_ckpt_path,
        device=device,
    )

    base_dino.train() 
    model=DiffusionGroundingDINOCounter(sd_config_path=sd_config_path, sd_ckpt_path=sd_ckpt_path,
                                        gd_base_model=base_dino, gd_hidden_dim=gdconfig.hidden_dim,
                                        unet_config=unet_config, gd_threshold=gdconfig.threshold,
                                        fusion_option=fusion_option
                                        )
    return model