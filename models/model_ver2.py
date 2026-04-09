from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from gdcount.groundingdino.groundingdino.util.misc import NestedTensor
from gdcount.groundingdino.groundingdino.util.inference import load_model
from T2ICount.models.reg_model import UNetWrapper, load_model_from_config


    
class DGDProjection(nn.Module):
    """
    Lớp chiếu (Projection) để đồng bộ số kênh (channels) từ các feature maps
    của U-Net về hidden_dim của GroundingDINO (thường là 256).
    """
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
    """
    Bọc transformer của GroundingDINO để lấy hs (decoder queries) mỗi lần forward.
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self.last_hs = None

    def __getattr__(self, name: str):
        # proxy attribute sang transformer gốc (tránh thiếu field)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(self, *args, **kwargs):
        outputs = self.transformer(*args, **kwargs)
        # GroundingDINO thường trả tuple, hs ở outputs[0]
        if isinstance(outputs, tuple) and len(outputs) > 0:
            self.last_hs = outputs[0]
        else:
            self.last_hs = None
        return outputs


class CountHead(nn.Module):
    """
    Count head: đọc hs (decoder queries) -> logit per query -> soft/hard count.
    """

    def __init__(self, d_model: int, threshold: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.threshold = float(threshold)

    def forward(self, hs):
        """
        hs :
          - Tensor [num_layers, B, Q, D]
          - Tensor [B, Q, D]
          - List[Tensor[B, Q, D]]
        """
        if isinstance(hs, (list, tuple)):
            if len(hs) == 0 or not isinstance(hs[0], torch.Tensor):
                raise ValueError(f"Unexpected hs list: {type(hs)}, first={type(hs[0]) if hs else None}")
            hs = torch.stack(hs, dim=0)  # [L,B,Q,D]

        if hs.dim() == 4:
            hs_last = hs[-1]  # (B,Q,D)
        elif hs.dim() == 3:
            hs_last = hs
        else:
            raise ValueError(f"Unexpected hs shape: {tuple(hs.shape)}")

        hs_last = self.norm(hs_last)
        logits = self.fc(hs_last).squeeze(-1)     # (B,Q)

        soft = torch.relu(logits).sum(dim=1)      # (B,)
        hard = (logits > self.threshold).sum(dim=1).to(torch.int64)
        return logits, soft, hard



class DiffusionGroundingDINOCounter(nn.Module):
    def __init__(self, 
                sd_config_path: str,
                sd_ckpt_path: str,
                gd_base_model: nn.Module,
                unet_channels: List[int] = [320, 640, 1280, 1280], # Đã sửa typo
                gd_hidden_dim: int = 256,
                unet_config: dict = dict(),
                gd_threshold: float = 0.23,
                clip_text_dim: int=768
                ):
        super().__init__()

        # 1. initialize stable diffusion
        sd_config_obj = OmegaConf.load(sd_config_path)
        sd_model = load_model_from_config(sd_config_obj, sd_ckpt_path)
        self.vae = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, **unet_config)
        self.clip = sd_model.cond_stage_model
        self.hidden_dim = gd_hidden_dim

        self.vae.eval()
        self.clip.eval()
        
        # 2. projection
        self.feat_proj = DGDProjection(unet_channels=unet_channels, hidden_dim=self.hidden_dim)
        self.text_proj = nn.Linear(clip_text_dim, self.hidden_dim)
        # 3. gdino components
        self.gd_model = gd_base_model
        self.gd_model.transformer = TransformerWrapper(self.gd_model.transformer)

        if hasattr(self.gd_model, 'backbone') and len(self.gd_model.backbone) > 1:
            self.pos_embed_layer = self.gd_model.backbone[1] # Giữ lại Positional Embedding
            del self.gd_model.backbone
        
        if hasattr(self.gd_model, 'bert'):
            del self.gd_model.bert
        if hasattr(self.gd_model, 'tokenizer'):
            del self.gd_model.tokenizer
        if hasattr(self.gd_model, 'input_proj'):
            del self.gd_model.input_proj
        if hasattr(self.gd_model, 'feat_map'):
            del self.gd_model.feat_map

        # 5. countHead
        self.count_head = CountHead(int(self.hidden_dim), threshold=gd_threshold)
        self.set_frozen_parameters()


    def set_frozen_parameters(self):
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad= False
            
        if hasattr(self, 'pos_embed_layer'):
            for param in self.pos_embed_layer.parameters():
                param.requires_grad = False


        
    def extract_unet_features(self, img: torch.Tensor, prompts: List[str]) -> Tuple[Tuple, torch.Tensor]:
        with torch.no_grad():
            latents = self.vae.encode(img)
            c_crossattn = self.clip.encode(prompts) 

        # Kiểm tra kĩ latents có phải Distribution object không
        if hasattr(latents, 'mode'):
            latents = latents.mode().detach()
        else:
            # Dành cho diffusers wrapper
            latents = latents.latent_dist.mode().detach()

        t = torch.zeros((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        
        # return: outs: (img_feat_list, attn12, attn24, attn48); c_crossattn: clip text_feat
        return outs, c_crossattn 

    def forward(self, x: torch.Tensor, prompts: List[str], targets: Any = None, exemplars: Any = None, labels: Any = None, **kwargs):
        outs, clip_text_feat = self.extract_unet_features(img=x, prompts=prompts)
        
        # outs là tuple 4 phần tử (từ T2ICount): img_feats(List), attn12, attn24, attn48
        img_feats, attn12, attn24, attn48 = outs
        
        # Project channels của Unet về không gian của GDino (256)
        proj_img_feats = self.feat_proj(img_feats)
        
        srcs = []
        masks = []
        pos_embeds = []
        for feat in proj_img_feats:
            # Tạo mask padding (toàn False)
            mask = torch.zeros((feat.shape[0], feat.shape[2], feat.shape[3]), dtype=torch.bool, device=feat.device)
            srcs.append(feat)
            masks.append(mask)
            pos_embed = self.pos_embed_layer(NestedTensor(feat, mask)).to(feat.dtype)
            pos_embeds.append(pos_embed)

        
        text_feat = self.text_proj(clip_text_feat)
        max_length = clip_text_feat.shape[1] # 77
        tokens = self.clip.tokenizer(
            prompts, 
            padding="max_length", 
            max_length=max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(x.device)
        attention_mask = tokens.attention_mask.bool()

        # 3. Tính toán Square Self-Attention Masks (B, L, L)
        text_self_attention_masks = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        B, L = attention_mask.shape
        idx = torch.arange(L, device=x.device)
        text_self_attention_masks[:, idx, idx] = True

        # 4. Tạo Text Dict cho GDino
        text_dict = {
            "encoded_text": text_feat,
            "text_token_mask": attention_mask,
            "position_ids": torch.arange(L, device=x.device).unsqueeze(0).repeat(B, 1),
            "text_self_attention_masks": text_self_attention_masks
        }

        hs, references, hs_enc, ref_enc, init_box_proposal = self.gd_model.transformer(
            srcs=srcs,
            masks=masks,
            refpoint_embed=None,  # Bắt buộc truyền None (hoặc tensor phù hợp nếu train)
            pos_embeds=pos_embeds,
            tgt=None,             # Bắt buộc truyền None
            attn_mask=None,       # Có thể bỏ qua vì default là None
            text_dict=text_dict   # Truyền text_dict vào đúng tham số
        )
        if hs is None:
            raise RuntimeError("TransformerWrapper cannot capture hs (decoder output).")
        
        hs_last_layer = hs[-1]
        # normalized pred bboxes
        outputs_coord = self.gd_model.bbox_embed[-1](hs_last_layer).sigmoid()
        # Tính toán logits và counts
        query_logits, soft_counts, hard_counts = self.count_head(hs_last_layer)
        
        outputs = {
            "pred_boxes": outputs_coord,
            "query_logits": query_logits,
            "pred_logits": query_logits,
            "soft_counts": soft_counts,
            "hard_counts": hard_counts,
            "caption": prompts,
        }
        return outputs

@dataclass
class GDinoConfig:
    """
    threshold: hard count threshold
    hidden_dim: GDino's hidden dim (usually 256)
    """
    threshold: float = 0.0 
    hidden_dim: int = 256

def build_dgd_model(
    gd_config_path: str,
    gd_ckpt_path:str,
    sd_config_path: str,
    sd_ckpt_path: str,
    unet_config: dict,
    gdconfig: GDinoConfig,
    device:str
) -> DiffusionGroundingDINOCounter:
    base_dino = load_model(
        model_config_path=gd_config_path,
        model_checkpoint_path=gd_ckpt_path,
        device=device,
    )

    base_dino.train() 
    model=DiffusionGroundingDINOCounter(sd_config_path=sd_config_path, sd_ckpt_path=sd_ckpt_path,
                                        gd_base_model=base_dino, gd_hidden_dim=gdconfig.hidden_dim,
                                        unet_config=unet_config, gd_threshold=gdconfig.threshold)
    return model
    