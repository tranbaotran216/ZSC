import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from typing import List, Dict, Any, Tuple, Optional

def poly_to_xyxy(poly):
    """Chuyển đổi polygon 4 điểm thành [x1, y1, x2, y2]"""
    poly = np.array(poly)
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return [x1, y1, x2, y2]

def xyxy_to_cxcywh_normalized(box, img_w, img_h):
    """Chuyển đổi [x1, y1, x2, y2] sang [cx, cy, w, h] chuẩn hóa (0-1) cho DETR/GroundingDINO"""
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cx, cy, w, h]

class DGDCountDataset(Dataset):
    def __init__(self, root_dir, ann_file, class_file, split_file, split='train', 
                    target_size=384,
                    normalize: bool=None
                ):
        super().__init__()
        self.root_dir = root_dir
        self.target_size = target_size
        self.split = split
        self.normalize = normalize
        
        # Load file annotations
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Load split
        with open(split_file, 'r') as f:
            self.raw_ids = json.load(f)[split]
            

        self.image_ids: List[str] = []
        for v in self.raw_ids:
            name = os.path.basename(v)
            base, ext = os.path.splitext(name)
            img_id = name if ext.lower() == ".jpg" else (base + ".jpg")
            if img_id in self.annotations:
                img_path = os.path.join(self.root_dir, 'images_384_VarV2', img_id)
                if os.path.exists(img_path):
                    self.image_ids.append(img_id)
        
        # 3) Load map image_id -> class_name
        self.img2class: Dict[str, str] = {}
        with open(class_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                raw_id, cls_name = parts[0], parts[1]
                name = os.path.basename(raw_id)
                base, ext = os.path.splitext(name)
                img_id = name if ext.lower() == ".jpg" else (base + ".jpg")
                self.img2class[img_id] = cls_name


        # Khởi tạo CLIP Tokenizer (để xử lý prompt_attn_mask)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Chuẩn hóa ảnh cho VAE Encoder (SD) -> dải [-1, 1]
        self.sd_mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.sd_std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

        # chuẩn hoá cho swin transformer theo chuẩn ImageNet [0, 1]
        self.swin_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.swin_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx:int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        ann = self.annotations[img_id]
        img_path = os.path.join(self.root_dir, 'images_384_VarV2', img_id)
        
        # 1. Dữ liệu Hình ảnh (Image) 
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        ratio_w:float = self.target_size/ orig_w
        ratio_h:float = self.target_size/ orig_h
        
        # Resize cho VAE
        img_resized = img.resize((self.target_size, self.target_size), Image.BICUBIC)
        img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) # (C, H, W)

        img_tensor_sd = (img_tensor - self.sd_mean) / self.sd_std       # Normalize [-1, 1]
        img_tensor_swin = (img_tensor - self.swin_mean) / self.swin_std # Normalize ImageNet

        # 2. Xử lý Văn bản (Prompt & Attention Mask) ---------------------------------
        cls_name = self.img2class.get(img_id, "object")
        prompt = f"{cls_name} ." if not cls_name.endswith(".") else cls_name
        
        # Tạo mask 77 tokens cho CLIP
        prompt_attn_mask = torch.zeros(77, dtype=torch.long)
        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        # Bật mask cho các token thuộc prompt (cộng 1 do token [BOS] ở đầu)
        prompt_attn_mask[1 : 1 + cls_name_length] = 1

        # points chuẩn trong 384x384, ko scale
        points_scaled = torch.tensor(
            ann.get("points", []),
            dtype=torch.float32
        )

        # 3. Dữ liệu Bounding Box và Mẫu (Exemplars & Boxes) -------------------------
        # exemplar boxes
        polys_scaled = [
            torch.tensor(poly, dtype=torch.float32)
            for poly in ann.get("box_examples_coordinates", [])
        ]
        

        if points_scaled.numel() > 0:
            normalized_points = points_scaled.clone()
            normalized_points[:, 0] = points_scaled[:, 0] / orig_w
            normalized_points[:, 1] = points_scaled[:, 1] / orig_h

            points_scaled = points_scaled.clone()
            points_scaled[:, 0] = points_scaled[:, 0] * ratio_w
            points_scaled[:, 1] = points_scaled[:, 1] * ratio_h
            
        exemplar_xyxy = (
            torch.tensor([poly_to_xyxy(p) for p in polys_scaled], dtype=torch.float32)
            if len(polys_scaled) > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        if exemplar_xyxy.numel() > 0:
            exemplar_xyxy= exemplar_xyxy.clone()
            exemplar_xyxy[:, 0] *= ratio_w
            exemplar_xyxy[:, 2] *= ratio_w

            exemplar_xyxy[:, 1] *= ratio_h
            exemplar_xyxy[:, 3] *= ratio_h

        # 4. Dữ liệu Giám sát (Density Map & Attention Map) --------------------------
        # Đọc Density Map (Ground Truth)
        den_path = os.path.join(self.root_dir, 'gt_density_map_adaptive_384_VarV2', img_id.replace('.jpg', '.npy'))
        den_map_tensor = torch.zeros((self.target_size, self.target_size), dtype=torch.float32)
        if os.path.exists(den_path):
            den_map = np.load(den_path)
            gt_count = den_map.sum()

            den_map_resized = cv2.resize(den_map, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            if den_map_resized.sum() > 0 and gt_count >0:
                den_map_resized = den_map_resized * (gt_count / (den_map_resized.sum()))
            den_map_tensor = torch.tensor(den_map_resized, dtype=torch.float32)
 
        img_attn_map = np.ones((self.target_size, self.target_size ), dtype=np.float32)
        img_attn_map_tensor = torch.tensor(img_attn_map, dtype=torch.float32)
        

        num_points = normalized_points.shape[0]
        boxes_tensor = torch.zeros((num_points, 4), dtype=torch.float32)
        if num_points > 0:
            boxes_tensor[:, :2] = normalized_points
            
        # Tất cả gán nhãn 0 (vì prompt của dataset này chỉ hỏi 1 class mỗi ảnh)
        labels_tensor = torch.zeros(num_points, dtype=torch.long)

        target = {
            "images": img_tensor_sd,
            "images_swin": img_tensor_swin,
            "prompts": prompt,
            "prompt_attn_mask": prompt_attn_mask,
            "exemplars": exemplar_xyxy,
            "class_name": cls_name,
            "class_name_token": cls_name_tokens,
            "den_map": den_map_tensor,
            "img_attn_map": img_attn_map_tensor,
            "image_id": img_id,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "points": points_scaled,
            "normalized_points": normalized_points,
            "gt_count": gt_count,
            "boxes": boxes_tensor,  
            "labels": labels_tensor
        }
        return target

# --- Collate Function ---
def dgd_collate_fn(batch):
    batch_dict = {
        "images": torch.stack([b["images"] for b in batch], dim=0),
        "images_swin": torch.stack([b["images_swin"] for b in batch], dim=0) if "images_swin" in batch[0] else None,
        "prompts": [b["prompts"] for b in batch],
        "prompt_attn_mask": torch.stack([b["prompt_attn_mask"] for b in batch], dim=0),
        "exemplars": [b["exemplars"] for b in batch], 
        "points": [b["points"] for b in batch],
        "normalized_points": [b["normalized_points"] for b in batch],
        "gt_counts": torch.tensor([b["gt_count"] for b in batch], dtype=torch.float32),
        "den_map": torch.stack([b["den_map"] for b in batch], dim=0) if "den_map" in batch[0] else None,
        "img_attn_map": torch.stack([b["img_attn_map"] for b in batch], dim=0) if "img_attn_map" in batch[0] else None,
        "image_id": [b["image_id"] for b in batch],
        "orig_size": torch.stack([b["orig_size"] for b in batch], dim=0),
        "class_name": [b["class_name"] for b in batch],
        "class_name_token": [b["class_name_token"] for b in batch] if "class_name_token" in batch[0] else None,

        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch]
    }
    return batch_dict