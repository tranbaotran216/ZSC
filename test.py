import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\projects\zsc")

from dataset import DGDCountDataset, dgd_collate_fn
from models import dgd_model, model_ver2
from models import model_ver3_fusion as model_ver3  
from src.helpers import * 


def parse_test_args():
    parser = argparse.ArgumentParser("Test DGD-Count Model")
    # Paths
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained .pth file")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--ann", type=str, default="FSC147/annotation_FSC147_384.json")
    parser.add_argument("--img-root", type=str, default="images_384_VarV2")
    parser.add_argument("--split-file", type=str, default="FSC147/Train_Test_Val_FSC_147.json")
    parser.add_argument("--class-map", type=str, default="FSC147/ImageClasses_FSC147.txt")
    
    # Model Configs
    parser.add_argument("--gd_config_path", default="./gdcount/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gd_ckpt_path", default="./gdcount/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--sd_config_path", default="./T2ICount/configs/v1-inference.yaml")
    parser.add_argument("--sd_ckpt_path", default="./T2ICount/configs/v1-5-pruned-emaonly.ckpt")
    
    # Params
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--crop_size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--nms_iou", type=float, default=0.5)

    # Thêm train_version 3 và fusion_option
    parser.add_argument("--train_version", type=int, default=1, choices=[1, 2, 3], help="1 for dgd_model, 2 for model_ver2, 3 for model_ver3")
    parser.add_argument("--fusion_option", type=str, default="base", choices=['base', 'attn'], help="option fusion for model arch 3")
    
    return parser.parse_args()

def test():
    args = parse_test_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root if args.data_root else os.path.join(project_root, "data")
    data_root = os.path.normpath(data_root)

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(data_root, p))

    args.data_root = data_root
    args.ann = _resolve(args.ann)
    args.img_root = _resolve(args.img_root)
    args.split_file = _resolve(args.split_file)
    args.class_map = _resolve(args.class_map)

    # 1. Load Dataset
    dataset = DGDCountDataset(
        root_dir=args.data_root, ann_file=args.ann, class_file=args.class_map,
        split_file=args.split_file, split='test', target_size=args.crop_size
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dgd_collate_fn)

    # 2. Build Model
    if args.train_version == 1:
        cfg = dgd_model.GDinoConfig(threshold=args.threshold, hidden_dim=256)
        model = dgd_model.build_dgd_model(
            args.gd_config_path, args.gd_ckpt_path, args.sd_config_path, args.sd_ckpt_path,
            unet_config={
                'base_size': args.crop_size,
                'max_attn_size': args.crop_size // 8,
                'attn_selector': 'down_cross+up_cross'
            }, 
            gdconfig=cfg, device=args.device
        )
    elif args.train_version == 2:
        cfg = model_ver2.GDinoConfig(threshold=args.threshold, hidden_dim=256)
        model = model_ver2.build_dgd_model(
            gd_config_path=args.gd_config_path, gd_ckpt_path=args.gd_ckpt_path, 
            sd_config_path=args.sd_config_path, sd_ckpt_path=args.sd_ckpt_path,
            unet_config={'base_size': args.crop_size, 'max_attn_size': args.crop_size // 8,
                         'attn_selector': 'down_cross+up_cross'},
            gdconfig=cfg, device=args.device
        )
    elif args.train_version == 3:
        cfg = model_ver3.GDinoConfig(threshold=args.threshold, hidden_dim=256)
        model = model_ver3.build_dgd_model(
            gd_config_path=args.gd_config_path, 
            gd_ckpt_path=args.gd_ckpt_path, 
            sd_config_path=args.sd_config_path, 
            sd_ckpt_path=args.sd_ckpt_path,
            unet_config={'base_size': args.crop_size, 'max_attn_size': args.crop_size // 8,
                         'attn_selector': 'down_cross+up_cross'},
            gdconfig=cfg, 
            device=args.device,
            fusion_option=args.fusion_option 
        )
    else:
        raise ValueError(f"Invalid train_version: {args.train_version}")
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    # 4. Evaluation Loop
    mae_list, mse_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            images = batch["images"].to(device)
            # Thêm images_swin
            images_swin = batch["images_swin"].to(device) if batch.get("images_swin") is not None else None
            prompts = [sanitize_caption(p) for p in batch["prompts"]]
            gt_counts = batch["gt_counts"].to(device)

            outputs = model(images, prompts=prompts, images_swin=images_swin)
            
            pred_counts = count_by_det_nms(outputs, threshold=args.threshold, nms_iou=args.nms_iou, target_size=args.crop_size)
            
            pred_counts = pred_counts.to(gt_counts.device)
            diff = (pred_counts - gt_counts).abs()
            
            mae_list.extend(diff.tolist())
            mse_list.extend((diff**2).tolist())

    print(f"\n[FINAL RESULTS]")
    print(f"MAE: {np.mean(mae_list):.4f}")
    print(f"RMSE: {np.sqrt(np.mean(mse_list)):.4f}")

if __name__ == "__main__":
    test()