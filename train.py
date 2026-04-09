import argparse
import os, sys
import csv
from typing import Any, Dict, List, Tuple
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision.ops import nms as tv_nms
import time

import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
# sys.path.insert(0, r"D:\projects\zsc")

from dataset import DGDCountDataset, dgd_collate_fn
from models import dgd_model
from models import model_ver2  
from models import model_ver3_fusion as model_ver3  
from src.helpers import *
from CountGD.models.GroundingDINO.groundingdino import SetCriterion
from CountGD.models.GroundingDINO.matcher import HungarianMatcher


# ========================
#  ARGUMENTS
# ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train GDCount on FSC147 (CountGD-style)")

    # ---- GroundingDINO ----
    parser.add_argument("--gd_config_path", default=r"D:\projects\zsc\gdcount\groundingdino\groundingdino\config\GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gd_ckpt_path", default=r"D:\projects\zsc\gdcount\weights\groundingdino_swint_ogc.pth")

    parser.add_argument(
            "--data-root",
            type=str,
            default=None,
            help="Root folder that contains FSC147 and image folder. "
                "If not set, will default to <project_root>/data",
        )
    # ---- Data ----
    parser.add_argument("--ann", type=str, default="FSC147/annotation_FSC147_384.json")
    parser.add_argument("--img-root", type=str, default="images_384_VarV2")
    parser.add_argument("--split-file", type=str, default="FSC147/Train_Test_Val_FSC_147.json")
    parser.add_argument("--class-map", type=str, default="FSC147/ImageClasses_FSC147.txt")

    # ---- Train ----
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="checkpoints_gdcount")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--use-exemplar", action="store_true")
    parser.add_argument("--exp-name", type=str, default="")

    # ---- GDCount config ----
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--freeze-keywords", type=str, nargs="+", default=["backbone.0", "bert"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start-epoch", type=int, default=-1)
    parser.add_argument("--accum-steps", type=int, default=1)

    # ---- PSEUDO BOX SIZE: per-image NN + jitter + fallback ----
    parser.add_argument("--side-mode", type=str, default="nn",
                        choices=["nn", "sqrt"],
                        help="nn: side from nearest-neighbor; sqrt: side from sqrt(H*W/cnt)")
    parser.add_argument("--nn-k", type=float, default=0.8,
                        help="side0 = nn_k * median(nearest_neighbor_dist)")
    parser.add_argument("--nn-max-samples", type=int, default=256,
                        help="subsample points for cdist")
    parser.add_argument("--side-jitter-low", type=float, default=0.7,
                        help="log-uniform jitter low (per-image)")
    parser.add_argument("--side-jitter-high", type=float, default=1.4,
                        help="log-uniform jitter high (per-image)")
    parser.add_argument("--lambda_cls", default=5.0)
    parser.add_argument("--lambda_loc", default=1.0)

    # fallback sqrt(HW/cnt)
    parser.add_argument("--side-mult", type=float, default=0.5,
                        help="sqrt-mode: side = side_mult * sqrt(H*W/cnt)")

    # clamps/boost
    parser.add_argument("--side-min", type=float, default=6.0)
    parser.add_argument("--side-max-ratio", type=float, default=0.8,
                        help="Max side = side_max_ratio * min(H,W)")
    parser.add_argument("--boost-cnt2-ratio", type=float, default=0.50,
                        help="If gt_count<=2: side >= boost_cnt2_ratio*min(H,W)")
    parser.add_argument("--boost-cnt5-ratio", type=float, default=0.35,
                        help="If gt_count<=5: side >= boost_cnt5_ratio*min(H,W)")
    
    # Diffusion & Unet config
    parser.add_argument("--crop_size", default=384) #unet
    parser.add_argument("--sd_config_path", default=r"D:\projects\zsc\T2ICount\configs\v1-inference.yaml")
    parser.add_argument("--sd_ckpt_path", default=r"D:\projects\zsc\T2ICount\configs\v1-5-pruned-emaonly.ckpt")

    parser.add_argument("--train_version", type=int, default=1, choices=[1, 2, 3], help="1 for dgd_model, 2 for model_ver2, 3 for model_ver3")
    parser.add_argument("--fusion_option", type=str, choices=['base', 'attn'], help="option fusion for model arch 3")
    parser.add_argument("--log_one_epoch", action="store_true", help="train only 1 epoch and write log to csv")
    args = parser.parse_args()

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

    return args

def create_dataloader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = DGDCountDataset(
        root_dir=args.data_root,
        ann_file=args.ann,
        class_file=args.class_map,
        split_file=args.split_file,
        split=split,
        target_size=args.crop_size
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=(split=='train'),
        num_workers=4,
        pin_memory=True,
        collate_fn=dgd_collate_fn
    )
    return loader

# ========================
#  MODEL + OPTIMIZER
# ========================
def create_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.train_version == 1:
        cfg = dgd_model.GDinoConfig(threshold=args.threshold, hidden_dim=256)
        model = dgd_model.build_dgd_model(
            gd_config_path=args.gd_config_path, gd_ckpt_path=args.gd_ckpt_path, 
            sd_config_path=args.sd_config_path, sd_ckpt_path=args.sd_ckpt_path,
            unet_config={'base_size': args.crop_size, 'max_attn_size': args.crop_size // 8, 'attn_selector': 'down_cross+up_cross'},
            gdconfig=cfg, device=args.device
        )
    elif args.train_version == 2:
        cfg = model_ver2.GDinoConfig(threshold=args.threshold, hidden_dim=256)
        model = model_ver2.build_dgd_model(
            gd_config_path=args.gd_config_path, gd_ckpt_path=args.gd_ckpt_path, 
            sd_config_path=args.sd_config_path, sd_ckpt_path=args.sd_ckpt_path,
            unet_config={'base_size': args.crop_size, 'max_attn_size': args.crop_size // 8, 'attn_selector': 'down_cross+up_cross'},
            gdconfig=cfg, device=args.device
        )
    elif args.train_version == 3:
        if args.fusion_option not in ['base', 'attn']:
            raise ValueError(f"Invalid fusion option: {args.fusion_option}. Please provide '--fusion_option base' or '--fusion_option attn'.")
            
        cfg = model_ver3.GDinoConfig(threshold=args.threshold, hidden_dim=256)
        model = model_ver3.build_dgd_model(
            gd_config_path=args.gd_config_path, 
            gd_ckpt_path=args.gd_ckpt_path, 
            sd_config_path=args.sd_config_path, 
            sd_ckpt_path=args.sd_ckpt_path,
            unet_config={'base_size': args.crop_size, 'max_attn_size': args.crop_size // 8, 'attn_selector': 'down_cross+up_cross'},
            gdconfig=cfg, 
            device=args.device,
            fusion_option=args.fusion_option 
        )
    else:
        raise ValueError(f"Invalid train_version: {args.train_version}")
        
    return model


def create_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    head_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "count_head" in name:
            head_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": head_params, "lr": lr},
        {"params": base_params, "lr": lr * 0.5},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    return optimizer, scheduler   

# ========================
#  TRAIN / EVAL
# ========================

LOSS_KEYS = ["loss_total", "loss_loc", "loss_cls", "count_mae"]

def _init_loss_sums() -> Dict[str, float]:
    return {k: 0.0 for k in LOSS_KEYS}

def _avg_losses(loss_sums: Dict[str, float], num_samples: int) -> Dict[str, float]:
    if num_samples == 0:
        return {k: 0.0 for k in LOSS_KEYS}
    return {k: v / num_samples for k, v in loss_sums.items()}


def train_one_epoch(
    epoch: int, 
    model: torch.nn.Module, 
    criterion: torch.nn.Module, # Thêm tham số criterion
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scaler: GradScaler, 
    device: str, 
    log_interval: int, 
    args: argparse.Namespace
) -> Dict[str, float]:
    
    model.train()
    criterion.train()
    loss_sums = _init_loss_sums()
    num_samples = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [train]", ncols=150, leave=True)
    accum_steps = max(1, int(args.accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, batch in pbar:
        images = batch["images"].to(device)
        prompts = [sanitize_caption(x) for x in batch["prompts"]]
        
        # 1. Chuyển đổi batch thành List[Dict] targets cho SetCriterion
        targets = []
        for i in range(len(images)):
            targets.append({
                "boxes": batch["boxes"][i].to(device),
                "labels": batch["labels"][i].to(device)
            })
            
        # 2. Tạo cat_list (FSC147 mỗi ảnh 1 class nên luôn là [0])
        cat_list = [[0] for _ in range(len(images))]

        with autocast(enabled=(args.amp and device.startswith("cuda"))):
            if args.use_exemplar:
                raise ValueError("exemplar code not built yet ~~")
            else:
                outputs: Dict[str, Any] = model(images, prompts=prompts)
            outputs.pop("hard_counts", None)

            # 3. Tính Loss bằng SetCriterion
            # Trả về dict chứa loss_ce, loss_bbox, loss_giou,...
            loss_dict = criterion(outputs, targets, cat_list, prompts)
            weight_dict = criterion.weight_dict
            
            # Tổng hợp các loss có trọng số
            loss_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # 4. MAE Count (Không backprop)
            pred_counts = torch.relu(outputs["query_logits"]).sum(dim=1)
            gt_counts = batch["gt_counts"].to(device)
            count_mae = (pred_counts - gt_counts).abs().mean()
            
            # Lưu vào loss_dict để in ra log
            loss_dict["loss_total"] = loss_total
            loss_dict["count_mae"] = count_mae
            loss_dict["loss_loc"] = loss_dict.get("loss_bbox", torch.tensor(0.0))
            loss_dict["loss_cls"] = loss_dict.get("loss_ce", torch.tensor(0.0))
            
            loss_to_backprop = loss_total / accum_steps

        # 5. Backpropagation
        scaler.scale(loss_to_backprop).backward()

        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        num_samples += batch_size
        for k in LOSS_KEYS:
            v = loss_dict.get(k, torch.tensor(0.0, device=device)).item()
            loss_sums[k] += v * batch_size

        avg_now = _avg_losses(loss_sums, num_samples)
        
        if step % log_interval == 0:
            pbar.set_postfix({
                "L": f"{avg_now['loss_total']:.3f}",
                "Loc": f"{avg_now['loss_loc']:.3f}",
                "Cls": f"{avg_now['loss_cls']:.3f}",
                "MAE": f"{avg_now['count_mae']:.1f}"
            })

    return _avg_losses(loss_sums, num_samples)



def evaluate(
    epoch: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,  
    loader: DataLoader,
    device: str,
    args: argparse.Namespace,
    split_name: str = "val" 
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict]]:
    model.eval()
    criterion.eval()
    loss_sums = _init_loss_sums()
    num_samples = 0

    mae_sum = 0.0
    mse_sum = 0.0
    n_count_samples = 0
    predictions_list = [] 

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [{split_name}]  ", ncols=150)

    with torch.no_grad():
        for step, batch in pbar:
            images = batch["images"].to(device)
            images_swin = batch["images_swin"].to(device)
            prompts = [sanitize_caption(x) for x in batch["prompts"]]
            gt_counts = batch["gt_counts"].to(device)

            batch_size = images.shape[0]
            num_samples += batch_size
            
            # 1. Format targets và cat_list cho SetCriterion (Y hệt train_one_epoch)
            targets = []
            for i in range(batch_size):
                targets.append({
                    "boxes": batch["boxes"][i].to(device),
                    "labels": batch["labels"][i].to(device)
                })
            cat_list = [[0] for _ in range(batch_size)]

            with autocast(enabled=(args.amp and device.startswith("cuda"))):
                if args.use_exemplar:
                    exemplars, labels = build_exemplar_inputs(batch, device)
                    has_any = any(ex.shape[0] > 0 for ex in exemplars)
                    if has_any:
                        outputs: Dict[str, Any] = model(images, prompts=prompts, images_swin=images_swin, exemplars=exemplars, labels=labels)
                    else:
                        outputs: Dict[str, Any] = model(images, prompts=prompts, images_swin=images_swin)
                else:
                    outputs: Dict[str, Any] = model(images, prompts=prompts, images_swin=images_swin)

                outputs.pop("hard_counts", None)

                # 2. Tính Loss bằng SetCriterion
                loss_dict = criterion(outputs, targets, cat_list, prompts)
                weight_dict = criterion.weight_dict
                loss_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # Gán lại các key cho đồng nhất với LOSS_KEYS để log
                loss_dict["loss_total"] = loss_total
                loss_dict["loss_loc"] = loss_dict.get("loss_bbox", torch.tensor(0.0))
                loss_dict["loss_cls"] = loss_dict.get("loss_ce", torch.tensor(0.0))

                # Count the total query object (MAE Count loss - Không backprop)
                pred_counts_raw = torch.relu(outputs["query_logits"]).sum(dim=1)
                loss_dict["count_mae"] = (pred_counts_raw - gt_counts).abs().mean()

            for k in LOSS_KEYS:
                v = loss_dict.get(k, torch.tensor(0.0, device=device)).item()
                loss_sums[k] += v * batch_size

            pred_counts = count_by_det_nms(outputs, threshold=args.threshold, nms_iou=args.nms_iou).to(gt_counts.dtype)
            
            # --- UPDATE: Lưu chi tiết cho log_one_epoch ---
            img_ids = batch.get("image_id", batch.get("image_path", [f"batch_{step}_idx_{j}" for j in range(batch_size)]))
            
            for j in range(batch_size):
                p_cnt = pred_counts[j].item()
                g_cnt = gt_counts[j].item()
                
                img_id = img_ids[j]
                if torch.is_tensor(img_id): img_id = img_id.item()
                
                predictions_list.append({
                    "image_id": img_id,
                    "gt_count": g_cnt,
                    "pred_count": p_cnt,
                    "abs_err_mae": abs(p_cnt - g_cnt),
                    "sq_err_mse": (p_cnt - g_cnt) ** 2
                })
            # ----------------------------------------------

            diff = (pred_counts - gt_counts).abs()
            mae_sum += diff.sum().item()
            mse_sum += (diff ** 2).sum().item()
            n_count_samples += gt_counts.numel()

            avg_now = _avg_losses(loss_sums, num_samples)
            pbar.set_postfix({
                "L_tot": f"{avg_now['loss_total']:.3f}",
                "L_loc": f"{avg_now['loss_loc']:.3f}",
                "MAE": f"{avg_now.get('count_mae', 0.0):.3f}",
            })

    avg_losses = _avg_losses(loss_sums, num_samples)
    if n_count_samples > 0:
        mae = mae_sum / n_count_samples
        rmse = math.sqrt(mse_sum / n_count_samples)
    else:
        mae, rmse = 0.0, 0.0
        
    return avg_losses, {"mae": mae, "rmse": rmse}, predictions_list


# ========================
#  CHECKPOINT + CSV
# ========================

def save_clean_checkpoint(model, optimizer, scheduler, scaler, epoch, save_dir, mode="last"):
    os.makedirs(save_dir, exist_ok=True)
    for f in os.listdir(save_dir):
        if f.startswith(f"{mode}_epoch_") and f.endswith(".pth"):
            os.remove(os.path.join(save_dir, f))
            
    ckpt_name = f"{mode}_epoch_{epoch}.pth"
    save_path = os.path.join(save_dir, ckpt_name)
    
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(state, save_path)
    print(f" >>> Saved {mode} checkpoint: {ckpt_name}")


def load_checkpoint(ckpt_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: GradScaler, device: str) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1


def init_csv_logger(log_dir: str = "logs", filename: str = "train_log.csv") -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "train_loss_total","train_count_mae","train_loss_ce","train_loss_bbox","train_loss_giou","train_loss_query",
                "val_loss_total","val_count_mae","val_loss_ce","val_loss_bbox","val_loss_giou","val_loss_query",
                "val_mae","val_rmse",
            ])
    return path


def append_csv_log(csv_path: str, epoch: int, train_losses: Dict[str, float], val_losses: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            train_losses.get("loss_total",0.0), train_losses.get("count_mae", 0.0),
            train_losses.get("loss_ce",0.0), train_losses.get("loss_bbox",0.0),
            train_losses.get("loss_giou",0.0), train_losses.get("loss_query",0.0),

            val_losses.get("loss_total",0.0), val_losses.get("count_mae", 0.0),
            val_losses.get("loss_ce",0.0), val_losses.get("loss_bbox",0.0),
            val_losses.get("loss_giou",0.0), val_losses.get("loss_query",0.0),

            val_metrics.get("mae",0.0), val_metrics.get("rmse",0.0),
        ])


# ========================
#  MAIN
# ========================

def main():
    args = parse_args()
    exp = args.exp_name.strip() or ("text_exemplar" if args.use_exemplar else "text_only")

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"
    print(f"Using device: {device}")

    train_loader = create_dataloader(args, split="train")
    val_loader = create_dataloader(args, split="val")

    model = create_model(args).to(device)
    print_model_parameter_summary(model)

    optimizer, scheduler = create_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.startswith("cuda")))

    best_mae = float("inf")
    start_epoch = 1

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
    elif args.log_one_epoch:
        print("\n[WARNING]đang bật --log_one_epoch nhưng không truyền --resume. Model sẽ chạy từ scratch/pretrained!\n")

    if args.start_epoch is not None and args.start_epoch > 0:
        start_epoch = args.start_epoch

    print(f"Start epoch: {start_epoch}")

    # ==========================================
    # INITIALIZE CRITERION 
    # ==========================================
    matcher = HungarianMatcher(
        cost_class=args.lambda_cls if hasattr(args, 'lambda_cls') else 5.0, 
        cost_bbox=args.lambda_loc if hasattr(args, 'lambda_loc') else 1.0, 
        cost_giou=0.0
    )
    
    weight_dict = {
        'loss_ce': args.lambda_cls if hasattr(args, 'lambda_cls') else 5.0, 
        'loss_bbox': args.lambda_loc if hasattr(args, 'lambda_loc') else 1.0, 
        'loss_giou': 0.0
    }
    
    losses = ['labels', 'boxes']
    criterion = SetCriterion(
        matcher=matcher, 
        weight_dict=weight_dict, 
        focal_alpha=0.25, 
        focal_gamma=2.0, 
        losses=losses
    ).to(device)

    # ==========================================
    # MODE LOG ONE EPOCH (TRAIN 1 EPOCH, VAL, TEST, EXIT)
    # ==========================================
    if args.log_one_epoch:
        print("\n" + "="*60)
        print("🚀 RUNNING IN --log_one_epoch MODE")
        print("="*60)
        
        # Load thêm test_loader
        test_loader = create_dataloader(args, split="test")
        
        print("Extracting Train details...")
        _, train_metrics, train_preds = evaluate(start_epoch, model, criterion, train_loader, device, args, split_name="train")
        
        # 2. Eval Val
        _, val_metrics, val_preds = evaluate(start_epoch, model, criterion, val_loader, device, args, split_name="val")
        
        # 3. Eval Test
        _, test_metrics, test_preds = evaluate(start_epoch, model, criterion, test_loader, device, args, split_name="test")
        
        # 4. Ghi toàn bộ lịch sử ra CSV
        csv_path = "log_one_epoch.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image_id", "gt_count", "pred_count", "abs_err_MAE", "sq_err_MSE"])
            
            # Ghi cả 3 tập
            for p in train_preds:
                writer.writerow(["train", p["image_id"], p["gt_count"], p["pred_count"], p["abs_err_mae"], p["sq_err_mse"]])
            for p in val_preds:
                writer.writerow(["val", p["image_id"], p["gt_count"], p["pred_count"], p["abs_err_mae"], p["sq_err_mse"]])
            for p in test_preds:
                writer.writerow(["test", p["image_id"], p["gt_count"], p["pred_count"], p["abs_err_mae"], p["sq_err_mse"]])
            
            # Summary cho cả 3
            writer.writerow([])
            writer.writerow(["SUMMARY_TRAIN", "MAE", train_metrics["mae"], "RMSE", train_metrics["rmse"]])
            writer.writerow(["SUMMARY_VAL", "MAE", val_metrics["mae"], "RMSE", val_metrics["rmse"]])
            writer.writerow(["SUMMARY_TEST", "MAE", test_metrics["mae"], "RMSE", test_metrics["rmse"]])
            
        print(f"\n✅ Đã lưu kết quả log của 3 tập Train, Val và Test vào: {csv_path}")
        print("="*60)
        return  

    # ==========================================
    # TRAIN
    # ==========================================
    # test evaluate() first
    val_losses, val_metrics, _ = evaluate(
            epoch=1, model=model, criterion=criterion, loader=val_loader, device=device, args=args, split_name="val"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_losses = train_one_epoch(
            epoch=epoch, model=model, criterion=criterion, loader=train_loader, optimizer=optimizer,
            scaler=scaler, device=device, log_interval=args.log_interval, args=args,
        )

        val_losses, val_metrics, _ = evaluate(
            epoch=epoch, model=model, criterion=criterion, loader=val_loader, device=device, args=args, split_name="val"
        )

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Train Loss: {train_losses['loss_total']:.4f} | MAE: {train_losses['count_mae']:.2f}")
        # soft count
        print(f"Val Loss: {val_losses['loss_total']:.4f} | MAE (soft count): {val_losses['count_mae']:.2f}")
        # hard count sau khi áp dụng nms
        print(f"Val MAE (nms): {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f}\n")

        save_dir = os.path.join(args.save_dir, exp)
        os.makedirs(save_dir, exist_ok=True)

        csv_path = init_csv_logger(log_dir="logs", filename=f"train_log_{exp}.csv")
        append_csv_log(csv_path, epoch, train_losses, val_losses, val_metrics)

        save_clean_checkpoint(model, optimizer, scheduler, scaler, epoch, save_dir, mode="last")

        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            save_clean_checkpoint(model, optimizer, scheduler, scaler, epoch, save_dir=save_dir, mode="best")

        scheduler.step()

if __name__ == "__main__":
    main()