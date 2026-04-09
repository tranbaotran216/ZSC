import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
import gc
import atexit
from torchvision.ops import box_convert, nms # Thêm NMS để lọc bbox

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    
sys.path.insert(0, r"D:\projects\zsc")
sys.path.insert(1, r"D:\projects\zsc\T2ICount")

from src.helpers import sanitize_caption, count_by_det_nms

CKPT_PATH = "D:/projects/zsc/checkpoints_dgdcount/unet_gdino_text_only/best_epoch_17.pth"
GD_CONFIG = "D:/projects/zsc/gdcount/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CKPT = "D:/projects/zsc/gdcount/weights/groundingdino_swint_ogc.pth"
SD_CONFIG = "D:/projects/zsc/T2ICount/configs/v1-inference.yaml"
SD_CKPT = "D:/projects/zsc/T2ICount/configs/v1-5-pruned-emaonly.ckpt"
PATH_V1 = "D:/projects/zsc/checkpoints_dgdcount/unet_gdino_text_only"
PATH_V2 = "D:/projects/zsc/checkpoints_dgdcount/model_arch_ver_2/unet_gdino_text_only_arch_ver_2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
current_model = None
current_model_version = None

def clean_vram():
    global current_model, current_model_version
    if current_model is not None:
        print("----- cleaning vram -----")
        del current_model
        current_model = None
        current_model_version = None
    
    gc.collect() # force python to clean trash

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("torch cache cleaned!")

atexit.register(clean_vram)


def load_model_func(version, checkpoint_name):
    """Hàm để load model dựa trên lựa chọn ở sidebar"""
    global current_model, current_model_version
    if current_model is not None and current_model_version == version:
        return f"Model {current_model_version} is loaded, ready to count"
    
    clean_vram()

    if not checkpoint_name:
        return "Vui lòng chọn file checkpoint!"
        
    print(f"Loading {version} with checkpoint {checkpoint_name}...")
    
    base_dir = PATH_V1 if version == "DGD-Model V1" else PATH_V2
    ckpt_full_path = os.path.join(base_dir, checkpoint_name)
    
    if not os.path.exists(ckpt_full_path):
        return f"Lỗi: Không tìm thấy file {ckpt_full_path}"
    
    if version == "DGD-Model V1":
        from models import dgd_model
        cfg = dgd_model.GDinoConfig(threshold=0.23, hidden_dim=256)
        model_arch = dgd_model.build_dgd_model(
            GD_CONFIG, GD_CKPT, SD_CONFIG, SD_CKPT,
            unet_config={'base_size': 384, 'max_attn_size': 384 // 8, 'attn_selector': 'down_cross+up_cross'}, 
            gdconfig=cfg, device=DEVICE
        ) 
    else:
        from models import model_ver2
        cfg = model_ver2.GDinoConfig(threshold=0.23, hidden_dim=256)
        model_arch = model_ver2.build_dgd_model(
            GD_CONFIG, GD_CKPT, SD_CONFIG, SD_CKPT,
            unet_config={'base_size': 384, 'max_attn_size': 384 // 8, 'attn_selector': 'down_cross+up_cross'}, 
            gdconfig=cfg, device=DEVICE
        ) 
        
    model_arch.to(DEVICE)
    ckpt = torch.load(ckpt_full_path, map_location=DEVICE)
    model_arch.load_state_dict(ckpt['model'])
    model_arch.eval()
    
    current_model = model_arch
    current_model_version = version
    return f"Đã load xong {version} với {checkpoint_name}!"


def predict(input_img, prompt, model_ver, ckpt_name):
    global current_model
    if input_img is None or prompt == "":
        return None, 0
    
    if current_model is None:
        load_model_func(model_ver, ckpt_name)
        
    # Tiền xử lý ảnh
    orig_w, orig_h = input_img.size
    img_resized = input_img.resize((384, 384), Image.BICUBIC)
    img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    img_tensor = (img_tensor - 0.5) / 0.5 # Chuẩn hóa [-1, 1] cho VAE

    # Inference
    with torch.no_grad():
        outputs = current_model(img_tensor, prompts=[sanitize_caption(prompt)])
    
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_bboxes"][0]
    
    if logits.ndim == 1:
        mask = logits > 0.23
        scores = logits[mask]
    else:
        max_scores, _ = logits.max(dim=1)
        mask = max_scores > 0.23
        scores = max_scores[mask]
        
    final_boxes = boxes[mask]
    
    # Áp dụng NMS để tránh trùng lặp bbox
    if len(final_boxes) > 0:
        # GroundingDINO trả về dạng [cx, cy, w, h], NMS yêu cầu [x1, y1, x2, y2]
        boxes_xyxy = box_convert(final_boxes, 'cxcywh', 'xyxy')
        keep_idx = nms(boxes_xyxy, scores, iou_threshold=0.5)
        final_boxes = final_boxes[keep_idx]
    
    # Vẽ point (điểm tròn) lên ảnh thay vì bbox
    draw = ImageDraw.Draw(input_img)
    count = 0
    
    # Bán kính của điểm (point radius), bạn có thể chỉnh to nhỏ ở đây
    r = 4 
    
    for box in final_boxes:
        cx, cy, w, h = box.tolist()
        
        # Tính toán tọa độ pixel thực tế của tâm
        center_x = cx * orig_w
        center_y = cy * orig_h
        
        # Vẽ một hình tròn (ellipse) bao quanh tọa độ tâm
        draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r], 
            fill="red", outline="white"
        )
        count += 1
        
    return input_img, count


def get_checkpoint_list(version):
    path = PATH_V1 if version == "DGD-Model V1" else PATH_V2
    if os.path.exists(path):
        ckpts = [f for f in os.listdir(path) if f.endswith('.pth')]
        return gr.update(choices=ckpts, value=ckpts[0] if ckpts else None)
    return gr.update(choices=["No checkpoints found"], value=None)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DGD-Count: Zero-shot Counting")
    
    with gr.Sidebar():
        gr.Markdown("### ⚙️ Cấu hình Model")
        
        # Ô chọn Version
        model_version = gr.Dropdown(
            choices=["DGD-Model V1", "DGD-Model V2"], 
            value="DGD-Model V1", 
            label="Model Architecture"
        )
        
        # Ô chọn Checkpoint (khởi tạo rỗng hoặc theo V1 mặc định)
        init_ckpts = [f for f in os.listdir(PATH_V1) if f.endswith('.pth')] if os.path.exists(PATH_V1) else []
        checkpoint = gr.Dropdown(
            choices=init_ckpts, 
            value=init_ckpts[0] if init_ckpts else None, 
            label="Checkpoint (.pth)"
        )
        
        load_btn = gr.Button("Cập nhật Model", variant="primary")
        status_msg = gr.Textbox(label="Trạng thái", interactive=False)

        model_version.change(
            fn=get_checkpoint_list,
            inputs=model_version,
            outputs=checkpoint
        )
        
        # Nút load model
        load_btn.click(
            fn=load_model_func,
            inputs=[model_version, checkpoint], 
            outputs=status_msg
        )

    # KHU VỰC CHÍNH (MAIN AREA)
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(
                placeholder="What to count? (e.g. apple, bird...)", 
                label="Prompt"
            )
            run_btn = gr.Button("Bắt đầu đếm", variant="primary")
            
        with gr.Column():
            img_output = gr.Image(type="pil", label="Result")
            count_output = gr.Number(label="Object Count")

    # Xử lý sự kiện khi nhấn nút Run
    run_btn.click(
        fn=predict,
        inputs=[img_input, prompt_input, model_version, checkpoint],
        outputs=[img_output, count_output]
    )

if __name__ == "__main__":
    demo.launch()