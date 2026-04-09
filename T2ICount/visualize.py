from models.reg_model import Count
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from utils.regression_trainer import setup_seed
from transformers import CLIPTokenizer
from torchvision import transforms
from PIL import Image
from utils.tools import extract_patches, reassemble_patches
import cv2
import torch.nn.functional as F

import sys
sys.path.append("D:/projects/zsc/T2ICount")
sys.path.append("D:/projects/zsc")
setup_seed(15)

config = 'configs/v1-inference.yaml'
sd_path = 'configs/v1-5-pruned-emaonly.ckpt'
crop_size = 384
model = Count(config, sd_path, unet_config={'base_size': crop_size, 'max_attn_size': crop_size // 8,
                                            'attn_selector': 'down_cross+up_cross'})
model.load_state_dict(torch.load('checkpoint/best_model_paper.pth', map_location='cpu'), strict=False)
model = model.to('cuda')
img_path = r"D:\projects\zsc\data\demo_imgs\birds and butterflies.jpg"
cls_name = 'birds'

attention_mask = torch.zeros(77)
cls_name_tokens = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")(cls_name, add_special_tokens=False,
                                                                                 return_tensors='pt')
cls_name_length = cls_name_tokens['input_ids'].shape[1]
attention_mask[1: 1 + cls_name_length] = 1
attention_mask = attention_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).to('cuda')
cls_name = (cls_name,)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
im = Image.open(img_path).convert('RGB')
im = transform(im).unsqueeze(0).to('cuda')

h, w = im.shape[2], im.shape[3]
pad_h = (384 - h % 384) % 384
pad_w = (384 - w % 384) % 384

im_padded = F.pad(im, (0, pad_w, 0, pad_h), mode='constant', value=0)

cropped_imgs, num_h, num_w = extract_patches(im_padded, patch_size=384, stride=384)

outputs = []

with torch.set_grad_enabled(False):
    num_chunks = (cropped_imgs.size(0) + 4 - 1) // 4
    for i in range(num_chunks):
        start_idx = i * 4
        end_idx = min((i + 1) * 4, cropped_imgs.size(0))
        outputs_partial, sim_x1, sim_x2, _ = model(cropped_imgs[start_idx:end_idx], cls_name * (end_idx - start_idx),
                                attention_mask.repeat((end_idx - start_idx), 1, 1, 1))
        outputs.append(outputs_partial)

    results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, 
                                 im_padded.size(2), im_padded.size(3),
                                 patch_size=384, stride=384).detach().cpu().squeeze(0).squeeze(0) / 60
    
    results = results[:h, :w]

    pred_density = results.numpy()
    print('Predicted Number:', pred_density.sum())
    pred_density = pred_density / pred_density.max()
    pred_density_write = 1. - pred_density
    pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
    pred_density_write = pred_density_write / 255
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    heatmap_pred = 0.33 * img + 0.67 * pred_density_write
    heatmap_pred = heatmap_pred / heatmap_pred.max()
    cv2.imwrite('den.jpg', cv2.cvtColor((heatmap_pred * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

