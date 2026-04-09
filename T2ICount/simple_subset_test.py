import os.path

from models.reg_model import Count
import torch
import numpy as np
from utils.tools import setup_seed, extract_patches, reassemble_patches
from transformers import CLIPTokenizer
from torchvision import transforms
from PIL import Image
import json


with open('FSC-147-S.json', 'r') as f:
    data = json.load(f)


setup_seed(15)

config = 'configs/v1-inference.yaml'
sd_path = 'configs/v1-5-pruned-emaonly.ckpt'
crop_size = 384
model = Count(config, sd_path, unet_config={'base_size': crop_size, 'max_attn_size': crop_size // 8,
                                            'attn_selector': 'down_cross+up_cross'})

model.load_state_dict(torch.load('best_model_paper.pth', map_location='cpu')) # Change the path to the pretrained weight
model = model.to('cuda')

error = []
for img_file in data.keys():
    gt = data[img_file]['count']
    cls_name = data[img_file]['class']
    prompt_attn_mask = torch.zeros(77)
    cls_name_tokens = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")(cls_name, add_special_tokens=False,
                                                                                      return_tensors='pt')
    cls_name_length = cls_name_tokens['input_ids'].shape[1]
    prompt_attn_mask[1: 1 + cls_name_length] = 1
    prompt_attn_mask = prompt_attn_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).to('cuda')
    cls_name = (cls_name,)
    img_path = 'data/FSC/images_384_VarV2/' + img_file

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    im = Image.open(img_path).convert('RGB')
    im = transform(im).unsqueeze(0).to('cuda')

    cropped_imgs, num_h, num_w = extract_patches(im, patch_size=384, stride=384)
    outputs = []

    with torch.set_grad_enabled(False):
        num_chunks = (cropped_imgs.size(0) + 4 - 1) // 4
        for i in range(num_chunks):
            start_idx = i * 4
            end_idx = min((i + 1) * 4, cropped_imgs.size(0))
            outputs_partial = model(cropped_imgs[start_idx:end_idx], cls_name * (end_idx - start_idx),
                                    prompt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
            outputs.append(outputs_partial)
        results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, im.size(2), im.size(3),
                                         patch_size=384, stride=384).detach().cpu().squeeze(0).squeeze(0) / 60

        error.append(abs(gt - results.sum().item()))

mae = np.array(error).mean()
mse = np.sqrt(np.mean(np.square(error)))
print('MAE:', mae, 'MSE:', mse)

