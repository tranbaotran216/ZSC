import torch
from models.reg_model import Count
from datasets.dataset import ObjectCount
from datasets.carpk import CARPK
import numpy as np
from utils.tools import extract_patches, reassemble_patches
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default="./best_model_pater.pth", type=str,
                        help='what is it?')
    parser.add_argument('--data', default="carpk", type=str,)
    parser.add_argument('--batch-size', default=16, type=int, )
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_arg()
    unet_config = {'base_size': 384,
                   'max_attn_size': 384 // 8,
                   'attn_selector': 'down_cross+up_cross'}

    batch_size = args.batch_size
    crop_size = 384

    model = Count('configs/v1-inference.yaml',
                  'configs/v1-5-pruned-emaonly.ckpt', unet_config=unet_config)

    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    assert args.data in ['carpk', 'fsc147'], f"Invalid dataset: {args.data}. Must be 'carpk' or 'fsc147'."
    if args.data == 'carpk':
        data = CARPK('data/CARPK', 'data/CARPK/ImageSets/test.txt')
    else:
        data = ObjectCount('data/FSC', crop_size, 8, 'test', 224)

    dataloader = torch.utils.data.DataLoader(data, batch_size=1)
    device = torch.device('cuda')
    model = model.to(device)
    model.set_eval()
    epoch_res = []
    for step, (img, gt_counts, prompts, prompt_attn_mask, name) in enumerate(dataloader):
        inputs = img.to(device)
        gt_prompt_attn_mask = prompt_attn_mask.to(device).unsqueeze(2).unsqueeze(3)
        cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=crop_size, stride=crop_size)
        outputs = []
        with torch.set_grad_enabled(False):
            num_chunks = (cropped_imgs.size(0) + batch_size - 1) // batch_size
            for i in range(num_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, cropped_imgs.size(0))
                outputs_partial = model(cropped_imgs[start_idx:end_idx], prompts * (end_idx - start_idx), gt_prompt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
                outputs.append(outputs_partial)
            results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                                         patch_size=crop_size, stride=crop_size) / 60
            res = gt_counts[0].item() - torch.sum(results).item()
            results = results.squeeze(0).squeeze(0).detach().cpu().numpy()
            epoch_res.append(res)
    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    print(f'Test, MAE:{mae}, MSE:{mse}')

