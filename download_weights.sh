#!/bin/bash

mkdir -p countgd/weights
mkdir -p T2ICount/configs

echo "--- Đang tải Grounding DINO weights ---"
wget -O countgd/weights/groundingdino_swint_ogc.pth \
"https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"

echo "--- Đang tải Stable Diffusion weights ---"
wget -O T2ICount/configs/v1-5-pruned-emaonly.ckpt \
"https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"

echo "--- Tải weights hoàn tất! ---"