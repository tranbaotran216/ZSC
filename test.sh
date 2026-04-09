#!/bin/bash

source /d/miniconda3/etc/profile.d/conda.sh
conda activate zsc

which python
python -V

PROJECT_DIR="D:/projects/zsc" 
export PYTHONPATH="$PROJECT_DIR;$PROJECT_DIR/gdcount/groundingdino;$PROJECT_DIR/T2ICount;$PYTHONPATH"

echo "PYTHONPATH=$PYTHONPATH"
DATA_ROOT="$PROJECT_DIR/data"

# Thay đổi đường dẫn checkpoint thực tế của bạn tại đây
CHECKPOINT="$PROJECT_DIR/checkpoints_dgdcount/model_arch_ver_2/unet_gdino_text_only_arch_ver_2/best_epoch_15.pth"


python "$PROJECT_DIR/test.py" \
    --ckpt "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --device "cuda" \
    --threshold 0.23 \
    --nms_iou 0.5 \
    --train_version 2