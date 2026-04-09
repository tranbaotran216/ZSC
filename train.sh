#!/bin/bash

source /d/miniconda3/etc/profile.d/conda.sh
conda activate zsc

which python
python -V

PROJECT_DIR="D:/projects/zsc" 
export PYTHONPATH="$PROJECT_DIR;$PROJECT_DIR/gdcount/groundingdino;$PROJECT_DIR/T2ICount;$PYTHONPATH"

echo "PYTHONPATH=$PYTHONPATH"

# Dataset
DATA_ROOT="$PROJECT_DIR/data"
ANN_FILE="FSC147/annotation_FSC147_384.json"
IMG_ROOT="images_384_VarV2"
SPLIT_FILE="FSC147/Train_Test_Val_FSC_147.json"
CLASS_MAP="FSC147/ImageClasses_FSC147.txt"

# Weights & Configs
GD_CONFIG="$PROJECT_DIR/gdcount/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CKPT="$PROJECT_DIR/gdcount/weights/groundingdino_swint_ogc.pth"
SD_CONFIG="$PROJECT_DIR/T2ICount/configs/v1-inference.yaml"
SD_CKPT="$PROJECT_DIR/T2ICount/configs/v1-5-pruned-emaonly.ckpt"


# HYPERPARAMETERS 
BATCH_SIZE=4      
ACCUM_STEPS=2     
EPOCHS=100
LR=1e-4
DEVICE="cuda"   
MODEL_ARCH_VER=2  


SAVE_DIR="$PROJECT_DIR/checkpoints_dgdcount/model_arch_ver_$MODEL_ARCH_VER"
EXP_NAME="unet_gdino_text_only_arch_ver_$MODEL_ARCH_VER"

mkdir -p ${SAVE_DIR}

export CUDA_VISIBLE_DEVICES=0 


echo "==================================================="
echo "BẮT ĐẦU HUẤN LUYỆN DGD-COUNT (U-Net + GroundingDINO) model arch version ${MODEL_ARCH_VER}"
echo "Experiment: $EXP_NAME"
echo "Device: $DEVICE (GPU ID: $CUDA_VISIBLE_DEVICES)"
echo "Batch Size: $BATCH_SIZE | Accumulation Steps: $ACCUM_STEPS"
echo "==================================================="

python -u "$PROJECT_DIR/train.py" \
    --data-root "$DATA_ROOT" \
    --ann "$ANN_FILE" \
    --img-root "$IMG_ROOT" \
    --split-file "$SPLIT_FILE" \
    --class-map "$CLASS_MAP" \
    --gd_config_path "$GD_CONFIG" \
    --gd_ckpt_path "$GD_CKPT" \
    --sd_config_path "$SD_CONFIG" \
    --sd_ckpt_path "$SD_CKPT" \
    --save-dir "$SAVE_DIR" \
    --exp-name "$EXP_NAME" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accum-steps $ACCUM_STEPS \
    --lr $LR \
    --device "$DEVICE" \
    --amp \
    --log-interval 50 \
    --train_version $MODEL_ARCH_VER \
    --resume D:/projects/zsc/checkpoints_dgdcount/model_arch_ver_2/unet_gdino_text_only_arch_ver_2/best_epoch_1.pth \
    2>&1 | tee "$SAVE_DIR/$EXP_NAME.log"

echo "Training finished!"