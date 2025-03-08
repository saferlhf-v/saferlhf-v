#!/usr/bin/env bash

MODEL_NAME_OR_PATH="" # model path

TRAIN_DATASETS=""
TRAIN_TEMPLATE="" # dataset template
TRAIN_NAME="" # subset the dataset
TRAIN_SPLIT="" # split the dataset

OUTPUT_DIR="" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --module safe_rlhf_v.trainers.text_image_to_text.dpo \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_name ${TRAIN_NAME} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 1000 \
     --epochs 3