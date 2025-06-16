#!/usr/bin/env bash

MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf" # model path

TRAIN_DATASETS="saferlhf-v/BeaverTails-V"
TRAIN_TEMPLATE="DPO_Harmless" # dataset template
TRAIN_NAME="animal_abuse" # subset the dataset
TRAIN_SPLIT="train" # split the dataset

OUTPUT_DIR="../outputs/dpo" # output dir

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