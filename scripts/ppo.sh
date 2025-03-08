#!/usr/bin/env bash

ACTOR_MODEL_NAME_OR_PATH="" # model path
REWARD_MODEL_NAME_OR_PATH="" # trained reward model path
CRITIC_MODEL_NAME_OR_PATH=""

TRAIN_DATASETS=""
TRAIN_TEMPLATE="" # dataset template
TRAIN_NAME="" # subset the dataset
TRAIN_SPLIT="" # split the dataset

PTX_DATASETS=""
PTX_TEMPLATE="" # dataset template
PTX_SPLIT="" # split the dataset

OUTPUT_DIR="" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
    --module safe_rlhf_v.trainers.text_image_to_text.ppo \
    --actor_model_name_or_path ${ACTOR_MODEL_NAME_OR_PATH} \
    --reward_model_name_or_path ${REWARD_MODEL_NAME_OR_PATH} \
    --reward_critic_model_name_or_path ${CRITIC_MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_template ${TRAIN_TEMPLATE} \
    --train_name ${TRAIN_NAME} \
    --train_split ${TRAIN_SPLIT} \
    --ptx_datasets ${PTX_DATASETS} \
    --ptx_template ${PTX_TEMPLATE} \
    --ptx_split ${PTX_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --save_interval 1000 \
    --epochs 3