#!/bin/bash 

dataset='Chimpanzee'

##### GPU System #####
GPU_ID=1

##### 2D Detector #####
config_file=configs/detector.yaml
scale_factor=200
leverage_prior_object_knowledge='True'
unittest='False'
pretrain_model_dataset=$dataset
img_type=.jpg

# echo "*********** Evaluation mode ****************"
CUDA_VISIBLE_DEVICES=$GPU_ID python modules/detector/eval.py \
         --cfg $config_file \
         --dataset=$dataset \
         --scale_factor=$scale_factor \
         --MBW_Iteration=1 \
         --leverage_prior_object_knowledge=$leverage_prior_object_knowledge \
         --unittest=$unittest \
         --datasetunittest=$1 \
         --img_type=$img_type \
         --pretrain_model_dataset=$pretrain_model_dataset
