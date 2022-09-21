#!/bin/bash 

dataset='Chimpanzee'

##### GPU System #####
GPU_ID=0

#### Iteration #####
MBW_Iteration=1

##### 2D Detector #####
config_file=configs/detector.yaml
scale_factor=200
TRAIN_END_EPOCH=200
img_type=.jpg
leverage_prior_object_knowledge='True'
unittest='True'

# echo "*********** Training 2D Detector ****************"
CUDA_VISIBLE_DEVICES=$GPU_ID python modules/detector/detector.py \
         --cfg $config_file \
         --dataset=$dataset \
         --scale_factor=$scale_factor \
         --MBW_Iteration=$MBW_Iteration \
         --TRAIN_END_EPOCH=$TRAIN_END_EPOCH \
         --img_type=$img_type \
         --leverage_prior_object_knowledge=$leverage_prior_object_knowledge \
         --unittest=$unittest \
         --datasetunittest=$1
