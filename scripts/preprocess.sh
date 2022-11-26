#!/bin/bash

dataset='train'
percentage_if_gt=5

##### GPU System #####
GPU_ID=0

#### Iteration #####
MBW_Iteration=0

## Annotation server ##
host=localhost
port=8980

unittest="False"

## Preprocess
CUDA_VISIBLE_DEVICES=$GPU_ID python common/annotate.py \
                                --dataset=$dataset \
                                --percentage_if_gt=$percentage_if_gt \
                                --host=$host \
                                --port=$port \
                                --unittest=$unittest \
                                --datasetunittest=$1
