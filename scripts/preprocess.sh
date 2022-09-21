#!/bin/bash

dataset='Chimpanzee'
percentage_if_gt=5

##### GPU System #####
GPU_ID=0

#### Iteration #####
MBW_Iteration=0

## Annotation server ##
host=localhost
port=8980

unittest="True"

## Preprocess
CUDA_VISIBLE_DEVICES=$GPU_ID python common/preprocess.py \
                                --dataset=$dataset \
                                --percentage_if_gt=$percentage_if_gt \
                                --host=$host \
                                --port=$port \
                                --unittest=$unittest \
                                --datasetunittest=$1
