#!/bin/bash 

dataset='Human36M'

##### GPU System #####
GPU_ID=1

#### Iteration #####
MBW_Iteration=0

##### MV-NRSfM #####
mode=train
batch=1000
model=models/mvnrsfm/$dataset
dataset_metric_multiplier=1000
break_training_counter=10000
from_scratch='False'
unittest='False'

CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 modules/mvnrsfm/mvnrsfm_train.py 	--mode=$mode \
														--dataset=$dataset \
														--batch=$batch \
														--model=$model \
														--dataset_metric_multiplier=$dataset_metric_multiplier \
														--MBW_Iteration=$MBW_Iteration \
														--break_training_counter=$break_training_counter \
														--from_scratch=$from_scratch \
														--unittest=$unittest \
														--datasetunittest=$1

