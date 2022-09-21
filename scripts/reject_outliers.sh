#!/bin/bash 

dataset='Chimpanzee'

##### GPU System #####
GPU_ID=0

#### Iteration #####
MBW_Iteration=1

##### Rejection 
outlier_threshold=0.05
break_training_counter=10000
input_source='Detector'
batch=1000
dataset_metric_multiplier=1000
trust_MVNRSfM_Inliers='False'
from_scratch='False'
update_pickle='False'
leave_bbox_out='False'
update_bbox_only='True'
model=models/mvnrsfm/$dataset
unittest='False'
img_type=.jpg

CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 modules/mvnrsfm/mvnrsfm_outlier_rejection.py 	--mode=$mode \
														--dataset=$dataset \
														--batch=$batch \
														--model=$model \
														--dataset_metric_multiplier=$dataset_metric_multiplier \
														--MBW_Iteration=$MBW_Iteration \
														--outlier_threshold=$outlier_threshold \
														--input_source=$input_source \
														--break_training_counter=$break_training_counter \
														--trust_MVNRSfM_Inliers=$trust_MVNRSfM_Inliers \
														--from_scratch=$from_scratch \
														--update_pickle=$update_pickle \
														--leave_bbox_out=$leave_bbox_out \
														--update_bbox_only=$update_bbox_only \
														--unittest=$unittest \
														--img_type=$img_type \
														--datasetunittest=$1
