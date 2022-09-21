#!/bin/bash 

dataset='Chimpanzee'
GPU_ID=0
field_of_validation='Inliers' # Options: "Inliers" | "Flow" | "MV_Train" | "MV" | "Detector"
mode_of_validation='BBox'
MBW_Iteration=1
plot_separate='False'
validate_manual_labels='False'
img_type=.jpg


CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 common/visualize.py --dataset=$dataset \
														--mode_of_validation=$mode_of_validation \
														--field_of_validation=$field_of_validation \
														--MBW_Iteration=$MBW_Iteration \
														--plot_separate=$plot_separate \
														--validate_manual_labels=$validate_manual_labels \
														--img_type=$img_type
