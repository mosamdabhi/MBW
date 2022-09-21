#!/bin/bash 

# Chimpanzee  Colobus_Monkey  Fish          Seahorse
# Clownfish	  Flamingo        Human36M      Tiger

dataset='Chimpanzee'
##### GPU System #####
GPU_ID=1

#### Iteration #####
MBW_Iteration=6


######################### VISUALIZATIONS "2D Predictions" and "BBox" #########################
field_of_validation='Predictions' # Options: "Inliers" | "Flow" | "MV_Train" | "MV" | "Detector"
mode_of_validation='2D'
plot_separate='False'

CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 common/visualize.py --dataset=$dataset \
														--mode_of_validation=$mode_of_validation \
														--field_of_validation=$field_of_validation \
														--MBW_Iteration=$MBW_Iteration \
														--plot_separate=$plot_separate

field_of_validation='Predictions' # Options: "Inliers" | "Flow" | "MV_Train" | "MV" | "Detector"
mode_of_validation='BBox'
plot_separate='False'

CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 common/visualize.py --dataset=$dataset \
														--mode_of_validation=$mode_of_validation \
														--field_of_validation=$field_of_validation \
														--MBW_Iteration=$MBW_Iteration \
														--plot_separate=$plot_separate
