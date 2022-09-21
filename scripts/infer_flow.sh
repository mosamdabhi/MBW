#!/bin/bash 

dataset='Chimpanzee'

##### GPU System #####
GPU_ID=0

#### Iteration #####
MBW_Iteration=0

##### Flow #####
model=models/flow/flow-pretrained.pth
flow_iters=20
img_type=.jpg
log_dir=logs/flow/$dataset
to_plot_results="False"
unittest="True"


CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python modules/flow/flow_infer.py \
									--model=$model \
									--flow_iters=$flow_iters \
                                    --dataset=$dataset \
                                    --img_type=$img_type \
                                    --log_dir=$log_dir \
                                    --MBW_Iteration=$MBW_Iteration \
                                    --to_plot_results=$to_plot_results \
                                    --unittest=$unittest \
                                    --datasetunittest=$1