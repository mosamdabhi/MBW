#!/bin/bash 

dataset='Chimpanzee'
percentage_if_gt=5

##### GPU System #####
GPU_ID=1

#### Iteration #####
MBW_Iteration=0

## Annotation server ##
host=localhost
port=8980

# Preprocess
CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 common/preprocess.py \
                               --dataset=$dataset \
                               --percentage_if_gt=$percentage_if_gt \
                               --host=$host \
                               --port=$port



##### Flow #####
model=models/flow/flow-pretrained.pth
flow_iters=20
img_type=.jpg
log_dir=logs/flow/$dataset
to_plot_results="False"


CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python modules/flow/flow_infer.py \
									--model=$model \
									--flow_iters=$flow_iters \
                                   --dataset=$dataset \
                                   --img_type=$img_type \
                                   --log_dir=$log_dir \
                                   --MBW_Iteration=$MBW_Iteration \
                                   --to_plot_results=$to_plot_results
                                    

##### MV-NRSfM #####
mode=train
batch=1000
model=models/mvnrsfm/$dataset
dataset_metric_multiplier=1000
break_training_counter=10000
from_scratch='True'

CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 modules/mvnrsfm/mvnrsfm_train.py 	--mode=$mode \
														--dataset=$dataset \
														--batch=$batch \
														--model=$model \
														--dataset_metric_multiplier=$dataset_metric_multiplier \
														--MBW_Iteration=$MBW_Iteration \
														--break_training_counter=$break_training_counter \
														--from_scratch=$from_scratch

##### Outlier Rejection 
outlier_threshold=0.05
break_training_counter=10000
input_source='Flow'
batch=1000
dataset_metric_multiplier=1000
trust_MVNRSfM_Inliers='False'
from_scratch='False'
update_pickle='True'
leave_bbox_out='False'
update_bbox_only='False'
model=models/mvnrsfm/$dataset
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
                                                        --img_type=$img_type



#####################################################################################

MBW_Iteration=1

##### Retrain MV-NRSfM #####
mode=train
batch=1000
model=models/mvnrsfm/$dataset
dataset_metric_multiplier=1000
break_training_counter=10000
from_scratch='True'

CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 modules/mvnrsfm/mvnrsfm_train.py 	--mode=$mode \
														--dataset=$dataset \
														--batch=$batch \
														--model=$model \
														--dataset_metric_multiplier=$dataset_metric_multiplier \
														--MBW_Iteration=$MBW_Iteration \
														--break_training_counter=$break_training_counter \
														--from_scratch=$from_scratch



##### 2D Detector #####
config_file=configs/detector.yaml
scale_factor=400
TRAIN_END_EPOCH=100
leverage_prior_object_knowledge='True'
img_type=.jpg

echo "*********** Training 2D Detector ****************"
CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python modules/detector/detector.py \
         --cfg $config_file \
         --dataset=$dataset \
         --scale_factor=$scale_factor \
         --MBW_Iteration=$MBW_Iteration \
         --TRAIN_END_EPOCH=$TRAIN_END_EPOCH \
         --img_type=$img_type \
         --leverage_prior_object_knowledge=$leverage_prior_object_knowledge

#### BBox update
mode=train
batch=1000
outlier_threshold=0.05
break_training_counter=10000
dataset_metric_multiplier=1000
input_source='Detector'
model=models/mvnrsfm/$dataset
trust_MVNRSfM_Inliers='False'
from_scratch='False'
update_pickle='False'
leave_bbox_out='False'
update_bbox_only='True'
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
                                                        --img_type=$img_type


##### 2D Detector #####
config_file=configs/detector.yaml
scale_factor=200
TRAIN_END_EPOCH=200
leverage_prior_object_knowledge='True'

echo "*********** Training 2D Detector ****************"
CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python modules/detector/detector.py \
         --cfg $config_file \
         --dataset=$dataset \
         --scale_factor=$scale_factor \
         --MBW_Iteration=$MBW_Iteration \
         --TRAIN_END_EPOCH=$TRAIN_END_EPOCH \
         --leverage_prior_object_knowledge=$leverage_prior_object_knowledge

##### Pickle update
mode=train
batch=1000
outlier_threshold=0.05
break_training_counter=10000
dataset_metric_multiplier=1000
input_source='Detector'
model=models/mvnrsfm/$dataset
trust_MVNRSfM_Inliers='False'
from_scratch='False'
update_pickle='True'
leave_bbox_out='False'
update_bbox_only='False'
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
                                                        --img_type=$img_type



##############################################################################
for MBW_Iteration in {2..6}
do
    # ##### Retrain MV-NRSfM #####
    mode=train
    batch=1000
    model=models/mvnrsfm/$dataset
    dataset_metric_multiplier=1000
    break_training_counter=10000
    from_scratch='True'

    CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 modules/mvnrsfm/mvnrsfm_train.py 	--mode=$mode \
                                                            --dataset=$dataset \
                                                            --batch=$batch \
                                                            --model=$model \
                                                            --dataset_metric_multiplier=$dataset_metric_multiplier \
                                                            --MBW_Iteration=$MBW_Iteration \
                                                            --break_training_counter=$break_training_counter \
                                                            --from_scratch=$from_scratch



    ##### 2D Detector #####
    config_file=configs/detector.yaml
    scale_factor=200
    leverage_prior_object_knowledge='True'
    img_type=.jpg

    echo "*********** Training 2D Detector ****************"
    CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python modules/detector/detector.py \
            --cfg $config_file \
            --dataset=$dataset \
            --scale_factor=$scale_factor \
            --MBW_Iteration=$MBW_Iteration \
            --leverage_prior_object_knowledge=$leverage_prior_object_knowledge \
            --img_type=$img_type



    ##### Pickle update
    outlier_threshold=0.05
    break_training_counter=10000
    input_source='Detector'
    trust_MVNRSfM_Inliers='False'
    from_scratch='False'
    update_pickle='True'
    leave_bbox_out='True'
    update_bbox_only='False'
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
                                                            --img_type=$img_type

done

# ######################### EVAL MODE #########################
# config_file=configs/detector.yaml
# scale_factor=200
# leverage_prior_object_knowledge='True'
# pretrain_model_dataset=$dataset


# echo "*********** Evaluation mode ****************"
# CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python modules/detector/eval.py \
#          --cfg $config_file \
#          --dataset=$dataset \
#          --scale_factor=$scale_factor \
#          --leverage_prior_object_knowledge=$leverage_prior_object_knowledge \
#          --pretrain_model_dataset=$pretrain_model_dataset \
#          --MBW_Iteration=$MBW_Iteration \


# ######################### VISUALIZATIONS "2D Predictions" and "BBox" #########################
# field_of_validation='Predictions' # Options: "Inliers" | "Flow" | "MV_Train" | "MV" | "Detector"
# mode_of_validation='2D'
# MBW_Iteration=6
# plot_separate='False'

# CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 common/visualize.py --dataset=$dataset \
# 														--mode_of_validation=$mode_of_validation \
# 														--field_of_validation=$field_of_validation \
# 														--MBW_Iteration=$MBW_Iteration \
# 														--plot_separate=$plot_separate

# field_of_validation='Predictions' # Options: "Inliers" | "Flow" | "MV_Train" | "MV" | "Detector"
# mode_of_validation='BBox'
# MBW_Iteration=6
# plot_separate='False'

# CUDA_VISIBLE_DEVICES=$GPU_ID nice -10 python3 common/visualize.py --dataset=$dataset \
# 														--mode_of_validation=$mode_of_validation \
# 														--field_of_validation=$field_of_validation \
# 														--MBW_Iteration=$MBW_Iteration \
# 														--plot_separate=$plot_separate
