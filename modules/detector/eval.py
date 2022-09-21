import argparse
import json
import sys
from pathlib import Path

import _init_paths
import _pickle as cPickle
import cv2
import dataset
import matplotlib.image as img_matplotlib
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from config import cfg, update_config
from core.loss import JointsMSELoss
from natsort import natsorted
from tensorboardX import SummaryWriter

sys.path.append("modules/helpers")
sys.path.append("configs")
sys.path.append("common")


from core.function import train, validate
from util_common import calculate_BBox, get_device, make_dir, parse_boolean
from util_errors import computeMPJPE, computeMPJPE_array, error_ratio
from utils.utils import create_logger, get_model_summary, get_optimizer, save_checkpoint


def gen_image(im, W_Pred, W_GT, img_store_location, joint_connections):

    color_kpts_gt = "r"
    color_kpts_pred = "b"

    subplots = plt.subplots(1, 1)
    fig = subplots[0]
    axes = subplots[1]

    axes.imshow(im)
    # 2D label visualizations
    axes.scatter(x=W_GT[:, 0], y=W_GT[:, 1], c=color_kpts_gt, s=240)
    axes.scatter(x=W_Pred[:, 0], y=W_Pred[:, 1], c=color_kpts_pred, s=240)

    axes.axes.xaxis.set_visible(False)
    axes.axes.yaxis.set_visible(False)

    figure = plt.gcf()
    figure.set_size_inches(38, 36)
    plt.savefig(img_store_location, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()


def main(args):

    # GPU Device allotment
    DEVICE = get_device()

    # Extract joint connections
    with open("configs/joints/" + str(args.dataset) + ".json", "r") as fid:
        joints_data = json.load(fid)
    joint_connections = joints_data["joint_connections"]

    # Extract annotation file paths
    annot_path = "data/" + str(args.dataset) + "/annot"
    pkl_names = []
    for path in Path(annot_path).rglob("*.pkl"):
        pkl_names.append(path)
    pkl_names = [Path(p) for p in natsorted([str(p) for p in pkl_names])]

    # Extract total number of views
    total_views = len(pkl_names)

    """ If we have G.T., extract G.T. 2D and 3D, 
        else only 2D """
    # Find total number of frames
    with open(str(pkl_names[0]), "rb") as fid:
        pickled_data = cPickle.load(fid)
    total_frames = len(pickled_data)

    # Check whether the data has G.T. 2D and 3D
    GT_Flag = False
    Flow_Flag = False
    scale_factor_ = args.scale_factor
    if "S_GT" in pickled_data[0]:
        GT_Flag = True
    else:
        Ws = []
        W_Flows = []
        for idx in range(len(pickled_data)):
            Ws.append(pickled_data[idx]["W_GT"])
            # if flow exists, fill it
            if "W_Flow_Iter_0" in pickled_data[idx]:
                W_Flows.append(pickled_data[idx]["W_Flow_Iter_0"])
                Flow_Flag = True
        Ws = np.asarray(Ws)
        if Flow_Flag:
            W_Flows = np.asarray(W_Flows)
        valids = np.sum(~np.any(np.any(np.isnan(Ws), axis=1), axis=1))
        if valids == len(pickled_data):
            GT_Flag = True

    if GT_Flag:
        scale_factor_ = 200

    # Find number of joints/keypoints
    num_joints = pickled_data[0]["W_GT"].shape[0]

    test_json = []
    W_GT, confidence = [], []

    # Iterate over views
    for cam_idx in range(total_views):

        test_json_ = []
        W_, W_GT_, confidence_ = [], [], []

        with open(pkl_names[cam_idx], "rb") as fid:
            pickled_data = cPickle.load(fid)
        images_dir = "data/" + args.dataset + "/images/CAM_" + str(cam_idx + 1)
        image_paths = []
        for path in Path(images_dir).rglob("*" + str(args.img_type)):
            image_paths.append(path)
        image_paths = [Path(p) for p in natsorted([str(p) for p in image_paths])]
        img = cv2.imread(str(image_paths[0]))
        height_img, width_img = img.shape[0], img.shape[1]
        W_None = np.zeros((num_joints, 2))

        # Iterate over frames
        for frame_idx in range(total_frames):
            W_.append(pickled_data[frame_idx]["W_Inliers"])
            W_GT_.append(pickled_data[frame_idx]["W_GT"])
            confidence_.append(pickled_data[frame_idx]["confidence"])

            if GT_Flag:
                xmax, xmin, ymax, ymin = calculate_BBox(pickled_data[frame_idx]["W_GT"])
            else:
                if "BBox" in pickled_data[frame_idx]:
                    if ~np.any(np.isnan(pickled_data[frame_idx]["BBox"])):
                        xmax, xmin, ymax, ymin = pickled_data[frame_idx]["BBox"]
                    else:
                        xmax, xmin, ymax, ymin = width_img, 0, height_img, 0
                else:
                    xmax, xmin, ymax, ymin = width_img, 0, height_img, 0
            width, height = abs(xmax - xmin), abs(ymax - ymin)

            scale = width * (1 / scale_factor_) + height * (1 / scale_factor_)
            center = [(xmax + xmin) / 2, (ymax + ymin) / 2]

            if not args.unittest:
                if GT_Flag:
                    test_joints_field = "W_GT"
                else:
                    if args.MBW_Iteration == 1:
                        test_joints_field = "W_Flow_Iter_0"
                    else:
                        test_joints_field = "W_Detector_Iter_" + str(args.MBW_Iteration - 1)

                if test_joints_field in pickled_data[frame_idx]:
                    if np.any(
                        np.any(np.isnan(pickled_data[frame_idx][test_joints_field]), axis=1), axis=0
                    ):
                        test_joints = W_None.tolist()
                    else:
                        test_joints = pickled_data[frame_idx][test_joints_field].tolist()
                else:
                    test_joints = W_None.tolist()
            else:
                test_joints = W_None.tolist()

            d_test = {
                "joints_vis": np.ones(num_joints).astype(int).tolist(),
                "joints": test_joints,
                "image": str(image_paths[frame_idx]).split("images/")[-1],
                "scale": scale,
                "center": center,
            }
            test_json_.append(d_test)

        W_GT_ = np.asarray(W_GT_)
        confidence_ = np.asarray(confidence_)
        non_nans = np.argwhere(confidence_ == True)[:, 0]

        test_json.extend(test_json_)
        confidence.extend(confidence_)
        W_GT.extend(W_GT_)

    confidence = np.asarray(confidence)
    W_GT = np.asarray(W_GT)
    confidence_test = ~np.any(np.any(np.isnan(W_GT), axis=1), axis=1) == True

    if not args.unittest:
        print(
            "Detector validates over {:.2f}% samples".format(
                (np.argwhere(confidence_test == True).shape[0] / (total_frames * total_views)) * 100
            )
        )
        print("--------------------------------------------------------")

    with open(annot_path + "/valid.json", "w") as outfile:
        json.dump(test_json, outfile)

    # W_GT, confidence = np.asarray(W_GT), np.asarray(confidence)
    W_GT_reshaped = W_GT.reshape(total_frames * total_views, num_joints, 2)
    W_GT_non_NaN = ~np.any(np.any(np.isnan(W_GT_reshaped), axis=1), axis=1)

    # Retrieve the pretrained model path
    pretrained_model_path = "models/detector/" + args.pretrain_model_dataset
    make_dir(pretrained_model_path)
    update_config(cfg, num_joints, args, pretrained_model_path)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "eval")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    if not args.unittest:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(DEVICE)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).to(DEVICE)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    eval_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        args.unittest,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if not args.unittest:
        batch_size_ = cfg.TEST.BATCH_SIZE_PER_GPU
    else:
        batch_size_ = 1

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size_ * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    all_detections = validate(
        cfg,
        args.unittest,
        DEVICE,
        eval_loader,
        eval_dataset,
        model,
        criterion,
        final_output_dir,
        tb_log_dir,
        writer_dict,
    )[:, :, 0:2]

    curr_error_MPJPE = computeMPJPE(
        torch.from_numpy(all_detections[W_GT_non_NaN, :, :]),
        torch.from_numpy(W_GT_reshaped[W_GT_non_NaN, :, :]),
    ).numpy()

    curr_error_MPJPE_per_array = computeMPJPE_array(
        torch.from_numpy(all_detections[W_GT_non_NaN, :, :]),
        torch.from_numpy(W_GT_reshaped[W_GT_non_NaN, :, :]),
    ).numpy()

    ### plot top 25
    all_detections = all_detections.reshape(total_views, total_frames, num_joints, 2)
    all_GT = W_GT_reshaped.reshape(total_views, total_frames, num_joints, 2)
    # make_dir("tmp/")
    # for cam_idx in range(total_views):
    #     make_dir("tmp/CAM_" + str(cam_idx + 1))
    #     errors = computeMPJPE_array(all_detections[cam_idx, :, :, :], all_GT[cam_idx, :, :, :])
    #     best_idxs = np.argsort(errors)[0:25]
    #     vals = np.sort(errors)[0:25]

    #     for frame_idx in best_idxs:
    #         im_path = (
    #             "data/"
    #             + args.dataset
    #             + "/images/CAM_"
    #             + str(cam_idx + 1)
    #             + "/"
    #             + str(frame_idx).zfill(4)
    #             + str(args.img_type)
    #         )
    #         im = img_matplotlib.imread(im_path)
    #         gen_image(
    #             im,
    #             all_detections[cam_idx, frame_idx, :, :],
    #             all_GT[cam_idx, frame_idx, :, :],
    #             "tmp/CAM_" + str(cam_idx + 1) + "/" + str(frame_idx).zfill(4) + str(args.img_type),
    #             joint_connections,
    #         )

    # pdb.set_trace()

    print("Error between Predictions and G.T. labels is: ", curr_error_MPJPE)

    for cam_idx in range(total_views):
        with open(pkl_names[cam_idx], "rb") as fid:
            pickled_data = cPickle.load(fid)

        W_Predictions, confidence = [], []
        for idx in range(len(pickled_data)):
            W_Predictions.append(pickled_data[idx]["W_Inliers"])
            confidence.append(pickled_data[idx]["confidence"])

        W_Predictions, confidence = np.asarray(W_Predictions), np.asarray(confidence)
        nan_indices = np.argwhere(confidence == False)[:, 0]
        W_Predictions[nan_indices, :, :] = all_detections[cam_idx, nan_indices, :, :]

        for frame_idx in range(total_frames):
            pickled_data[frame_idx]["W_Predictions"] = W_Predictions[frame_idx, :, :]

        with open(pkl_names[cam_idx], "wb") as fid:
            cPickle.dump(pickled_data, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Parameters: 2D Detection """
    parser.add_argument("--cfg", help="config file", default="")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument("--prevModelDir", help="prev Model directory", type=str, default="")

    parser.add_argument("--dataset", help="dataset selection", default="Colobus_Monkey")
    parser.add_argument(
        "--pretrain_model_dataset", help="dataset selection", default="Colobus_Monkey"
    )
    parser.add_argument("--datasetunittest", help="dataset name", default="Colobus_Monkey")
    parser.add_argument("--scale_factor", default=200, type=int)
    parser.add_argument("--MBW_Iteration", default=0, type=int)
    parser.add_argument("--TRAIN_END_EPOCH", default=200, type=int)
    parser.add_argument("--img_type", help="image format", default=".jpg")

    parser.add_argument(
        "--leverage_prior_object_knowledge",
        type=parse_boolean,
        default=True,
        help="Flag deciding whether to leverage prior learned weights for the given object category.",
    )

    parser.add_argument(
        "--unittest",
        type=parse_boolean,
        default=False,
        help="Flag for unit test.",
    )

    args = parser.parse_args()

    if args.unittest:
        args.dataset = args.datasetunittest

    main(args)
