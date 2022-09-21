import argparse
import json
import os
import pprint
import shutil
import sys
from pathlib import Path

import _init_paths
import _pickle as cPickle
import cv2
import dataset
import numpy as np

sys.path.append("modules/helpers")
sys.path.append("configs")
sys.path.append("common")


import models
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from config import cfg, update_config
from core.function import train, validate
from core.loss import JointsMSELoss
from natsort import natsorted
from tensorboardX import SummaryWriter
from util_common import calculate_BBox, get_device, make_dir, parse_boolean
from util_errors import computeMPJPE, error_ratio
from utils.utils import create_logger, get_model_summary, get_optimizer, save_checkpoint


def main(args):
    best_error_MPJPE = 1e10

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

    # Retrieve the pretrained model path
    pretrained_model_path = "models/detector/" + args.dataset
    make_dir(pretrained_model_path)
    update_config(cfg, num_joints, args, pretrained_model_path)

    train_json, test_json = [], []
    W_GT, confidence = [], []

    # Iterate over views
    for cam_idx in range(total_views):

        train_json_ = []
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

        # if args.unittest:
        #     xmax, xmin, ymax, ymin = width_img, 0, height_img, 0

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

            d_train = {
                "joints_vis": np.ones(num_joints).astype(int).tolist(),
                "joints": pickled_data[frame_idx]["W_Inliers"].tolist(),
                "image": str(image_paths[frame_idx]).split("images/")[-1],
                "scale": scale,
                "center": center,
            }

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
            train_json_.append(d_train)
            test_json_.append(d_test)

        W_GT_ = np.asarray(W_GT_)

        confidence_ = np.asarray(confidence_)
        non_nans = np.argwhere(confidence_ == True)[:, 0]

        if not args.unittest:
            print(
                "--------------------------------------------------------------------------------"
            )
            print(
                "Total labeled samples: {:.2f}% for view #{}".format(
                    (non_nans.shape[0] / total_frames) * 100, cam_idx + 1
                )
            )

        train_json_ = [train_json_[index] for index in non_nans]
        train_json.extend(train_json_)
        test_json.extend(test_json_)

        confidence.extend(confidence_)
        W_GT.extend(W_GT_)

    confidence = np.asarray(confidence)
    W_GT = np.asarray(W_GT)
    confidence_test = ~np.any(np.any(np.isnan(W_GT), axis=1), axis=1) == True

    if not args.unittest:
        print("--------------------------------------------------------")
        print(
            "Detector trains over {:.2f}% samples".format(
                (np.argwhere(confidence == True).shape[0] / (total_frames * total_views)) * 100
            )
        )
        print(
            "Detector validates over {:.2f}% samples".format(
                (np.argwhere(confidence_test == True).shape[0] / (total_frames * total_views)) * 100
            )
        )
        print("--------------------------------------------------------")

    with open(annot_path + "/train.json", "w") as outfile:
        json.dump(train_json, outfile)
    with open(annot_path + "/valid.json", "w") as outfile:
        json.dump(test_json, outfile)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, args.unittest, "train")

    if not args.unittest:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(os.path.join(this_dir, "lib/models", cfg.MODEL.NAME + ".py"), final_output_dir)

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    writer_dict["writer"].add_graph(model, (dump_input,))

    if not args.unittest:
        logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(DEVICE)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).to(DEVICE)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        args.unittest,
        True,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    valid_dataset = eval("dataset." + cfg.DATASET.DATASET)(
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
        batch_size_train_ = cfg.TRAIN.BATCH_SIZE_PER_GPU
        batch_size_test_ = cfg.TEST.BATCH_SIZE_PER_GPU
    else:
        batch_size_train_, batch_size_test_ = 1, 1

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train_ * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size_test_ * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, "checkpoint.pth")

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        if not args.unittest:
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint["epoch"]
        best_perf = checkpoint["perf"]
        last_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        if not args.unittest:
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint["epoch"])
            )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, args.TRAIN_END_EPOCH):
        # for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        train(
            cfg,
            args.unittest,
            DEVICE,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            final_output_dir,
            tb_log_dir,
            writer_dict,
        )
        lr_scheduler.step()

        all_detections = validate(
            cfg,
            args.unittest,
            DEVICE,
            valid_loader,
            valid_dataset,
            model,
            criterion,
            final_output_dir,
            tb_log_dir,
            writer_dict,
        )[:, :, 0:2]

        curr_error_MPJPE = computeMPJPE(
            torch.from_numpy(all_detections[confidence_test, :, :]),
            torch.from_numpy(W_GT[confidence_test, :, :]),
        ).numpy()

        if best_error_MPJPE > curr_error_MPJPE:
            if not args.unittest:
                print(
                    "====> Euclidean error | Current {} | Best {}".format(
                        curr_error_MPJPE, best_error_MPJPE
                    )
                )
            best_error_MPJPE = curr_error_MPJPE
            best_model = True
            all_detections = all_detections.reshape(total_views, total_frames, num_joints, 2)

            for cam_idx in range(total_views):
                with open(pkl_names[cam_idx], "rb") as fid:
                    pickled_data = cPickle.load(fid)

                for frame_idx in range(total_frames):
                    pickled_data[frame_idx][
                        "W_Detector_Iter_" + str(args.MBW_Iteration)
                    ] = all_detections[cam_idx, frame_idx, :, :]

                if not args.unittest:
                    with open(pkl_names[cam_idx], "wb") as fid:
                        cPickle.dump(pickled_data, fid)

        else:
            if not args.unittest:
                print(
                    "====> Euclidean error | Current {} | Best {}".format(
                        curr_error_MPJPE, best_error_MPJPE
                    )
                )
            best_model = False

        if not args.unittest:
            logger.info("====> Saving checkpoint to {}".format(final_output_dir))
            print("=======================================================")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": cfg.MODEL.NAME,
                    "state_dict": model.state_dict(),
                    "best_state_dict": model.module.state_dict(),
                    "perf": best_error_MPJPE,
                    "optimizer": optimizer.state_dict(),
                },
                best_model,
                final_output_dir,
            )
            os.system(
                "cp "
                + os.path.join(final_output_dir, "model_best.pth")
                + " "
                + pretrained_model_path
            )

        if args.unittest:
            print("-------------------------------")
            print("2D Detector Unit Test: OK")
            return

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth")

    if not args.unittest:
        logger.info("=> saving final model state to {}".format(final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict["writer"].close()


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
    parser.add_argument("--datasetunittest", help="dataset name", default="Colobus_Monkey")
    parser.add_argument("--MBW_Iteration", default=0, type=int)
    parser.add_argument("--scale_factor", default=200, type=int)
    parser.add_argument("--TRAIN_END_EPOCH", default=100, type=int)
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
