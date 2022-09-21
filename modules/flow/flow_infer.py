import sys

sys.path.append("modules/flow/core")
sys.path.append("configs")
sys.path.append("common")

import argparse
import copy
import json
import warnings
from pathlib import Path

import _pickle as cPickle
import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from util_common import (
    extract_bone_connections,
    get_device,
    load_image_flow,
    make_dir,
    parse_boolean,
    viz_flow,
)

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from raft import RAFT
from util_errors import error_ratio_array
from utils.utils import InputPadder


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

    # Find total camera views
    total_views = len(pkl_names)

    # Log dataset directory
    log_root = "logs/flow"
    make_dir(log_root)
    log_path_dataset = str(args.log_dir)
    make_dir(log_path_dataset)

    for cam_idx in range(total_views):

        # Extract annotations
        with open(str(pkl_names[cam_idx]), "rb") as fid:
            pickled_data = cPickle.load(fid)
        W_GT, W_annotated, confident_labels = [], [], []

        for frame_idx in range(len(pickled_data)):
            W_annotated.append(pickled_data[frame_idx]["W_Inliers"])
            W_GT.append(pickled_data[frame_idx]["W_GT"])
            confident_labels.append(pickled_data[frame_idx]["confidence"])

        W_annotated = np.asarray(W_annotated)
        confident_labels = np.asarray(confident_labels)
        confident_label_flags = np.argwhere(confident_labels == 1)
        W_GT = np.asarray(W_GT)

        # Print experiment information
        if not args.unittest:
            print(
                "Propagating flow using {0:.2f}% labels".format(
                    (np.argwhere(confident_labels == 1).shape[0] / confident_labels.shape[0]) * 100
                )
            )
        error_2D = error_ratio_array(W_annotated, W_GT)[confident_label_flags].mean()
        if not args.unittest:
            print("Error b/w Input 2D and G.T. 2D: {0:.2f} ".format(error_2D))

        # Extract images
        images_path = "data/" + str(args.dataset) + "/images/CAM_" + str(cam_idx + 1) + "/"
        img_names = []
        for path in Path(images_path).rglob("*" + str(args.img_type)):
            img_names.append(path)
        img_names = [Path(p) for p in natsorted([str(p) for p in img_names])]

        # Initialize pre-trained Flow model (RAFT)
        model = torch.nn.DataParallel(RAFT(args))

        if DEVICE.type == "cpu":
            model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
        else:
            model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        # Log views directory
        log_path_views = log_path_dataset + "/CAM_" + str(cam_idx + 1)
        make_dir(log_path_views)

        # Verify the W_annotated is accurate
        W_Flow = []

        # Find whether flow can propagate.
        start_index = np.argwhere(~np.any(np.any(np.isnan(W_annotated), axis=1), axis=1) == True)[
            :, 0
        ][0]
        tmp_nan = np.zeros((W_annotated.shape[1], 2))
        tmp_nan[:] = np.nan

        if start_index != 0:
            for nan_idx in range(start_index):
                W_Flow.append(copy.deepcopy(tmp_nan))

        W_Flow.append(copy.deepcopy(W_annotated[start_index]))
        query = W_annotated[start_index]

        if args.to_plot_results:
            image = np.array(Image.open(img_names[start_index])).astype(np.uint8)
            img_store_name = log_path_views + "/" + str(start_index) + args.img_type
            viz_flow(image, W_Flow[start_index], joint_connections, img_store_name)

        # Generate dense flow preditions and fill W_Flow
        with torch.no_grad():

            for img_idx in range(start_index, len(img_names) - 1):
                imfile1 = img_names[img_idx]
                imfile2 = img_names[img_idx + 1]
                image1 = load_image_flow(imfile1, DEVICE)
                image2 = load_image_flow(imfile2, DEVICE)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                if image1.shape[2] != image2.shape[2]:
                    print("-------------------------------")
                    print(
                        "Optical flow cannot generate predictions because the input images have different dimensions. Exiting ... "
                    )
                    print("Optical Flow Unit Test: FAIL")
                    return

                if confident_labels[img_idx + 1]:
                    if not args.unittest:
                        print("--------------------- Readjusting the flow!--------------------")
                    W_Flow.append(copy.deepcopy(W_annotated[img_idx + 1]))
                    query = W_annotated[img_idx + 1]
                else:
                    print("---- Calculating the flow!----")
                    flow_low, flow_up = model(image1, image2, iters=args.flow_iters, test_mode=True)
                    flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    for kpts_idx in range(query.shape[0]):
                        tmp = flo[
                            min(int(query[kpts_idx, 1]), int(flo.shape[0]) - 1),
                            min(int(query[kpts_idx, 0]), int(flo.shape[1]) - 1),
                            :,
                        ]
                        query[kpts_idx, :] = query[kpts_idx, :] + (tmp)
                    W_Flow.append(copy.deepcopy(query))
                    if args.unittest:
                        print("-------------------------------")
                        print("Optical Flow Unit Test: OK")
                        return

                if args.to_plot_results:
                    image = image2[0].permute(1, 2, 0).cpu().numpy() / 255.0
                    img_store_name = log_path_views + "/" + str(img_idx + 1) + "." + args.img_type
                    viz_flow(image, W_Flow[img_idx + 1], joint_connections, img_store_name)

                if not args.unittest:
                    print("Flow propagation: {}/{}".format(img_idx, len(img_names) - 1))

        W_Flow = np.asarray(W_Flow)
        field_name = "W_Flow_Iter_" + str(args.MBW_Iteration)
        for idx in range(len(pickled_data)):
            pickled_data[idx][field_name] = W_Flow[idx, :, :]
        with open(str(pkl_names[cam_idx]), "wb") as fid:
            cPickle.dump(pickled_data, fid)

        print("Sequence finished. Saving the pose tracks with dimension {}".format(W_Flow.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument(
        "--alternate_corr", action="store_true", help="use efficent correlation implementation"
    )
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--model", help="load learned model")
    parser.add_argument("--dataset", help="dataset selection", default="Colobus_Monkey")
    parser.add_argument("--datasetunittest", help="dataset name", default="Colobus_Monkey")
    parser.add_argument("--flow_iters", default=20, type=int)
    parser.add_argument("--img_type", help="image format", default=".jpg")
    parser.add_argument("--log_dir", help="path to store flow outputs")
    parser.add_argument("--MBW_Iteration", default=0, type=int)

    parser.add_argument(
        "--to_plot_results",
        type=parse_boolean,
        default=False,
        help="Flag deciding whether to plot the flow predictions.",
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
