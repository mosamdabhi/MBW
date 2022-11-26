import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import relu
from pathlib import Path
from natsort import natsorted
import _pickle as cPickle
import json

sys.path.append("configs")

def process_dataset(args, GT_Flag, pkl_names):
    """
    Dataset processing
    """

    W_Pred, W_GT, S_Pred, S_GT, confidence, reprojection_error, BBox = [], [], [], [], [], [], []

    for cam_idx in range(len(pkl_names)):

        """Load the pickle file"""
        with open(pkl_names[cam_idx], "rb") as fid:
            pickled_data = cPickle.load(fid)

        """ Fill the data relevant to the camera id """
        W_tmp, S_tmp, W_GT_tmp, S_GT_tmp, confidence_tmp, reprojection_error_tmp, BBox_tmp = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for frame_idx in range(len(pickled_data)):

            if args.field_of_validation == "Inliers" or args.field_of_validation == "Predictions":
                W_field_name = "W_" + args.field_of_validation
            else:
                W_field_name = "W_" + args.field_of_validation + "_Iter_" + str(args.MBW_Iteration)

            if args.field_of_validation == "MV_Train":
                S_field_name = "S_" + args.field_of_validation + "_Iter_" + str(args.MBW_Iteration)
            else:
                S_field_name = "S_MV" + "_Iter_" + str(args.MBW_Iteration)

            W_tmp.append(pickled_data[frame_idx][W_field_name])
            confidence_tmp.append(pickled_data[frame_idx]["confidence"])

            if not args.validate_manual_labels:
                if S_field_name in pickled_data[frame_idx]:
                    S_tmp.append(pickled_data[frame_idx][S_field_name])

                if "reprojection_error_Iter_" + str(args.MBW_Iteration) in pickled_data[frame_idx]:
                    reprojection_error_tmp.append(
                        pickled_data[frame_idx][
                            "reprojection_error_Iter_" + str(args.MBW_Iteration)
                        ]
                    )
                if "BBox" in pickled_data[frame_idx]:
                    BBox_tmp.append(pickled_data[frame_idx]["BBox"])
                W_GT_tmp.append(pickled_data[frame_idx]["W_GT"])

                if GT_Flag:
                    S_GT_tmp.append(pickled_data[frame_idx]["S_GT"])

        W_tmp = np.asarray(W_tmp)
        S_tmp = np.asarray(S_tmp)
        confidence_tmp = np.asarray(confidence_tmp)
        reprojection_error_tmp = np.asarray(reprojection_error_tmp)
        BBox_tmp = np.asarray(BBox_tmp)
        W_GT_tmp = np.asarray(W_GT_tmp)
        if GT_Flag:
            S_GT_tmp = np.asarray(S_GT_tmp)

        W_Pred.append(W_tmp)
        S_Pred.append(S_tmp)
        confidence.append(confidence_tmp)
        reprojection_error.append(reprojection_error_tmp)
        BBox.append(BBox_tmp)
        W_GT.append(W_GT_tmp)
        if GT_Flag:
            S_GT.append(S_GT_tmp)

    W_Pred = np.asarray(W_Pred)
    S_Pred = np.asarray(S_Pred)
    confidence = np.asarray(confidence)
    reprojection_error = np.asarray(reprojection_error)
    BBox = np.asarray(BBox)
    W_GT = np.asarray(W_GT)
    if GT_Flag:
        S_GT = np.asarray(S_GT)
    
    loaded_data = {'W_Pred': W_Pred, \
                   'S_Pred': S_Pred, \
                   'BBox': BBox, \
                   'confidence': confidence, \
                   'reprojection_error': reprojection_error, \
                   'W_GT': W_GT, \
                   'S_GT': S_GT}

    return loaded_data



def load_datasets(args):
    dataset = "data/" + args.dataset
    dataset_img = dataset + "/images"
    dataset_annot = dataset + "/annot/"

    # Annotation pkl names
    pkl_names = []
    for path in Path(dataset_annot).rglob("*.pkl"):
        pkl_names.append(path)
    pkl_names = [Path(p) for p in natsorted([str(p) for p in pkl_names])]

    # Extract total number of views
    total_views = len(pkl_names)

    # Image path names
    image_paths_cam = []
    for cam_idx in range(total_views):
        image_paths_cam_ = []
        for path in Path(dataset_img + "/CAM_" + str(cam_idx + 1) + "/").rglob(
            "*" + str(args.img_type)
        ):
            image_paths_cam_.append(path)
        image_paths_cam_ = [Path(p) for p in natsorted([str(p) for p in image_paths_cam_])]
        image_paths_cam.append(image_paths_cam_)

    ## If we have G.T., extract G.T. 2D and 3D, else only 2D
    # Find total number of frames
    with open(str(pkl_names[0]), "rb") as fid:
        pickled_data = cPickle.load(fid)
    total_frames = len(pickled_data)

    # Check whether the data has G.T. 2D and 3D
    if "S_GT" in pickled_data[0]:
        GT_Flag = True
    else:
        GT_Flag = False

    # Find number of joints/keypoints
    num_joints = pickled_data[0]["W_GT"].shape[0]

    # Extract joint connections
    with open("configs/joints/" + str(args.dataset) + ".json", "r") as fid:
        joints_data = json.load(fid)
    
    joint_connections = joints_data["joint_connections"]
    range_scale = joints_data["range_scale"]
    rigid_rotation = joints_data["R"]

    """ Generate the validation and refined directories, if required """
    loaded_data = process_dataset(
        args, GT_Flag, pkl_names
    )

    misc_data = {'joint_connections': joint_connections, \
                    'range_scale': range_scale, \
                    'rigid_rotation': rigid_rotation, \
                    'num_joints': num_joints, \
                    'GT_Flag': GT_Flag, \
                    'total_frames': total_frames, \
                    'total_views': total_views, \
                    'image_paths': image_paths_cam}

    return loaded_data, misc_data


def calculate_BBox(input_matrix):
    BBox = (
        max(abs(input_matrix[:, 0])),
        min(abs(input_matrix[:, 0])),
        max(abs(input_matrix[:, 1])),
        min(abs(input_matrix[:, 1])),
    )
    return BBox


def extract_bone_connections(W, joint_connections):
    xlines, ylines = [], []
    for line in joint_connections:
        for point in line:
            xlines.append(W[point, 0])
            ylines.append(W[point, 1])
        xlines.append(None)
        ylines.append(None)

    return xlines, ylines


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


def get_colors():
    color_palette = {
        "color_blue": "rgb(0, 0, 255)",
        "color_red": "rgb(255, 0, 0)",
        "color_green": "rgb(0,255,0)",
    }
    return color_palette


def relu_threshold(input, thrsh):
    """
    Function: ReLU nonlinearity.
    input: Input Tensor, ReLU Threshold Value
    output: ReLU'ed Tensor
    """
    return relu(input + thrsh.view(1, -1, 1, 1))


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def make_dir(str_path):
    """Make directory"""
    if not os.path.isdir(str_path):
        os.mkdir(str_path)


def load_image_flow(imfile, DEVICE):
    img = np.array(Image.open(imfile)).astype(np.uint8)

    if len(img.shape) < 3:
        img_new = np.zeros((img.shape[0], img.shape[1], 3))
        img_new[:, :, 0] = img
        img_new[:, :, 1] = img
        img_new[:, :, 2] = img
        img = img_new

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz_flow(img, W, joint_connections, img_store_name):
    xlines, ylines = extract_bone_connections(W, joint_connections)
    plt.imshow(img)
    plt.scatter(W[:, 0], W[:, 1], color="red", s=3)
    plt.plot(xlines, ylines, c="r", linewidth=0.5)

    plt.savefig(img_store_name, bbox_inches="tight", dpi=500)
    plt.close()
    plt.cla()
    plt.clf()
