import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import relu

sys.path.append("configs")


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
