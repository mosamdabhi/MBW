import argparse
import json
import sys
from pathlib import Path

import _pickle as cPickle
import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import scipy.io as sio
from matplotlib.patches import Rectangle
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

sys.path.append("configs")
sys.path.append("common")
from util_common import extract_bone_connections, get_colors, get_device, make_dir, parse_boolean


def get_trace3d(joint_connections, points3d, point_color=None, line_color=None, name="PointCloud"):
    """Yields plotly traces for visualization"""
    if point_color is None:
        point_color = "rgb(30, 20, 160)"
    if line_color is None:
        line_color = "rgb(30, 20, 160)"

    # Trace of points
    trace_pts = go.Scatter3d(
        x=points3d[:, 0],
        y=points3d[:, 2],
        z=points3d[:, 1],
        mode="markers",
        name=name,
        marker=dict(symbol="circle", size=6, color=point_color),
    )

    # Trace of line
    xlines, ylines, zlines = [], [], []
    for line in joint_connections:
        for point in line:
            xlines.append(points3d[point, 0])
            ylines.append(points3d[point, 2])
            zlines.append(points3d[point, 1])
        xlines.append(None)
        ylines.append(None)
        zlines.append(None)
    trace_lines = go.Scatter3d(
        x=xlines, y=ylines, z=zlines, mode="lines", name=name, line=dict(width=6, color=line_color)
    )

    return [trace_pts, trace_lines]


def get_figure3d(joint_connections, input_cam_x, input_cam_y, points3d, gt=None, range_scale=2500):
    """Yields plotly figure for visualization"""
    color_palette = get_colors()
    color_blue, color_red = color_palette["color_blue"], color_palette["color_red"]
    traces = get_trace3d(joint_connections, points3d, color_blue, color_blue, "Predicted KP")
    if gt is not None:
        traces += get_trace3d(joint_connections, gt, color_red, color_red, "Groundtruth KP")

    layout = go.Layout(
        scene=dict(
            aspectratio=dict(x=0.8, y=0.8, z=2),
            xaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale)),
            yaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale)),
            zaxis=dict(range=(-1 * range_scale, 1 * range_scale)),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=input_cam_x, y=input_cam_y, z=1.25),
        ),
    )
    return go.Figure(data=traces, layout=layout)


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

    return W_Pred, S_Pred, confidence, reprojection_error, BBox, W_GT, S_GT


def gen_predictions_with_confidence(im, W_Pred, confidence, joint_connections):

    color_kpts = "r"    

    total_views = len(im)

    if total_views == 1:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]
        axes.imshow(im[0])
        # 2D label visualizations
        axes.scatter(x=W_Pred[:, 0], y=W_Pred[:, 1], c=color_kpts, s=40)
        xlines, ylines = extract_bone_connections(W_Pred, joint_connections)
        axes.plot(xlines, ylines, c=color_kpts, linewidth=4)
        axes.axes.xaxis.set_visible(False)
        axes.axes.yaxis.set_visible(False)

    else:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]

        for cam_idx in range(total_views):

            axes[cam_idx].imshow(im[cam_idx])

            # 2D label visualizations
            axes[cam_idx].scatter(x=W_Pred[cam_idx, :, 0], y=W_Pred[cam_idx, :, 1], c=color_kpts, s=40)
            xlines, ylines = extract_bone_connections(W_Pred[cam_idx, :, :], joint_connections)
            axes[cam_idx].plot(xlines, ylines, c=color_kpts, linewidth=4)
            axes[cam_idx].axes.xaxis.set_visible(False)
            axes[cam_idx].axes.yaxis.set_visible(False)

            if confidence[cam_idx] == True:
                fontdict = {"fontsize": 50, "color": "g"}
                label = "Accepted"
            else:
                fontdict = {"fontsize": 50, "color": "r"}
                label = "Rejected"

            axes[cam_idx].axes.set_title(label, fontdict)

    figure = plt.gcf()
    figure.set_size_inches(38, 36)
    return figure
    # plt.savefig(img_store_location, bbox_inches="tight")
    # plt.cla()
    # plt.clf()
    # plt.close()




def gen_image_with_errors(
    im, W, reprojection_error, confidence, img_store_location, joint_connections
):

    color_kpts = "r"

    reproj_errors_in_plot = False

    total_views = len(im)

    if total_views == 1:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]
        axes.imshow(im[0])
        # 2D label visualizations
        axes.scatter(x=W[:, 0], y=W[:, 1], c=color_kpts, s=40)
        xlines, ylines = extract_bone_connections(W, joint_connections)
        axes.plot(xlines, ylines, c=color_kpts, linewidth=4)
        axes.axes.xaxis.set_visible(False)
        axes.axes.yaxis.set_visible(False)

    else:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]

        for cam_idx in range(total_views):

            axes[cam_idx].imshow(im[cam_idx])

            # 2D label visualizations
            axes[cam_idx].scatter(x=W[cam_idx, :, 0], y=W[cam_idx, :, 1], c=color_kpts, s=40)
            xlines, ylines = extract_bone_connections(W[cam_idx, :, :], joint_connections)
            axes[cam_idx].plot(xlines, ylines, c=color_kpts, linewidth=4)
            axes[cam_idx].axes.xaxis.set_visible(False)
            axes[cam_idx].axes.yaxis.set_visible(False)

            if confidence[cam_idx] == True:
                fontdict = {"fontsize": 50, "color": "g"}
                if total_views <= 2 and reproj_errors_in_plot:
                    label = "Accepted | Reproj. error: {:.4f}".format(reprojection_error[cam_idx])
                else:
                    label = "Accepted"

            else:
                fontdict = {"fontsize": 50, "color": "r"}
                if total_views <= 2 and reproj_errors_in_plot:
                    label = "Rejected | Reproj. error: {:.4f}".format(reprojection_error[cam_idx])
                else:
                    label = "Rejected"

            axes[cam_idx].axes.set_title(label, fontdict)

    figure = plt.gcf()
    figure.set_size_inches(38, 36)
    plt.savefig(img_store_location, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()


def gen_image(im, W, img_store_location, joint_connections):

    color_kpts = "r"

    total_views = len(im)

    if total_views == 1:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]
        axes.imshow(im[0])
        # 2D label visualizations
        axes.scatter(x=W[:, 0], y=W[:, 1], c=color_kpts, s=40)
        xlines, ylines = extract_bone_connections(W, joint_connections)
        axes.plot(xlines, ylines, c=color_kpts, linewidth=4)
        axes.axes.xaxis.set_visible(False)
        axes.axes.yaxis.set_visible(False)

    else:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]

        for cam_idx in range(total_views):
            axes[cam_idx].imshow(im[cam_idx])

            # 2D label visualizations
            axes[cam_idx].scatter(x=W[cam_idx, :, 0], y=W[cam_idx, :, 1], c=color_kpts, s=40)
            xlines, ylines = extract_bone_connections(W[cam_idx, :, :], joint_connections)
            axes[cam_idx].plot(xlines, ylines, c=color_kpts, linewidth=4)
            axes[cam_idx].axes.xaxis.set_visible(False)
            axes[cam_idx].axes.yaxis.set_visible(False)

    figure = plt.gcf()
    figure.set_size_inches(38, 36)
    plt.savefig(img_store_location, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()


def gen_BBox_demo(im, BBox, joint_connections):

    color_kpts = "r"

    total_views = len(im)

    if total_views == 1:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]
        axes.imshow(im[0])
        # 2D label visualizations
        # BBox label visualizations
        (xmax, xmin, ymax, ymin) = BBox.tolist()
        width = xmax - xmin
        height = ymax - ymin
        axes.add_patch(
            Rectangle((xmin, ymin), width, height, edgecolor="red", facecolor="none", linewidth=3)
        )
        axes.axes.xaxis.set_visible(False)
        axes.axes.yaxis.set_visible(False)

    else:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]

        for cam_idx in range(total_views):
            axes[cam_idx].imshow(im[cam_idx])

            # BBox label visualizations
            (xmax, xmin, ymax, ymin) = BBox[cam_idx, :].tolist()
            width = xmax - xmin
            height = ymax - ymin
            axes[cam_idx].add_patch(
                Rectangle(
                    (xmin, ymin), width, height, edgecolor="red", facecolor="none", linewidth=3
                )
            )
            axes[cam_idx].axes.xaxis.set_visible(False)
            axes[cam_idx].axes.yaxis.set_visible(False)

    figure = plt.gcf()
    figure.set_size_inches(38, 36)

    return figure

def gen_BBox(im, BBox, img_store_location, joint_connections):

    color_kpts = "r"

    total_views = len(im)

    if total_views == 1:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]
        axes.imshow(im[0])
        # 2D label visualizations
        # BBox label visualizations
        (xmax, xmin, ymax, ymin) = BBox.tolist()
        width = xmax - xmin
        height = ymax - ymin
        axes.add_patch(
            Rectangle((xmin, ymin), width, height, edgecolor="red", facecolor="none", linewidth=3)
        )
        axes.axes.xaxis.set_visible(False)
        axes.axes.yaxis.set_visible(False)

    else:
        subplots = plt.subplots(1, total_views)
        fig = subplots[0]
        axes = subplots[1]

        for cam_idx in range(total_views):
            axes[cam_idx].imshow(im[cam_idx])

            # BBox label visualizations
            (xmax, xmin, ymax, ymin) = BBox[cam_idx, :].tolist()
            width = xmax - xmin
            height = ymax - ymin
            axes[cam_idx].add_patch(
                Rectangle(
                    (xmin, ymin), width, height, edgecolor="red", facecolor="none", linewidth=3
                )
            )
            axes[cam_idx].axes.xaxis.set_visible(False)
            axes[cam_idx].axes.yaxis.set_visible(False)

    figure = plt.gcf()
    figure.set_size_inches(38, 36)
    plt.savefig(img_store_location, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()


def main(args):
    # Extract annotation and image file paths
    dataset = "data/" + args.dataset
    dataset_img = dataset + "/images"
    dataset_annot = dataset + "/annot/"
    logs_path = "logs/" + args.dataset + "/"
    make_dir(logs_path)

    pkl_names = []
    for path in Path(dataset_annot).rglob("*.pkl"):
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
    W_Pred, S_Pred, confidence, reprojection_error, BBox, W_GT, S_GT = process_dataset(
        args, GT_Flag, pkl_names
    )
    # pdb.set_trace()

    if not args.validate_manual_labels:
        if S_Pred.shape[1] > 0:
            S_Pred = S_Pred[0, :, :, :]
        if GT_Flag:
            S_GT = S_GT[0, :, :, :]

    if args.mode_of_validation == "2D" or args.mode_of_validation == "BBox":

        image_paths_cam = []
        for cam_idx in range(total_views):
            image_paths_cam_ = []
            for path in Path(dataset_img + "/CAM_" + str(cam_idx + 1) + "/").rglob(
                "*" + str(args.img_type)
            ):
                image_paths_cam_.append(path)
            image_paths_cam_ = [Path(p) for p in natsorted([str(p) for p in image_paths_cam_])]
            image_paths_cam.append(image_paths_cam_)

        if args.mode_of_validation == "2D":
            logs_path_dir = logs_path + args.mode_of_validation + "_" + args.field_of_validation
            make_dir(logs_path_dir)
            non_nan_indices = np.any(np.any(np.any(~np.isnan(W_Pred), axis=2), axis=2), axis=0)
            non_nan_indices = np.argwhere(non_nan_indices == True)[:, 0].tolist()
            # pdb.set_trace()

            if args.plot_separate:
                logs_path_dir_cam = logs_path_dir + "/Separate"
                make_dir(logs_path_dir_cam)
                for cam_idx in range(total_views):
                    logs_path_dir_cam_specific = logs_path_dir_cam + "/CAM_" + str(cam_idx + 1)
                    make_dir(logs_path_dir_cam_specific)
                    for frame_idx in tqdm(non_nan_indices):
                        im = []
                        im.append(img.imread(image_paths_cam[cam_idx][frame_idx]))
                        img_store_location = (
                            logs_path_dir_cam_specific + "/" + str(frame_idx) + str(args.img_type)
                        )
                        gen_image(
                            im,
                            W_Pred[cam_idx, frame_idx, :, :],
                            img_store_location,
                            joint_connections,
                        )
                        # return
                        # pdb.set_trace()
            else:
                logs_path_dir_cam = logs_path_dir + "/Combined"
                make_dir(logs_path_dir_cam)
                for frame_idx in tqdm(non_nan_indices):
                    im = []
                    for cam_idx in range(total_views):
                        im.append(img.imread(image_paths_cam[cam_idx][frame_idx]))

                    img_store_location = (
                        logs_path_dir_cam + "/" + str(frame_idx) + str(args.img_type)
                    )

                    if args.field_of_validation == "Predictions":
                        gen_image_with_errors(
                            im,
                            W_Pred[:, frame_idx, :, :],
                            reprojection_error[:, frame_idx],
                            confidence[:, frame_idx],
                            img_store_location,
                            joint_connections,
                        )
                    else:
                        gen_image(
                            im, W_Pred[:, frame_idx, :, :], img_store_location, joint_connections
                        )

        elif args.mode_of_validation == "BBox":
            logs_path_dir = logs_path + args.mode_of_validation
            make_dir(logs_path_dir)

            if args.plot_separate:
                logs_path_dir_cam = logs_path_dir + "/Separate"
                make_dir(logs_path_dir_cam)
                for cam_idx in range(total_views):
                    logs_path_dir_cam_specific = logs_path_dir_cam + "/CAM_" + str(cam_idx + 1)
                    make_dir(logs_path_dir_cam_specific)
                    for frame_idx in tqdm(range(BBox.shape[1])):
                        im = []
                        im.append(img.imread(image_paths_cam[cam_idx][frame_idx]))
                        img_store_location = (
                            logs_path_dir_cam_specific + "/" + str(frame_idx) + ".jpg"
                        )
                        gen_BBox(
                            im, BBox[cam_idx, frame_idx, :], img_store_location, joint_connections
                        )
            else:
                logs_path_dir_cam = logs_path_dir + "/Combined"
                make_dir(logs_path_dir_cam)
                for frame_idx in tqdm(range(BBox.shape[1])):
                    im = []
                    for cam_idx in range(total_views):
                        im.append(img.imread(image_paths_cam[cam_idx][frame_idx]))

                    img_store_location = logs_path_dir_cam + "/" + str(frame_idx) + ".jpg"

                    gen_BBox(im, BBox[:, frame_idx, :], img_store_location, joint_connections)

    elif args.mode_of_validation == "3D":
        R = rigid_rotation
        logs_path_dir = logs_path + args.mode_of_validation
        make_dir(logs_path_dir)
        mat_contents = sio.loadmat("common/cam_25_8.mat")
        cam_x = mat_contents["x_final"][0].tolist()
        cam_y = mat_contents["y_final"][0].tolist()
        cam_counter = 0

        non_nan_indices = np.any(np.any(~np.isnan(S_Pred), axis=1), axis=1)
        non_nan_indices = np.argwhere(non_nan_indices == True)[:, 0].tolist()

        for idx in tqdm(non_nan_indices):
            if ~np.any(S_Pred[idx, :, :] == 0):
                if cam_counter == len(cam_x):
                    cam_counter = 0
                img_store_location = logs_path_dir + "/" + str(idx) + ".png"
                fig = get_figure3d(
                    joint_connections,
                    cam_x[cam_counter],
                    cam_y[cam_counter],
                    S_Pred[idx, :, :] @ R,
                    range_scale=range_scale,
                )
                fig.write_image(img_store_location)
                cam_counter = cam_counter + 1
        print("3D validation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset selection", default="Colobus_Monkey")
    parser.add_argument(
        "--field_of_validation",
        help="Decide b/w Inliers, Flow, Detector, MV",
        default="Inliers",
    )
    parser.add_argument("--mode_of_validation", help="Decide b/w 2D, 3D, and BBox", default="2D")
    parser.add_argument("--MBW_Iteration", default=0, type=int)
    parser.add_argument("--img_type", help="image format", default=".jpg")

    parser.add_argument(
        "--plot_separate",
        type=parse_boolean,
        default=False,
        help="Flag deciding whether to plot from separate cameras/views.",
    )

    parser.add_argument(
        "--validate_manual_labels",
        type=parse_boolean,
        default=False,
        help="Just plot inital manual labels",
    )

    args = parser.parse_args()
    main(args)
