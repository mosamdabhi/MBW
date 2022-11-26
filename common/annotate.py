import argparse
import json
import math
import os
import shutil
import sys
from os.path import exists
from pathlib import Path

import _pickle as cPickle
import cv2
import numpy as np
from natsort import natsorted

sys.path.append("common")
from util_common import parse_boolean

import pdb

# function to display the coordinates of
# of the points clicked on the image

counter = 0
landmark_str_global = []
landmark_coordinates_global = []


def call_app():
    from app.app import app

    app.run(host=args.host, port=args.port, debug=False)


def make_dir(str_path):
    """Make directory"""
    if not os.path.isdir(str_path):
        os.mkdir(str_path)


def click_event(event, x, y, flags, params):
    global counter, landmark_str_global, landmark_coordinates_global
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        # print(x, ' ', y)
        print("Keypoint #: ", counter)
        landmark_str = input("Enter landmark name: ")
        landmark_str_global.append(landmark_str)
        landmark_coordinates_global.append((x, y))
        str_to_print = str(counter) + " | " + landmark_str

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (0, 0, 255), 2)
        cv2.putText(img, str_to_print, (x, y), font, 1, (0, 0, 255), 2)
        cv2.imshow("image", img)
        counter = counter + 1


def main(args):

    # Extract annotation file paths
    annot_path = "data/" + str(args.dataset) + "/annot"
    pkl_names = []
    for path in Path(annot_path).rglob("*.pkl"):
        pkl_names.append(path)
    pkl_names = [Path(p) for p in natsorted([str(p) for p in pkl_names])]

    if len(pkl_names) > 0:
        need_to_annotate = False
        # Find total camera views
        total_views = len(pkl_names)

        # Find total number of frames
        with open(str(pkl_names[0]), "rb") as fid:
            pickled_data = cPickle.load(fid)
        total_frames = len(pickled_data)

        # Check whether the data has G.T. 2D and 3D
        GT_Flag = False
        if "S_GT" in pickled_data[0]:
            GT_Flag = True
        else:
            Ws = []
            for idx in range(len(pickled_data)):
                Ws.append(pickled_data[idx]["W_GT"])
            Ws = np.asarray(Ws)
            valids = np.sum(~np.any(np.any(np.isnan(Ws), axis=1), axis=1))
            if valids == len(pickled_data):
                GT_Flag = True

        # Find number of joints/keypoints
        num_joints = pickled_data[0]["W_GT"].shape[0]
        W_NaN = np.zeros((num_joints, 2))
        W_NaN[:] = np.nan

        # Write uniformly sampled annotation if this is a dataset with G.T.
        if GT_Flag:
            uniform_samples = np.arange(
                0,
                total_frames,
                int(total_frames / ((total_frames * int(args.percentage_if_gt)) / 100)),
            )
            labels = np.zeros((total_frames))
            labels[uniform_samples] = 1
            for cam_idx in range(total_views):
                with open(str(pkl_names[cam_idx]), "rb") as fid:
                    pickled_data = cPickle.load(fid)
                for frame_idx in range(len(pickled_data)):
                    bbox_matrix = pickled_data[frame_idx]["W_GT"]
                    bbox = (
                        max(abs(bbox_matrix[:, 0])),
                        min(abs(bbox_matrix[:, 0])),
                        max(abs(bbox_matrix[:, 1])),
                        min(abs(bbox_matrix[:, 1])),
                    )
                    pickled_data[frame_idx]["BBox"] = bbox
                    if labels[frame_idx]:
                        pickled_data[frame_idx]["W_Inliers"] = pickled_data[frame_idx]["W_GT"]
                        pickled_data[frame_idx]["confidence"] = True
                    else:
                        pickled_data[frame_idx]["W_Inliers"] = W_NaN
                        pickled_data[frame_idx]["confidence"] = False
                with open(str(pkl_names[cam_idx]), "wb") as fid:
                    cPickle.dump(pickled_data, fid)

            if not args.unittest:
                print(
                    "Dataset with G.T. sampled with {0:.2f}% annotations".format(
                        (len(uniform_samples) / total_frames) * 100
                    )
                )

        else:
            manual_annotation_counter = 0
            total_available_frames = total_frames * total_views
            for cam_idx in range(total_views):
                with open(str(pkl_names[cam_idx]), "rb") as fid:
                    pickled_data = cPickle.load(fid)
                for frame_idx in range(len(pickled_data)):
                    if not np.any(np.isnan(pickled_data[frame_idx]["W_GT"])):
                        pickled_data[frame_idx]["confidence"] = True
                        manual_annotation_counter = manual_annotation_counter + 1
                        pickled_data[frame_idx]["W_Inliers"] = pickled_data[frame_idx]["W_GT"]
                    else:
                        pickled_data[frame_idx]["confidence"] = False
                        pickled_data[frame_idx]["W_Inliers"] = W_NaN
                with open(str(pkl_names[cam_idx]), "wb") as fid:
                    cPickle.dump(pickled_data, fid)

            percentage_annot = (manual_annotation_counter / total_available_frames) * 100
            total_frames_annot_per_view = manual_annotation_counter / total_views

            if not args.unittest:
                print(
                    "Dataset without G.T. | Manually annoated with {0:.2f}% annotations".format(
                        percentage_annot
                    )
                )
                print(
                    "Specifically, we have {} total frames annotated per view".format(
                        total_frames_annot_per_view
                    )
                )
                print(
                    "********************************************************************************************"
                )
            if total_frames_annot_per_view >= 1:
                if not args.unittest:
                    print("We have enough annotations. Let us proceed with MBW!")
            else:
                if not args.unittest:
                    print(
                        "HOLD ON! Why don't we annotate a few more frames, for better performance?"
                    )
                need_to_annotate = True

    else:
        need_to_annotate = True

    return need_to_annotate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name", default="Colobus_Monkey")
    parser.add_argument("--datasetunittest", help="dataset name", default="Colobus_Monkey")
    parser.add_argument(
        "--percentage_if_gt", help="Amount of percentage annotated if gt", default=2
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8888)

    parser.add_argument(
        "--unittest",
        type=parse_boolean,
        default=False,
        help="Flag for unit test.",
    )

    args = parser.parse_args()


    while 1:
        # Find out the total number of image frames
        images_dir_views = os.listdir("data/" + str(args.dataset) + "/images/")
        total_views = len(list(filter(lambda a: "CAM_" in a, images_dir_views)))
        images_path = "data/" + str(args.dataset) + "/images/CAM_1"
        img_names = []
        for path in Path(images_path).rglob("*.jpg"):
            img_names.append(path)
        img_names = [Path(p) for p in natsorted([str(p) for p in img_names])]

        print("Total frames in the video are: ", len(img_names))

        pdb.set_trace()


        # reading the image
        img = cv2.imread(str(img_names[0]), 1)

        # displaying the image
        cv2.imshow("image", img)


        cv2.setMouseCallback("image", click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

        print(
            "----------------------------------------------------------------------------------------"
        )
        print("Number of keypoints chosen: ", counter)
        print(
            "At a rate of 30 labels per minute, it would take you approximately {0:.2f} minutes to label all keypoints in all the views.".format(
                total_views * (num_joints) * total_annotations_required / 30
            )
        )

        print(
            "----------------------------------------------------------------------------------------"
        )

        pdb.set_trace()

    #     else:
    #         with open("configs/joints/" + str(args.dataset) + ".json", "r") as fid:
    #             joints_data = json.load(fid)
    #         joint_connections = joints_data["joint_connections"]
    #         num_joints = joints_data["num_joints"]
    #         joints_name = joints_data["joints_name"]
    #         img_loaded = cv2.imread(str(img_names[0]), 1)
    #         image_height, image_width = int(img_loaded.shape[0]), int(img_loaded.shape[1])
    #         print("Perfect. We found the joints config file. Let us annotate!")
    #         print(
    #             "----------------------------------------------------------------------------------------"
    #         )
    #         print("Number of keypoints chosen: ", num_joints)
    #         print(
    #             "----------------------------------------------------------------------------------------"
    #         )
    #         print(
    #             "At a rate of 30 labels per minute, it would take you ~ {} minutes to label all keypoints in all the views.".format(
    #                 math.ceil(total_views * (num_joints) * total_annotations_required / 30)
    #             )
    #         )
    #         print(
    #             "----------------------------------------------------------------------------------------"
    #         )

    #         # for cam_idx in range(total_views):

    #         # Creata a temporary image directory and copy the files we need to annotate
    #         tmp_dir_images = os.getcwd() + "/data/" + str(args.dataset) + "/images/tmp/"
    #         dir_annot_cam = os.getcwd() + "/data/" + str(args.dataset) + "/annot/annot"
    #         make_dir(tmp_dir_images)
    #         make_dir(dir_annot_cam)

    #         for cam_idx in range(total_views):
    #             # dir_annot_cam = os.getcwd() + "/data/" + str(args.dataset) + "/annot/CAM_" + str(cam_idx+1)

    #             for frame_idx in range(len(annotation_samples)):
    #                 shutil.copy(
    #                     "data/"
    #                     + str(args.dataset)
    #                     + "/images/CAM_"
    #                     + str(cam_idx + 1)
    #                     + "/"
    #                     + str(annotation_samples[frame_idx]).zfill(4)
    #                     + ".jpg",
    #                     tmp_dir_images
    #                     + str(annotation_samples[frame_idx]).zfill(4)
    #                     + "_CAM_"
    #                     + str(cam_idx + 1)
    #                     + ".jpg",
    #                 )

    #         for joint_idx in range(num_joints):
    #             tmp_dir_annot = os.getcwd() + "/data/" + str(args.dataset) + "/annot/tmp/"
    #             make_dir(tmp_dir_annot)
    #             # Create annotation directory per joint
    #             args_json = {
    #                 "ANNOTATION_DIR": tmp_dir_annot,
    #                 "IMAGE_DIR": tmp_dir_images,
    #                 "LANDMARK": "keypoint",
    #                 "IMAGE_HEIGHT": image_height,
    #                 "TOTAL_ANNOTATIONS": int(
    #                     total_views * (num_joints) * total_annotations_required
    #                 ),
    #             }

    #             with open("common/app/args_annotation.json", "w") as fid:
    #                 fid.write(json.dumps(args_json))

    #             print("-------------------------------------")
    #             print("We will be annotating all the views in one go! Let's do it!")
    #             print("Please annotate the {} joint".format(joints_name[joint_idx]))
    #             print("-------------------------------------")
    #             call_app()
    #             os.system(
    #                 "cp -r " + tmp_dir_annot + " " + dir_annot_cam + "/" + joints_name[joint_idx]
    #             )
    #             os.system("rm -rf " + tmp_dir_annot)

    # if not args.unittest:
    #     print("Alright, we have something")
    # else:
    #     print("-------------------------------")
    #     print("Preprocess Unit Test: OK")

    # if need_to_annotate:
    #     with open("configs/joints/" + str(args.dataset) + ".json", "r") as fid:
    #         joints_data = json.load(fid)
    #     num_joints = joints_data["num_joints"]
    #     joints_name = joints_data["joints_name"]
    #     W_NaN = np.zeros((num_joints, 2))
    #     W_NaN[:] = np.nan

    #     for cam_idx in range(total_views):
    #         data = []
    #         for frame_idx in range(len(img_names)):
    #             if frame_idx in annotation_samples:
    #                 W_ = []
    #                 for joint_idx in range(len(joints_name)):
    #                     coords_json = json.load(
    #                         open(
    #                             "data/"
    #                             + str(args.dataset)
    #                             + "/annot/annot/"
    #                             + joints_name[joint_idx]
    #                             + "/"
    #                             + str(frame_idx).zfill(4)
    #                             + "_CAM_"
    #                             + str(cam_idx + 1)
    #                             + ".jpg/"
    #                             + "keypoint.json"
    #                         )
    #                     )["coordinates"]
    #                     coords_concat = coords_json["x"], coords_json["y"]
    #                     W_.append(coords_concat)
    #                 d = {"W_GT": np.asarray(W_), "confidence": True}
    #                 data.append(d)
    #             else:
    #                 d = {"W_GT": W_NaN, "confidence": False}
    #                 data.append(d)

    #         with open(
    #             "data/" + str(args.dataset) + "/annot/CAM_" + str(cam_idx + 1) + ".pkl", "wb"
    #         ) as fid:
    #             cPickle.dump(data, fid)
