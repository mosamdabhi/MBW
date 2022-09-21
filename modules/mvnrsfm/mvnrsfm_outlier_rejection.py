import argparse
import json
import sys
from pathlib import Path

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from torch import optim

sys.path.append("modules/helpers")
sys.path.append("configs")
sys.path.append("common")

import matplotlib.image as img
import robust_loss_pytorch.general
from mv_nrsfm_configs import get_dictionary, get_seed
from neural_prior_modules import (
    BlockSparseCodingLayer,
    CameraEstimator,
    OnPEstimator,
    SparseCodeExtractionLayer,
    VecSparseCodingLayer,
)
from util_cameras import normalize_3d_structure, procrustes_align
from util_common import calculate_BBox, get_device, make_dir, parse_boolean
from util_errors import computeMPJPE, computeMPJPE_array, error_ratio


class MVNRSfM(nn.Module):
    """
    Class containing all the parameters for the MultiView Non-Rigid Structure-from-Motion.
    Lifts 2D to 3D
    """

    def __init__(
        self,
        args,
        GT_Flag=False,
        DEVICE="cuda",
        num_joints=21,
        total_views=4,
        total_frames=1000,
        dict_list=list(np.arange(125, 7, -10)),
    ):
        """
        Initialize the BaseAPI object
        :param num_joints: number of joints
        :param dict_list: sparse dictionary dimensions

        """
        # Necessary for proper multiple sub-classing
        super(MVNRSfM, self).__init__()
        self.model_path = args.model  # model path or trained weights path
        self.from_scratch = args.from_scratch
        self.DEVICE = DEVICE
        self.MBW_Iteration = args.MBW_Iteration
        self.num_joints = num_joints
        self.total_views = total_views
        self.total_frames = total_frames
        share_weight = True  # Share weights b/w Encoder and Decoder Dictionaries
        make_dir(args.model)

        # Neural Prior Module
        self.sparse_coding_layers = nn.ModuleList(
            [VecSparseCodingLayer(dict_list[0], num_joints, share_weight)]
        )
        for idx in range(1, len(dict_list)):
            self.sparse_coding_layers.append(
                BlockSparseCodingLayer(dict_list[idx], dict_list[idx - 1], share_weight)
            )
        self.code_estimator = SparseCodeExtractionLayer(code_dim=dict_list[-1], is_2d=True)

        # Camera Module
        self.onp_estimator = OnPEstimator()

        # MV-NRSfM Training
        self.GT_Flag = GT_Flag
        if ~self.GT_Flag:
            self.unit_scale = True
        self.dataset_metric_multiplier = args.dataset_metric_multiplier
        self.dict_dims = dict_list
        self.log_dir = args.model
        self.print_freq = 50
        self.save_freq = 1000
        self.lr = 0.0001
        self.max_itr = 1000000
        self.lr_decay_step = 5000
        self.lr_decay_rate = 0.95
        self.validation_freq = 1000
        self.batch = args.batch
        self.bottleneck_size = dict_list[-1]

        # Load model if not training from scratch
        if not args.from_scratch:
            if self.DEVICE.type == "cpu":
                self.model = self.load_state_dict(
                    torch.load(args.model + "/model_best.pth", map_location=torch.device("cpu"))
                )
            else:
                self.model = self.load_state_dict(torch.load(args.model + "/model_best.pth"))

    def save_model(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def get_parameters(self):
        return self.model.parameters()

    def forward(self, pts_2d_packet):

        layer_num = len(self.sparse_coding_layers)

        code_inner = (
            torch.from_numpy(
                np.zeros((len(pts_2d_packet), pts_2d_packet[0].shape[0], self.dict_dims[-1], 1, 1))
            )
            .float()
            .to(self.DEVICE)
        )

        """ Encode the 2D representation. """
        camera_matrix = []
        for outer_idx in range(len(pts_2d_packet)):
            flag_nan = ~torch.any(torch.any(torch.isnan(pts_2d_packet[outer_idx]), dim=1), dim=1)
            code_block = pts_2d_packet[outer_idx]
            code_block = code_block[flag_nan]
            indices_non_nan = torch.nonzero(flag_nan.float()).view(-1)
            for idx in range(layer_num):
                code_block = self.sparse_coding_layers[idx].encode_with_cam(code_block)

            code_inner[outer_idx, indices_non_nan, :, :, :] = (
                self.code_estimator(code_block).float().to(self.DEVICE)
            )

        """ Enforce multi-view consistency by pooling the codes from all the views. """
        code_pooled = torch.sum(code_inner, dim=0) / len(pts_2d_packet)
        code_canon = code_pooled.clone()

        """ Decode the pooled code and generate 3D structure in a canonical frame. """
        for idx in range(layer_num - 1, 0, -1):
            code_pooled = self.sparse_coding_layers[idx].decode(code_pooled)
        pts_recon_canonical, dict_final_pass = self.sparse_coding_layers[0].decode(code_pooled)
        W_cat = torch.cat((pts_2d_packet), dim=0)
        S_cat_pred = torch.cat([pts_recon_canonical] * len(pts_2d_packet))
        flag_nan = ~torch.any(torch.any(torch.isnan(W_cat), dim=1), dim=1)

        """ Estimate the camera using least squares algebraic solution of OnP """
        pts_2d_reproj, pts_3d_recon, cam_matrix, rotation_matrix = self.onp_estimator(
            W_cat[flag_nan], S_cat_pred[flag_nan]
        )

        prediction = (
            pts_3d_recon,
            pts_2d_reproj,
            W_cat,
            code_canon,
            cam_matrix,
            pts_recon_canonical,
        )

        return prediction, flag_nan

    def train(self, adaptive, input):
        prediction, flag_nan = self.forward(input)
        S_Pred, W_Pred, W_Input = (prediction[0], prediction[1], prediction[2])
        reprojection_loss = torch.mean(
            adaptive.lossfun(
                W_Pred.view(W_Pred.shape[0], W_Pred.shape[1] * W_Pred.shape[2])
                - W_Input[flag_nan].view(
                    W_Input[flag_nan].shape[0],
                    W_Input[flag_nan].shape[1] * W_Input[flag_nan].shape[2],
                )
            )
        )
        return reprojection_loss

    def validation(self, input):
        with torch.no_grad():

            W = input[0]
            T_Offset = torch.cat(input[1], dim=0)
            W_GT = torch.cat(input[2], dim=0)
            if self.GT_Flag:
                S_GT = torch.cat(input[3], dim=0)

            W_, S_, Errors_ = [], [], []

            prediction, flag_nan = self.forward(W)
            # All of the fields below are non-nan values. Fill them back in their respective size matrices.
            S_Pred, W_Pred, W_Input, code, cam, S_Canon = (
                prediction[0],
                prediction[1],
                prediction[2],
                prediction[3],
                prediction[4],
                prediction[5],
            )

            # Define full size
            dim_1, dim_2 = torch.cat(W, dim=0).shape[0], torch.cat(W, dim=0).shape[1]
            # Fill the full size matrices
            W_Pred = W_Pred + T_Offset[flag_nan]
            W_Input[flag_nan] = W_Input[flag_nan] + T_Offset[flag_nan]
            S_Pred_, W_Pred_ = (
                torch.zeros((dim_1, dim_2, 3)).to(self.DEVICE).float(),
                torch.zeros((dim_1, dim_2, 2)).to(self.DEVICE),
            )
            S_Pred_[:], W_Pred_[:] = torch.nan, torch.nan
            S_Pred_[flag_nan], W_Pred_[flag_nan] = S_Pred, W_Pred

            # Procrustes-Align if G.T. is available and pack the matrices to deliver back.
            if self.GT_Flag:
                # 3D Error
                S_Pred_ = procrustes_align(S_Pred_, S_GT)
                PA_MPJPE_3D = (
                    computeMPJPE(S_Pred_[flag_nan], S_GT[flag_nan]) * self.dataset_metric_multiplier
                )
                S_GT = S_GT.reshape(self.total_views, self.total_frames, self.num_joints, 3)
            else:
                S_Pred_ = normalize_3d_structure(S_Pred_)
                PA_MPJPE_3D = None

            # 2D Error
            W_GT = W_GT + T_Offset
            error_2D = (
                error_ratio(W_Pred_[flag_nan], W_GT[flag_nan]) * self.dataset_metric_multiplier
            )
            errors = {"2D_Error_Ratio": error_2D, "3D_PA_MPJPE": PA_MPJPE_3D}
            S_Pred_ = S_Pred_.reshape(self.total_views, self.total_frames, self.num_joints, 3)
            W_Pred_ = W_Pred_.reshape(self.total_views, self.total_frames, self.num_joints, 2)
            W_GT = W_GT.reshape(self.total_views, self.total_frames, self.num_joints, 2)
            W_Input = W_Input.reshape(self.total_views, self.total_frames, self.num_joints, 2)

        return errors, W_Pred_, S_Pred_


def save_packets(pkl_name, store_data):
    keys = [k for k, v in store_data.items()]
    with open(str(pkl_name), "rb") as fid:
        pickled_data = cPickle.load(fid)
    for frame_idx in range(len(pickled_data)):
        for key_idx in range(len(keys)):
            pickled_data[frame_idx][keys[key_idx]] = store_data[keys[key_idx]][frame_idx]
    with open(str(pkl_name), "wb") as fid:
        cPickle.dump(pickled_data, fid)


def save_bbox_only(pkl_name, BBox):
    with open(str(pkl_name), "rb") as fid:
        pickled_data = cPickle.load(fid)
    for frame_idx in range(len(pickled_data)):
        pickled_data[frame_idx]["BBox"] = BBox[frame_idx]
    with open(str(pkl_name), "wb") as fid:
        cPickle.dump(pickled_data, fid)


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
    if "S_GT" in pickled_data[0]:
        GT_Flag = True
    else:
        GT_Flag = False
    # Find number of joints/keypoints
    num_joints = pickled_data[0]["W_GT"].shape[0]

    W, W_GT, S_GT, T_Offset, W_Inliers, confidence_original = [], [], [], [], [], []
    # Iterate over views
    for cam_idx in range(total_views):
        with open(pkl_names[cam_idx], "rb") as fid:
            pickled_data = cPickle.load(fid)
        # Iterate over frames
        W_, W_GT_, S_GT_, T_Offset_, W_Inliers_, confidence_original_ = [], [], [], [], [], []
        for frame_idx in range(total_frames):
            W_.append(
                pickled_data[frame_idx][
                    "W_" + str(args.input_source) + "_Iter_" + str(args.MBW_Iteration)
                ]
            )
            W_GT_.append(pickled_data[frame_idx]["W_GT"])
            W_Inliers_.append(pickled_data[frame_idx]["W_Inliers"])
            confidence_original_.append(pickled_data[frame_idx]["confidence"])
            if GT_Flag:
                S_GT_.append(pickled_data[frame_idx]["S_GT"])
        W_ = np.asarray(W_)
        W_GT_ = np.asarray(W_GT_)
        W_Inliers_ = np.asarray(W_Inliers_)
        confidence_original_ = np.asarray(confidence_original_)
        S_GT_ = np.asarray(S_GT_)

        # Remove translation: Enforce object-centric assumptipn
        # T_Offset_ = W_GT_.mean(1, keepdims=True)
        T_Offset_ = W_.mean(1, keepdims=True)
        W_ = W_ - W_.mean(1, keepdims=True)
        # W_GT_ = W_GT_ - T_Offset_
        W_GT_ = W_GT_ - W_GT_.mean(1, keepdims=True)

        if GT_Flag:
            S_GT_ = S_GT_ - S_GT_.mean(1, keepdims=True)

        W.append(torch.from_numpy(W_).to(DEVICE).float())
        W_GT.append(torch.from_numpy(W_GT_).to(DEVICE).float())
        S_GT.append(torch.from_numpy(S_GT_).to(DEVICE).float())
        T_Offset.append(torch.from_numpy(T_Offset_).to(DEVICE).float())
        W_Inliers.append(W_Inliers_)
        confidence_original.append(confidence_original_)

        data_loaded = (W, T_Offset, W_GT, S_GT)

    W_Inliers = np.asarray(W_Inliers)
    confidence_original = np.asarray(confidence_original)

    # MV-NRSfM Dictionary
    dict_list = get_dictionary()

    # Load the Adaptive Loss function Kernel
    adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=num_joints * 2, float_dtype=torch.float32, device=DEVICE
    )

    # Load the MV-NRSfM Kernel
    mv_nrsfm_kernel = MVNRSfM(
        args,
        GT_Flag,
        DEVICE,
        num_joints=num_joints,
        total_views=total_views,
        total_frames=total_frames,
        dict_list=dict_list,
    )
    mv_nrsfm_kernel.to(DEVICE)

    # Set optimizer and scheduler
    params = list(mv_nrsfm_kernel.parameters()) + list(adaptive.parameters())
    optimizer = optim.Adam(params, lr=mv_nrsfm_kernel.lr)

    # Set scheduler and seed
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=np.arange(
            mv_nrsfm_kernel.lr_decay_step,
            max(mv_nrsfm_kernel.max_itr, mv_nrsfm_kernel.lr_decay_step),
            mv_nrsfm_kernel.lr_decay_step,
        ).tolist(),
        gamma=mv_nrsfm_kernel.lr_decay_rate,
    )
    torch.manual_seed(get_seed())

    # Run MV-NRSfM
    random_index = []
    num_min_batch = W[0].shape[0] // mv_nrsfm_kernel.batch
    random_index = torch.randperm(W[0].shape[0])

    Errors, output_mvnrsfm, output_3D = mv_nrsfm_kernel.validation(data_loaded)

    input_mvnrsfm = torch.cat(data_loaded[0], dim=0) + torch.cat(data_loaded[1], dim=0)
    input_mvnrsfm = input_mvnrsfm.reshape(total_views, total_frames, num_joints, 2)

    img_paths = "data/" + str(args.dataset) + "/images/CAM_1/"
    img_names = []
    for path in Path(img_paths).rglob("*" + str(args.img_type)):
        img_names.append(path)
    img_names = [Path(p) for p in natsorted([str(p) for p in img_names])]

    shapes_imgs = img.imread(str(img_names[0])).shape
    img_height = shapes_imgs[0]
    img_width = shapes_imgs[1]

    # import pdb
    # pdb.set_trace()

    for cam_idx in range(total_views):
        confidence = np.zeros((total_frames))

        input_scaled = normalize_3d_structure(input_mvnrsfm[cam_idx, :, :, :])
        output_scaled = normalize_3d_structure(output_mvnrsfm[cam_idx, :, :, :])

        input_scaled = input_scaled - input_scaled.mean(1, keepdims=True)
        output_scaled = output_scaled - output_scaled.mean(1, keepdims=True)

        # reprojection_error = torch.nan_to_num(computeMPJPE_array(output_scaled, input_scaled))

        reprojection_error = computeMPJPE_array(output_scaled, input_scaled)
        # pdb.set_trace()

        reprojection_error = (reprojection_error - min(torch.nan_to_num(reprojection_error))) / (
            max(torch.nan_to_num(reprojection_error)) - min(torch.nan_to_num(reprojection_error))
        )

        reprojection_error = reprojection_error.detach().cpu().numpy()
        # pdb.set_trace()

        confidence[np.argwhere(reprojection_error < args.outlier_threshold)] = True

        "save_packets"
        "logic: store original inliers as it is. Add on the new stuff"
        W_Inliers_updated = np.zeros((total_frames, num_joints, 2))
        W_Inliers_updated[:] = np.nan
        W_MV_Inference = np.zeros((total_frames, num_joints, 2))
        S_MV_Inference = np.zeros((total_frames, num_joints, 3))
        confidence_updated = np.zeros((total_frames))
        BBox_updated = np.zeros((total_frames, 4))

        if args.update_bbox_only:
            for frame_idx in range(total_frames):
                if confidence_original[cam_idx, frame_idx]:
                    BBox_updated[frame_idx, :] = calculate_BBox(W_Inliers[cam_idx, frame_idx, :, :])
                else:
                    BBox_updated[frame_idx, :] = calculate_BBox(
                        input_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                    )
                    # BBox_updated[frame_idx, :] = calculate_BBox(
                    #     output_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                    # )
            save_bbox_only(pkl_names[cam_idx], BBox_updated)

        else:
            for frame_idx in range(total_frames):
                W_MV_Inference[frame_idx, :, :] = (
                    output_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                )
                S_MV_Inference[frame_idx, :, :] = (
                    output_3D[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                )

                if confidence_original[cam_idx, frame_idx]:
                    W_Inliers_updated[frame_idx, :, :] = W_Inliers[cam_idx, frame_idx, :, :]
                    confidence_updated[frame_idx] = bool(confidence_original[cam_idx, frame_idx])
                    # if args.input_source == "Flow":
                    #     BBox_updated[frame_idx, :] = [img_width, 0, img_height, 0]
                    # else:
                    #     BBox_updated[frame_idx, :] = calculate_BBox(
                    #         W_Inliers[cam_idx, frame_idx, :, :]
                    #     )
                    BBox_updated[frame_idx, :] = calculate_BBox(W_Inliers[cam_idx, frame_idx, :, :])

                else:
                    if args.input_source == "Flow":
                        # BBox_updated[frame_idx, :] = [img_width, 0, img_height, 0]
                        BBox_updated[frame_idx, :] = calculate_BBox(
                            output_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                        )
                    elif args.input_source == "Detector":
                        BBox_updated[frame_idx, :] = calculate_BBox(
                            input_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                        )
                        # BBox_updated[frame_idx, :] = calculate_BBox(
                        #     output_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                        # )
                    if confidence[frame_idx]:
                        confidence_updated[frame_idx] = True
                        if args.trust_MVNRSfM_Inliers:
                            W_Inliers_updated[frame_idx, :, :] = (
                                output_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                            )
                        else:
                            W_Inliers_updated[frame_idx, :, :] = (
                                input_mvnrsfm[cam_idx, frame_idx, :, :].detach().cpu().numpy()
                            )

            infer_str_mv = [
                "W_MV_Iter_" + str(args.MBW_Iteration),
                "S_MV_Iter_" + str(args.MBW_Iteration),
            ]

            if args.update_pickle:
                if args.leave_bbox_out:
                    store_data = {
                        "W_Inliers": W_Inliers_updated,
                        "confidence": confidence_updated,
                        infer_str_mv[0]: W_MV_Inference,
                        infer_str_mv[1]: S_MV_Inference,
                        "reprojection_error_Iter_" + str(args.MBW_Iteration): reprojection_error,
                    }
                else:
                    store_data = {
                        "W_Inliers": W_Inliers_updated,
                        "BBox": BBox_updated,
                        "confidence": confidence_updated,
                        infer_str_mv[0]: W_MV_Inference,
                        infer_str_mv[1]: S_MV_Inference,
                        "reprojection_error_Iter_" + str(args.MBW_Iteration): reprojection_error,
                    }

                save_packets(pkl_names[cam_idx], store_data)

            if not args.unittest:
                print(
                    "--------------------------------------------------------------------------------"
                )
                print(
                    "Total percentage labeled: {:.2f}%".format(
                        (np.argwhere(confidence_updated == True).shape[0] / total_frames) * 100
                    )
                )

    if args.unittest:
        print("-------------------------------")
        print("Outlier Rejection Unit Test: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Parameters: Infer Shape and reject outliers """
    parser.add_argument("--mode", default="Test")
    parser.add_argument("--dataset", help="dataset selection", default="Colobus_Monkey")
    parser.add_argument("--datasetunittest", help="dataset name", default="Colobus_Monkey")
    parser.add_argument("--model", help="model selection", default="model/mvnrsfm/Colobus_Monkey")
    parser.add_argument("--batch", default=100, type=int)
    parser.add_argument("--MBW_Iteration", default=0, type=int)
    parser.add_argument("--img_type", help="image format", default=".jpg")
    parser.add_argument(
        "--input_source", help="input to MV-NRSfM. Choose b/w Flow and Detector", default="Flow"
    )
    parser.add_argument("--outlier_threshold", default=0.001, type=float)
    parser.add_argument("--break_training_counter", default=10000, type=int)
    parser.add_argument(
        "--trust_MVNRSfM_Inliers",
        type=parse_boolean,
        default=False,
        help="Flag for using MVNRSfM Inliers.",
    )
    parser.add_argument(
        "--from_scratch",
        type=parse_boolean,
        default=False,
        help="Redundant here since we are inferring over a learned MVNRSfM model.",
    )
    parser.add_argument(
        "--update_pickle",
        type=parse_boolean,
        default=False,
        help="Flag to overwrite the pickle data with new predictions.",
    )
    parser.add_argument(
        "--update_bbox_only",
        type=parse_boolean,
        default=False,
        help="As name suggests, only update bbox in the new pickle.",
    )
    parser.add_argument(
        "--leave_bbox_out",
        type=parse_boolean,
        default=False,
        help=".",
    )
    parser.add_argument(
        "--unittest",
        type=parse_boolean,
        default=False,
        help="Flag for unit test.",
    )

    parser.add_argument("--dataset_metric_multiplier", default=1, type=int)
    args = parser.parse_args()

    if args.unittest:
        args.dataset = args.datasetunittest

    main(args)
