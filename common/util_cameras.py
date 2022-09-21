import sys

sys.path.append("modules/helpers/torch-batch-svd")


import _pickle as cPickle
import numpy as np
import torch
from torch.nn.functional import conv2d, conv_transpose2d, relu

sys.path.append("common")
from util_common import get_device

DEVICE = get_device()
if DEVICE.type != "cpu":
    from torch_batch_svd import svd as batch_svd


def procrustes_align(pred, gt):
    scale = torch.norm(torch.norm(gt, dim=1), dim=1) / torch.norm(torch.norm(pred, dim=1), dim=1)
    out_scaled = torch.mul(pred, scale.view(-1, 1).repeat(1, gt.shape[1])[:, :, None])

    U, s, Vh = batch_svd(torch.matmul(torch.transpose(out_scaled, dim0=1, dim1=2), gt))

    return torch.matmul(out_scaled, torch.matmul(U, torch.transpose(Vh, dim0=1, dim1=2)))


def normalize_3d_structure(pred):
    scale = (torch.norm(torch.norm(pred, dim=1), dim=1) * 0) + 1 / torch.norm(
        torch.norm(pred, dim=1), dim=1
    )
    normalized = torch.mul(pred, scale.view(-1, 1).repeat(1, pred.shape[1])[:, :, None])
    return normalized


def make_orthonormal(input_mat, gpu_flag):
    """
    Function: Forces an orthonormal matrix.
    input: [-1 x 3 x 3]
    output: orthonormalized matrix with same size
    """
    batch_size = input_mat.size(0)
    mat_col_num = input_mat.size(2)

    if gpu_flag:
        u, s, v = batch_svd(input_mat)
    else:
        u, s, v = torch.svd(input_mat)

    orth_mat = torch.matmul(u, v.transpose(1, 2))

    """ Check reflection """
    if orth_mat.shape[-1] == 3:
        """Flip if reflected"""
        orth_mat_det = batch_det_3x3(orth_mat)
        u_flip = torch.cat((u[..., :2], u[..., 2:3] * orth_mat_det.view(-1, 1, 1)), 2)
        orth_mat = torch.matmul(u_flip, v.transpose(1, 2))

    return orth_mat


def make_rotation_matrix(rotm_xy):
    """
    rotm_xy - [-1 x 3 x2]
    """
    rotm_z = rotm_xy[:, :, 0].cross(rotm_xy[:, :, 1])
    rotm = torch.cat((rotm_xy, rotm_z.unsqueeze(-1)), 2)
    return rotm


def onp(pts2d, pts3d):
    """
    input:
    - pts2d: -1xPx2 (W)
    - pts3d: -1xPx3 (S)
    output: find R -1x3x2 such that
      min_R |SR_xy - W|_F
    """
    inv_ST_S = torch.inverse(torch.matmul(pts3d.transpose(1, 2), pts3d))
    pinv_S = inv_ST_S.matmul(pts3d.transpose(1, 2))
    rotm_unrectify = pinv_S.matmul(pts2d)

    if DEVICE.type != "cpu":
        u, s, v = batch_svd(rotm_unrectify)
    else:
        u, s, v = torch.svd(rotm_unrectify)

    orth_mat = torch.matmul(u, v.transpose(1, 2))
    rotm = make_rotation_matrix(orth_mat)

    pts2d_recon = pts3d.matmul(orth_mat)
    scale = (pts2d * pts2d_recon).sum(dim=(1, 2)) / (pts2d_recon**2).sum(dim=(1, 2))
    pts2d_recon_scaled = pts2d_recon * (scale.view(scale.shape[0], 1, 1))
    return pts2d_recon_scaled, pts3d, orth_mat, rotm
