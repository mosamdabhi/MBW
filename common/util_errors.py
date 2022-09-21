import _pickle as cPickle
import numpy as np
import torch


def process_data_type(pred, gt):
    """Convert everything into Tensors"""
    np_flag = False
    if not torch.is_tensor(pred) and not torch.is_tensor(gt):
        np_flag = True
        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
    return pred, gt, np_flag


def computeMPJPE(pred, gt):
    """Gives a scalar value of mean-per-joint-position.
    This is a un-normalized error metric."""
    pred, gt, np_flag = process_data_type(pred, gt)
    error = (pred - gt).norm(dim=2).mean(-1).mean()
    if np_flag:
        error = error.detach().cpu().numpy()
    return error


def computeMPJPE_array(pred, gt):
    """Gives an array of mean-per-joint-position.
    This is a un-normalized error metric."""
    pred, gt, np_flag = process_data_type(pred, gt)
    error = (pred - gt).norm(dim=2).mean(-1)
    if np_flag:
        error = error.detach().cpu().numpy()
    return error


def error_ratio(pred, gt):
    """Gives a scalar value of error ratio.
    This is a normalized error metric."""
    pred, gt, np_flag = process_data_type(pred, gt)
    error = ((pred - gt).norm(dim=(1, 2)) / gt.norm(dim=(1, 2))).mean()
    if np_flag:
        error = error.detach().cpu().numpy()
    return error


def error_ratio_array(pred, gt):
    """Gives an array of error ratio.
    This is a normalized error metric."""

    pred, gt, np_flag = process_data_type(pred, gt)
    error = (pred - gt).norm(dim=(1, 2)) / gt.norm(dim=(1, 2))
    if np_flag:
        error = error.detach().cpu().numpy()
    return error


def huber_loss(pred, gt):
    error = (pred - gt).norm(dim=(1, 2)) / (gt.norm(dim=(1, 2)))
    _l1 = torch.abs(error)
    _quad = torch.min(_l1, torch.tensor(0.1).cuda().float())
    _lin = _l1 - _quad

    output = (torch.tensor(0.5).cuda().float() * (_quad**2)) + (
        torch.tensor(0.1).cuda().float() * (_lin)
    )
    return (output * torch.tensor(1.0).cuda().float()).mean()
