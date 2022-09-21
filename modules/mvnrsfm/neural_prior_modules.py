import sys

import torch
import torch.nn as nn
from torch.nn.functional import conv2d, conv_transpose2d, relu

sys.path.append("common")
from util_cameras import make_orthonormal, make_rotation_matrix, onp
from util_common import relu_threshold


class VecSparseCodingLayer(nn.Module):
    def __init__(self, dict_size, pts_num, share_weight=True):
        super(VecSparseCodingLayer, self).__init__()
        self.dict_size = dict_size
        self.pts_num = pts_num
        self.share_weight = share_weight
        dictionary = torch.empty(pts_num, 3, dict_size)
        nn.init.kaiming_uniform_(dictionary)
        self.dictionary = nn.Parameter(dictionary)
        if not share_weight:
            self.dictionary_decode = nn.Parameter(dictionary.clone())

        self.bias_encode_with_cam = nn.Parameter(torch.zeros(dict_size))
        self.bias_decode = nn.Parameter(torch.zeros(pts_num * 3))
        self.encode_with_cam_thresh_func = relu_threshold

    def encode_with_cam(self, pts):
        """
        pts: [-1 x pts_num x 3]
        """
        dictionary = self.dictionary

        """ [pts_num x dict_size x 3 x 1] """
        conv_weights = dictionary.transpose(1, 2).unsqueeze(-1)

        """ [-1 x pts_num x 1 x 3] """
        block_code = conv_transpose2d(pts.unsqueeze(-2), conv_weights, stride=1, padding=0)

        return self.encode_with_cam_thresh_func(block_code, self.bias_encode_with_cam)

    def decode(self, code):
        """
        code: [-1 x dim (x 1)]
        output: [-1 x pts_num x 3]
        """
        if self.share_weight:
            dictionary = self.dictionary
        else:
            dictionary = self.dictionary_decode

        dict_final_pass = dictionary.view(-1, self.dict_size)

        return (
            conv2d(
                code.view(-1, self.dict_size, 1, 1),
                dictionary.view(-1, self.dict_size, 1, 1),
                self.bias_decode,
            ).view(-1, self.pts_num, 3),
            dict_final_pass,
        )


class BlockSparseCodingLayer(nn.Module):
    def __init__(self, dict_size, input_dim, share_weight=True):
        super(BlockSparseCodingLayer, self).__init__()

        """ dictionary is of size dict_size x input_dim """

        self.dict_size = dict_size
        self.input_dim = input_dim
        # self.encode_with_relu = encode_with_relu

        dictionary = torch.empty(input_dim, dict_size, 1, 1)
        nn.init.kaiming_uniform_(dictionary.view(input_dim, dict_size))

        self.dictionary = nn.Parameter(dictionary)
        self.bias_encode_with_cam = nn.Parameter(torch.zeros(dict_size))
        self.bias_decode = nn.Parameter(torch.zeros(input_dim))

        self.encode_with_cam_thresh_func = relu_threshold
        self.ae_thresh_func = relu_threshold
        self.share_weight = share_weight

        if not self.share_weight:
            self.dictionary_decode = nn.Parameter(dictionary.clone())

    def encode_with_cam(self, input):
        """
        input: PoseCodeCalibrateLayer[-1 x input_dim x 3 x 3]
        """

        block_code = conv_transpose2d(
            input, self.dictionary.view(self.input_dim, self.dict_size, 1, 1), stride=1, padding=0
        )

        return self.encode_with_cam_thresh_func(block_code, self.bias_encode_with_cam)

    def decode(self, code):
        """
        code: [-1 x dict_size (x 1)]
        """
        if self.share_weight:
            dictionary = self.dictionary
        else:
            dictionary = self.dictionary_decode

        output = conv2d(
            code.view(-1, self.dict_size, 1, 1),
            dictionary.view(self.input_dim, self.dict_size, 1, 1),
            padding=0,
            stride=1,
        )

        return self.ae_thresh_func(output, self.bias_decode).view(-1, self.input_dim)


class CameraEstimator(nn.Module):
    def __init__(self, input_chan):
        super(CameraEstimator, self).__init__()
        self.linear_comb_layer = nn.Conv2d(input_chan, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, input):
        camera = self.linear_comb_layer(input).squeeze(1)
        """ Use SVD to make camera matrix be orthonormal """
        return make_orthonormal(camera, True)


class OnPEstimator(nn.Module):
    def __init__(self):
        super(OnPEstimator, self).__init__()

    def forward(self, pts_2d, pts_3d):

        pts2d_recon, pts3d_recon, cam_mat, rotm = onp(pts_2d, pts_3d)

        return pts2d_recon, pts3d_recon, cam_mat, rotm


class SparseCodeExtractionLayer(nn.Module):
    def __init__(self, code_dim, is_2d=True):
        super(SparseCodeExtractionLayer, self).__init__()
        if is_2d:
            kernel_size = [3, 2]
        else:
            kernel_size = [3, 3]
        self.fc_layer = nn.Conv2d(code_dim, code_dim, kernel_size, stride=1, bias=False)

    def forward(self, input):
        """
        input: [-1 x code_dim x 3 x 3]
        """
        return self.fc_layer(input)
