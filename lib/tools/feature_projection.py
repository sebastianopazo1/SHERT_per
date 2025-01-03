'''
This includes the module to project the image colors to the reconstructed mesh.
Part of the codes is adapted from https://github.com/shunsukesaito/PIFu
'''

import numpy as np
import torch
import cv2
import os
import open3d as o3d
def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    # Ensure uv coordinates are within valid range
    uv = torch.clamp(uv, min=-1.0, max=1.0)
    
    # Reshape for grid_sample
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    
    # Use grid_sample with bounds checking
    samples = torch.nn.functional.grid_sample(
        feat, 
        uv, 
        mode='bilinear',
        padding_mode='border',  # Use border padding to handle out-of-bounds
        align_corners=True
    )
    
    return samples[:, :, :, 0]





