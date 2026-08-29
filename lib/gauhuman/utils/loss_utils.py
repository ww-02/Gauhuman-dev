#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.ndimage.morphology import distance_transform_edt
from pytorch3d.ops.knn import knn_points

def get_loss_render(viewpoint, gaussians, renderFunc, renderArgs, debug=False):
    gaussians.update_shs(viewpoint)
    render_output = renderFunc(viewpoint, gaussians, *renderArgs, return_smpl_rot=True)
    image = torch.clamp(render_output["render"], 0.0, 1.0)
    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
    if debug:
        cv2.imwrite("image.png",image.cpu().detach().numpy().transpose(1,2,0) * 255)
        cv2.imwrite("gt_image.png",gt_image.cpu().numpy().transpose(1,2,0) * 255)
    return l1_loss(image, gt_image).mean()
    
def get_loss_similar(org, current):
    return ((org - current) ** 2).mean()

def get_loss_vertex_in_mask(vertex_2d, bkgd_mask, H, W, device="cuda:0"):
    vertex_2d_uv = 2.0 * vertex_2d.unsqueeze(2).type(torch.float32) / torch.Tensor([W, H]).to(device) - 1.0
    human_out_mask_error_map = distance_transform_edt(1 - bkgd_mask.detach().cpu().numpy())
    human_out_mask_error_map = torch.from_numpy(human_out_mask_error_map[None,None]).float().to(device)
    loss_vertex_in_mask = torch.nn.functional.grid_sample(human_out_mask_error_map, vertex_2d_uv, mode="bilinear", padding_mode="zeros") ** 2
    return loss_vertex_in_mask.mean()

def get_loss_mask_to_vertex(vertex_2d, bkgd_mask, H, W, device="cuda:0"):
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    img_grid = torch.from_numpy(np.array([x,y])).reshape(1, 2,-1).permute(0, 2, 1).float().to(device)
    dists, idx, _ = knn_points(p1=img_grid,p2=vertex_2d)
    loss_mask_to_vertex = (dists.reshape(1, -1) * bkgd_mask.reshape(1, -1)) ** 2
    return loss_mask_to_vertex.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

from skimage.metrics import structural_similarity
import lpips
import cv2
import numpy as np
loss_fn_vgg = lpips.LPIPS(net='vgg')

def ssim_metric(rgb_pred, rgb_gt, mask_at_box, H, W):
    # convert the pixels into an image
    img_pred = np.zeros((H, W, 3))
    img_pred[mask_at_box] = rgb_pred.T
    img_gt = np.zeros((H, W, 3))
    img_gt[mask_at_box] = rgb_gt.T

    # crop the object region
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]
    
    # compute the ssim
    ssim = structural_similarity(img_pred, img_gt, channel_axis=-1,data_range=1.0)
    lpips = loss_fn_vgg(torch.from_numpy(img_pred).permute(2, 0, 1).to(torch.float32), torch.from_numpy(img_gt).permute(2, 0, 1).to(torch.float32)).reshape(-1).item()

    return ssim, lpips


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _to_4d(x):
    """(H, W) or (1, H, W) or (1, 1, H, W) -> (1, 1, H, W)."""
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def silhouette_edge_mask(mask, k=7):
    """Binary band around the silhouette boundary via morphological dilation - erosion.

    mask: (H, W) or (1, H, W) in [0, 1]. Returns (1, H, W) edge band in [0, 1].
    """
    m = _to_4d(mask.float())
    pad = k // 2
    dilated = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=pad)
    band = (dilated - eroded).clamp(0.0, 1.0)
    return band.squeeze(1)


def sil_edge_loss(alpha, gt_mask, weight=None, k=7):
    """Weighted L1 between rendered alpha and gt mask, restricted to the boundary band.

    weight: optional (H, W) or (1, H, W) emphasis map (e.g. joint heatmap).
    """
    a = _to_4d(alpha.float())
    g = _to_4d(gt_mask.float())
    band = _to_4d(silhouette_edge_mask(gt_mask, k))
    w = _to_4d(weight.float()) if weight is not None else torch.ones_like(band)
    diff = ((a - g).abs() * band * w).sum()
    denom = (band * w).sum().clamp_min(1e-6)
    return diff / denom


def _sobel_grad(x):
    """Sobel gradient magnitude of a (H, W) / (1, H, W) tensor -> (1, 1, H, W)."""
    x4 = _to_4d(x.float())
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      dtype=x4.dtype, device=x4.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x4, kx, padding=1)
    gy = F.conv2d(x4, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def sil_grad_loss(alpha, gt_mask, weight=None):
    """L1 between Sobel gradient magnitudes of rendered alpha and gt mask."""
    ga = _sobel_grad(alpha)
    gm = _sobel_grad(gt_mask)
    d = (ga - gm).abs()
    if weight is not None:
        d = d * _to_4d(weight.float())
    return d.mean()