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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from nets.joint_psd import apply_rotation_correction

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None, disable_psd=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz

    d_xyz = None
    d_scale = None
    d_rot = None
    d_alpha = None

    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl_param,
            viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:
            # pose offset
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D[None].detach())
            lbs_weights = lbs_weights.permute(0,2,1)

            # per-joint pose-space correction (canonical space)
            d_xyz, d_scale, d_rot, d_alpha = pc.pose_space_deform(
                means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param, correct_Rs)
            d_xyz = d_xyz.squeeze(0)
            d_scale = d_scale.squeeze(0)
            d_rot = d_rot.squeeze(0)
            d_alpha = d_alpha.squeeze(0)
            if disable_psd:
                d_xyz = torch.zeros_like(d_xyz)
                d_scale = torch.zeros_like(d_scale)
                d_rot = torch.zeros_like(d_rot)
                d_alpha = torch.zeros_like(d_alpha)
            means3D_corrected = means3D + d_xyz

            # transform points
            _, means3D, _, transforms, translation = pc.coarse_deform_c2source(means3D_corrected[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
        else:
            # cached transforms (eval): apply PSD without recomputing knn/LBS.
            # pose_space_deform only does forward kinematics + tiny MLPs (no knn /
            # blendshape), so eval stays fast while still applying the correction.
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            correct_Rs = pc.pose_decoder(dst_posevec)['Rs']
            d_xyz, d_scale, d_rot, d_alpha = pc.pose_space_deform(
                means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param, correct_Rs)
            d_xyz = d_xyz.squeeze(0)
            d_scale = d_scale.squeeze(0)
            d_rot = d_rot.squeeze(0)
            d_alpha = d_alpha.squeeze(0)
            if disable_psd:
                d_xyz = torch.zeros_like(d_xyz)
                d_scale = torch.zeros_like(d_scale)
                d_rot = torch.zeros_like(d_rot)
                d_alpha = torch.zeros_like(d_alpha)
            means3D = torch.matmul(transforms, (means3D + d_xyz)[..., None]).squeeze(-1) + translation


    means3D = means3D.squeeze()
    means2D = screenspace_points
    opacity = pc.get_opacity
    if d_alpha is not None:
        # logit-space opacity correction (identity at init; d_alpha tanh-clamped)
        opacity = torch.sigmoid(pc._opacity + d_alpha)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_corrected(scaling_modifier, transforms.squeeze(), d_scale, d_rot)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if d_scale is not None:
            scales = scales * torch.exp(d_scale)
        if d_rot is not None:
            rotations = apply_rotation_correction(rotations, d_rot)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # Expose deformed means for AIAP-style regularisation in train.py.
    # means3D at this point is post-PSD, post-LBS: observation space.
    means3D_obs = means3D
    # canonical-space means BEFORE LBS but AFTER PSD (self-consistent).
    means3D_canon = pc.get_xyz + d_xyz if d_xyz is not None else pc.get_xyz
    return {"render": rendered_image,
            "means3D_obs": means3D_obs,
            "means3D_canon": means3D_canon,
            "render_depth": depth,
            "render_alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,
            "d_xyz": d_xyz,
            "d_scale": d_scale,
            "d_rot": d_rot,
            "d_alpha": d_alpha,}
