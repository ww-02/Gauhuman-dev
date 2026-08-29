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

import os
import torch
from utils.loss_utils import l1_loss, l2_loss, ssim, sil_edge_loss, sil_grad_loss
from gaussian_renderer import render, network_gui
from nets.joint_psd import pose_residual_loss, IMPORTANT_JOINTS_SMPL
import sys
from scene import Scene, GaussianModel
from nets.dyn_skin import DynamicSkinWeights
from utils.general_utils import safe_state
import numpy as np
import cv2
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

from datasets.wildavatar_dataset import WildAvatarDatasetBatch
import time
from utils.loader_utils import InfiniteSampler, collate_fn, data_to_device
from test import test_single
import torch.nn.functional as F

HAND_JOINTS = {20, 21}


def build_joint_heatmap(viewpoint_cam, gaussians, sigma, hand_boost, joint_indices, hand_joints):
    """Project SMPL world joints to pixels and build a Gaussian heatmap over the image.

    Returns (H, W) heatmap in [0, 1] on the Gaussian device, or None if joints are unavailable.
    """
    if sigma <= 0:
        return None
    H = viewpoint_cam.image_height
    W = viewpoint_cam.image_width

    J_reg = gaussians.SMPL_NEUTRAL['J_regressor']
    n_joints = J_reg.shape[0]
    if n_joints == 0:
        return None
    dev = J_reg.device

    wv = viewpoint_cam.world_vertex
    if wv is None:
        return None
    wv = torch.as_tensor(wv, dtype=torch.float32, device=dev)
    if wv.dim() == 3:
        wv = wv[0]
    if wv.dim() != 2 or wv.shape[1] != 3:
        return None

    joints_world = torch.matmul(J_reg.to(torch.float32), wv)  # (J, 3)

    R = torch.as_tensor(viewpoint_cam.R, dtype=torch.float32, device=dev)
    T = torch.as_tensor(viewpoint_cam.T, dtype=torch.float32, device=dev).reshape(-1)
    K = torch.as_tensor(viewpoint_cam.K, dtype=torch.float32, device=dev)
    cam = torch.matmul(joints_world, R) + T          # (J, 3)
    xy = torch.matmul(cam, K.t())                    # (J, 3)
    z = xy[:, 2].clamp_min(1e-6)
    px = xy[:, 0] / z
    py = xy[:, 1] / z

    yy = torch.arange(H, device=dev, dtype=torch.float32)
    xx = torch.arange(W, device=dev, dtype=torch.float32)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')   # (H, W)

    heat = torch.zeros(H, W, device=dev)
    var = 2.0 * sigma * sigma
    for j in joint_indices:
        if j >= n_joints:
            continue
        amp = hand_boost if j in hand_joints else 1.0
        dx = gx - px[j]
        dy = gy - py[j]
        heat = heat + amp * torch.exp(-(dx * dx + dy * dy) / var)

    mx = heat.max()
    if mx <= 0:
        return None
    return heat / mx


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from):
    first_iter = 0
    dataset.dataset_name = "WildAvatar"
    data_root = os.path.join("data/WildAvatar", dataset.source_path.split("/")[-1])
    train_dataset = WildAvatarDatasetBatch(data_root=data_root, poses_start=0, poses_interval=2, poses_num=10, white_back=dataset.white_background)
    train_dataloader = InfiniteSampler(dataset=train_dataset, rank=0, num_replicas=1, shuffle=True, seed=0)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_dataloader, batch_size=1, collate_fn=collate_fn, num_workers=12))
    
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender,
                              getattr(dataset, 'psd_pe_L', 4), getattr(dataset, 'psd_n_layers', 2))
    gaussians.use_knn_soft = not getattr(opt, "no_knn_soft", False)
    scene = Scene(dataset, gaussians)
    if getattr(opt, "use_dyn_skin", False):
        gaussians.dyn_skin_net = DynamicSkinWeights(num_joints=24, pos_pe_L=4, hidden_dim=128, n_layers=3).cuda()

    gaussians.training_setup(opt)
    if getattr(opt, "use_dyn_skin", False):
        gaussians.optimizer.add_param_group({
            "params": list(gaussians.dyn_skin_net.parameters()),
            "lr": 1e-3,
            "name": "dyn_skin"
        })
        print("[dyn_skin] attached DynamicSkinWeights (single-frame, RnD-inspired).")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    ssim_loss_for_log = 0.0
    lpips_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    elapsed_time = 0
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        viewpoint_cam = next(training_set_iterator)
        viewpoint_cam = data_to_device(viewpoint_cam)
        # Render
        if iteration == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, disable_psd=opt.disable_psd)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        d_xyz, d_scale, d_rot, d_alpha = render_pkg["d_xyz"], render_pkg["d_scale"], render_pkg["d_rot"], render_pkg["d_alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()
        Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
        mask_loss = l2_loss(alpha[bound_mask==1], bkgd_mask[bound_mask==1])

        # hand / joint emphasis heatmap (2D projection of SMPL world joints)
        joint_extra = None
        if iteration >= opt.sil_joint_start_iter and opt.joint_region_weight > 0:
            heatmap = build_joint_heatmap(viewpoint_cam, gaussians, opt.joint_heatmap_sigma,
                                          opt.hand_region_boost, IMPORTANT_JOINTS_SMPL, HAND_JOINTS)
            if heatmap is not None:
                joint_extra = opt.joint_region_weight * heatmap  # (H, W)

        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        # ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        # lipis loss
        if getattr(opt, "lpips_shallow", False):
            _, _per_layer = loss_fn_vgg(img_pred, img_gt, retPerLayer=True)
            # shallow emphasis: relu1..relu5 weights (downweight deep semantics)
            _w = [2.0, 1.5, 1.0, 0.5, 0.3]
            lpips_loss = sum(_w[i] * _per_layer[i].reshape(-1) for i in range(len(_per_layer)))
        else:
            lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)

        loss = Ll1 + 0.1 * mask_loss + opt.lambda_ssim * (1.0 - ssim_loss) + opt.lambda_lpips * lpips_loss

        # contour sharpening: silhouette boundary band + gradient match
        if iteration >= opt.sil_joint_start_iter:
            if opt.sil_edge_weight > 0:
                edge_w = (1.0 + joint_extra).unsqueeze(0) if joint_extra is not None else None
                loss = loss + opt.sil_edge_weight * sil_edge_loss(alpha, bkgd_mask, weight=edge_w)
            if opt.sil_grad_weight > 0:
                loss = loss + opt.sil_grad_weight * sil_grad_loss(alpha, bkgd_mask)

        # extra L1 emphasis on hand / joint regions inside the body bounds
        if joint_extra is not None:
            inside = bound_mask[0] == 1
            pix_diff = (image - gt_image).abs()  # (C, H, W)
            joint_l1 = (pix_diff.permute(1, 2, 0)[inside] * joint_extra[inside].unsqueeze(1)).mean()
            loss = loss + joint_l1

        if d_xyz is not None:
            loss = loss + opt.lambda_psd_res * pose_residual_loss(d_xyz, d_scale, d_rot, d_alpha)
            if opt.joint_offset_reg > 0:
                off_norm = d_xyz.norm(dim=-1)
                loss = loss + opt.joint_offset_reg * (F.relu(off_norm - opt.max_joint_offset) ** 2).mean()
        # ------ AIAP loss (as-isometric-as-possible, per 3DGS-Avatar) ------
        if opt.lambda_aiap > 0:
            _xc = render_pkg["means3D_canon"]  # (N, 3)
            _xo = render_pkg["means3D_obs"]    # (N, 3)
            if _xc.dim() == 3:
                _xc = _xc.squeeze(0)
            if _xo.dim() == 3:
                _xo = _xo.squeeze(0)
            _N = _xc.shape[0]
            _S = min(1024, _N)
            _idx = torch.randperm(_N, device=_xc.device)[:_S]
            _q_c = _xc[_idx]; _q_o = _xo[_idx]
            # kNN in canonical space, k=6 to skip self (nearest = query itself)
            _, _nn = gaussians.knn_soft(_xc[None].detach(), _q_c[None].detach())
            _nn = _nn[0]  # (S, 4)
            _dc = (_q_c.unsqueeze(1) - _xc[_nn]).norm(dim=-1)  # (S, 4)
            _do = (_q_o.unsqueeze(1) - _xo[_nn]).norm(dim=-1)  # (S, 4)
            aiap_loss = (_dc - _do).abs().mean()
            loss = loss + opt.lambda_aiap * aiap_loss
        # ------ Rotation rigidity prior (AniGaussian lambda_rot): local d_rot consistency ------
        if opt.lambda_rot > 0 and d_rot is not None:
            _dr = d_rot.squeeze(0) if d_rot.dim() == 3 else d_rot          # (N, 3)
            _xyz = gaussians.get_xyz.detach()                              # (N, 3) canonical
            _N = _dr.shape[0]
            _S = min(1024, _N)
            _idx = torch.randperm(_N, device=_dr.device)[:_S]
            _q_dr = _dr[_idx]                                              # (S, 3)
            _, _nn = gaussians.knn_soft(_xyz[None], _xyz[_idx][None])      # (1, S, 4)
            _nn = _nn[0]                                                   # (S, 4)
            rot_loss = (_q_dr.unsqueeze(1) - _dr[_nn]).norm(dim=-1).mean()
            loss = loss + opt.lambda_rot * rot_loss
        loss.backward()
        
        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * mask_loss_for_log
            ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
            lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in testing_iterations:
                with torch.no_grad():   
                    test_single(tb_writer, scene, render, (args, background), visualing=True, args=args)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration)
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 1)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.exp_name == "":
        args.exp_name = args.source_path.replace("data/", "")
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")