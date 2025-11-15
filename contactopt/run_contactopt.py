# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
for p in ['.', '..']:
    sys.path.append(p)
from contactopt.loader import ContactDBDataset
from contactopt.deepcontact_net import DeepContactNet
from contactopt.util import get_mano_closed_faces
import glob
import argparse
from contactopt.optimize_pose import optimize_pose
from contactopt.visualize import show_optimization, show_compare_optim
import pickle
from contactopt.hand_object import HandObject
import contactopt.util as util
from tqdm import tqdm
import contactopt.arguments as arguments
import time
import torch
import os
import yaml
from torch.utils.data import DataLoader
import pytorch3d
import numpy as np
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
import trimesh

from common.utils.utils import update_list_dict

from common.model.physics_constraint_solver import PhysicsConstraintSolver


def get_newest_checkpoint():
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    :return: Model with loaded weights
    """
    list_of_files = glob.glob('logs/contactopt_ckpt/*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    model = DeepContactNet()
    model.load_state_dict(torch.load(latest_file))
    return model


def run_contactopt(args):
    """
    Actually run ContactOpt approach. Estimates target contact with DeepContact,
    then optimizes it. Performs random restarts if selected.
    Saves results to a pkl file.
    :param args: input settings
    """
    print('Running split', args.split)
    dataset = ContactDBDataset(args.test_dataset, min_num_cont=args.min_cont)
    shuffle = args.vis or args.partial > 0
    print('Shuffle:', shuffle)
    args.batch_size = 256
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=8, collate_fn=ContactDBDataset.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_newest_checkpoint()
    model.to(device)
    model.eval()

    all_data = list()
    with open(os.path.join('configs', 'phys_constraint_solver', 'pcs_cfg.yaml')) as pcsyml:
        pcs_cfg = edict(yaml.load(pcsyml, Loader=yaml.SafeLoader))
    physics_solver = PhysicsConstraintSolver(pcs_cfg.pcs_cfg, 'data/mano_v1_2/')

    for idx, data in enumerate(tqdm(test_loader)):
        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        if args.split != 'fine':
            with torch.no_grad():
                network_out = model(data_gpu['hand_verts_aug'], data_gpu['hand_feats_aug'],
                                    data_gpu['obj_sampled_verts_aug'], data_gpu['obj_feats_aug'])
                hand_contact_target = util.class_to_val(network_out['contact_hand']).unsqueeze(2)
                obj_contact_target = util.class_to_val(network_out['contact_obj']).unsqueeze(2)
        else:
            hand_contact_target = data_gpu['hand_contact_gt']
            obj_contact_target = util.batched_index_select(data_gpu['obj_contact_gt'], 1, data_gpu['obj_sampled_idx'])

        if args.sharpen_thresh > 0: # If flag, sharpen contact
            print('Sharpening')
            obj_contact_target = util.sharpen_contact(obj_contact_target, slope=2, thresh=args.sharpen_thresh)
            hand_contact_target = util.sharpen_contact(hand_contact_target, slope=2, thresh=args.sharpen_thresh)

        ## do contact optimization with physical constraints:
        # obj_pts = data_gpu['obj_sampled_verts_aug']
        # sampled_idx = data_gpu['obj_sampled_idx']
        meshes = data_gpu['mesh_aug']
        obj_verts = meshes.verts_list()
        obj_faces = meshes.faces_list()
        # obj_pts_normal = []
        coms = []
        obj_meshes = []
        for i in range(batch_size):
            # obj_pts_normal.append(data_gpu['obj_normals_aug'][i, sampled_idx[i]])
            trim = trimesh.Trimesh(obj_verts[i].detach().cpu().numpy(), obj_faces[i].detach().cpu().numpy())
            coms.append(torch.from_numpy(trim.center_mass))
            obj_meshes.append(trim)

        coms = torch.stack(coms, dim=0).float().to(device)
        # obj_pts_normal = torch.stack(obj_pts_normal, dim=0)
        # for i in range(8):
        #     obj_contact_target = physics_solver.conatct_lp_update(obj_contact_target.squeeze(-1), obj_pts, obj_pts_normal, coms, step=0.1, device=device)
        # obj_contact_target = obj_contact_target.unsqueeze(-1)
        ## Debugging without random restarts
        if data_gpu['obj_sampled_idx'].numel() > 1:
            obj_normals_sampled = util.batched_index_select(data_gpu['obj_normals_aug'], 1, data_gpu['obj_sampled_idx'])
        else:  # If we're optimizing over all verts
            obj_normals_sampled = data_gpu['obj_normals_aug']
        pred_forces = physics_solver.qp_force_estimate(obj_contact_target.squeeze(-1), data_gpu['obj_sampled_verts_aug'].squeeze(-1),
                                                       obj_normals_sampled.squeeze(-1), coms)

        calculate_cmp = False
        calculate_phy = True
        if args.rand_re > 1:    # If we desire random restarts
            mtc_orig = data_gpu['hand_mTc_aug'].detach().clone()
            print('Doing random optimization restarts')
            best_loss = torch.ones(batch_size) * 100000

            for re_it in range(args.rand_re):
                # Add noise to hand translation and rotation
                data_gpu['hand_mTc_aug'] = mtc_orig.detach().clone()
                random_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(torch.randn((batch_size, 3), device=device) * args.rand_re_rot / 180 * np.pi, 'ZYX')
                data_gpu['hand_mTc_aug'][:, :3, :3] = torch.bmm(random_rot_mat, data_gpu['hand_mTc_aug'][:, :3, :3])
                data_gpu['hand_mTc_aug'][:, :3, 3] += torch.randn((batch_size, 3), device=device) * args.rand_re_trans

                if calculate_cmp:
                    opt_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
                                               w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis, ncomps=args.ncomps,
                                               w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans, w_opt_pose=args.w_opt_pose,
                                               w_opt_rot=args.w_opt_rot,
                                               caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                               caps_on_hand=args.caps_hand,
                                               contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                               w_obj_rot=args.w_obj_rot, pen_it=args.pen_it)

                if calculate_phy:
                    phy_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter,
                                               lr=args.lr, w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis,
                                               ncomps=args.ncomps, w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans,
                                               w_opt_pose=args.w_opt_pose, w_opt_rot=args.w_opt_rot,
                                               caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                               caps_on_hand=args.caps_hand,
                                               contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                               w_obj_rot=args.w_obj_rot, pen_it=args.pen_it, physics_solver=physics_solver,
                                               coms=coms, object_meshes=obj_meshes, hand_meshes=None, contact_weights=pred_forces)
                # out_pose2, out_mTc2, obj_rot2, opt_state = phy_result

                if re_it == 0:
                    if calculate_cmp:
                        out_pose1 = torch.zeros_like(opt_result[0])
                        out_mTc1 = torch.zeros_like(opt_result[1])
                        obj_rot1 = torch.zeros_like(opt_result[2])
                        opt_state = opt_result[3]
                    if calculate_phy:
                        out_pose2 = torch.zeros_like(phy_result[0])
                        out_mTc2 = torch.zeros_like(phy_result[1])
                        obj_rot2 = torch.zeros_like(phy_result[2])
                        phy_state = phy_result[3]

                if calculate_cmp:
                    loss_val = opt_result[3][-1]['loss']
                    for b in range(batch_size):
                        if loss_val[b] < best_loss[b]:
                            best_loss[b] = loss_val[b]
                            out_pose1[b, :] = opt_result[0][b, :]
                            out_mTc1[b, :, :] = opt_result[1][b, :, :]
                            obj_rot1[b, :, :] = opt_result[2][b, :, :]
                            opt_state = update_list_dict(opt_state, opt_result[3], b)

                if calculate_phy:
                    loss_val = phy_result[3][-1]['loss']
                    for b in range(batch_size):
                        if loss_val[b] < best_loss[b]:
                            best_loss[b] = loss_val[b]
                            out_pose2[b, :] = phy_result[0][b, :]
                            out_mTc2[b, :, :] = phy_result[1][b, :, :]
                            obj_rot2[b, :, :] = phy_result[2][b, :, :]
                            phy_state = update_list_dict(phy_state, phy_result[3], b)

                # print('Loss, re', re_it, loss_val)
                # print('Best loss', best_loss)
        else:
            if calculate_cmp:
                result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
                                       w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis, ncomps=args.ncomps,
                                       w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans, w_opt_pose=args.w_opt_pose,
                                       w_opt_rot=args.w_opt_rot,
                                       caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                       caps_on_hand=args.caps_hand,
                                       contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                       w_obj_rot=args.w_obj_rot, pen_it=args.pen_it)
                out_pose1, out_mTc1, obj_rot1, opt_state = result

            if calculate_phy:
                phy_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter,
                                           lr=args.lr,
                                           w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis,
                                           ncomps=args.ncomps,
                                           w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans,
                                           w_opt_pose=args.w_opt_pose,
                                           w_opt_rot=args.w_opt_rot,
                                           caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                           caps_on_hand=args.caps_hand,
                                           contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                           w_obj_rot=args.w_obj_rot, pen_it=args.pen_it, physics_solver=physics_solver,
                                           coms=coms, object_meshes=obj_meshes, hand_meshes=None, contact_weights=pred_forces)
                out_pose2, out_mTc2, obj_rot2, phy_state = phy_result
            # Fs = torch.sum(torch.concatenate([opt_state2[i]['forces'] for i in range(len(opt_state2))], dim=0), dim=-1)
            # plt.plot(Fs)
            # plt.show()

        obj_contact_upscale = util.upscale_contact(data_gpu['mesh_aug'], data_gpu['obj_sampled_idx'], obj_contact_target)

        for b in range(obj_contact_upscale.shape[0]):    # Loop over batch
            gt_ho = HandObject()
            in_ho = HandObject()
            opt_ho = HandObject()
            phy_ho = HandObject()
            gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'], b)
            in_ho.load_from_batch(data['hand_beta_aug'], data['hand_pose_aug'], data['hand_mTc_aug'], hand_contact_target, obj_contact_upscale, data['mesh_aug'], b)
            all_data.append({'gt_ho': gt_ho, 'in_ho': in_ho})
            if calculate_phy:
                phy_ho.load_from_batch(data['hand_beta_aug'], out_pose2, out_mTc2, data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], b, obj_rot=obj_rot2)
                phy_ho.calc_dist_contact(hand=True, obj=True)
                all_data[-1]['phy_ho'] = phy_ho

            if calculate_cmp:
                opt_ho.load_from_batch(data['hand_beta_aug'], out_pose1, out_mTc1, data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], b, obj_rot=obj_rot1)
                opt_ho.calc_dist_contact(hand=True, obj=True)
                all_data[-1]['opt_ho'] = opt_ho

        if args.vis:
            # show_optimization(data, opt_state, hand_contact_target.detach().cpu().numpy(), obj_contact_upscale.detach().cpu().numpy(),
            #                   is_video=args.video, vis_method=args.vis_method)
            ## TODO: visualize the forces as heatmaps
            print('start visualization...')
            show_compare_optim(data, opt_state, phy_state, hand_contact_target.detach().cpu().numpy(),
                               obj_contact_upscale.detach().cpu().numpy(), is_video=args.video)

        if idx >= args.partial > 0:   # Speed up for eval
            break

    out_file = 'data/{}{}_{}.pkl'.format('opt' if calculate_cmp else '', 'phy' if calculate_phy else '',args.split)
    print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    pickle.dump(all_data, open(out_file, 'wb'))


def get_args():
    util.hack_filedesciptor()
    args = arguments.run_contactopt_parse_args()

    if args.split == 'aug':     # Settings defaults for Perturbed ContactPose
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.0,
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 2,
                    'w_opt_trans': 0.3,
                    'w_opt_rot': 1.0,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 0,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 8,
                    'rand_re_trans': 0.04,
                    'rand_re_rot': 5,
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'im' or args.split == 'demo':    # Settings defaults for image-based pose estimates
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.5,
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 2,
                    'w_opt_trans': 0.3,
                    'w_opt_rot': 1,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 0,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 320,
                    'pen_it': 0,
                    'rand_re': 8,
                    'rand_re_trans': 0.02,
                    'rand_re_rot': 5,
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'fine':  # Settings defaults for small-scale refinement
        defaults = {'lr': 0.003,
                    'n_iter': 250,
                    'w_cont_hand': 0,
                    'sharpen_thresh': 0.3,
                    'ncomps': 15,
                    'w_cont_asym': 4,
                    'w_opt_trans': 0.03,
                    'w_opt_rot': 1.0,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 5,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 1,
                    'rand_re_trans': 0.00,
                    'rand_re_rot': 0,
                    'w_obj_rot': 0,
                    'vis_method': 5}

    for k in defaults.keys():   # Override arguments that have not been manually set with defaults
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]

    print(args)
    return args

if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    run_contactopt(args)
    print('Elapsed time:', time.time() - start_time)

