import numpy as np
import torch
import torch.nn.functional as F
from copy import copy
from sklearn.cluster import AgglomerativeClustering

## TMP import
import matplotlib.pyplot as plt
import open3d as o3d

from contactopt.diffcontact import calculate_part_contact_capsule, calculate_penetration_cost
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_axis_angle
from kornia.geometry.linalg import inverse_transformation
from prev_sota.contactgen.manopth import rodrigues_layer
from common.utils.utils import linear_normalize
from common.utils.geometry import get_perpend_vecs, normalize_vec_tensor, cp_match
from common.simulation.mujoco_hand_object_simulator import kine_tree_w_tips
from scipy.sparse.csc import csc_matrix
from qpsolvers import solve_qp
from itertools import combinations

def phy_optimize_pose(obj_verts, obj_normals, mano_layer, obj_contact, obj_parts, obj_pressure, coms=None, obj_name=None,
                  n_iter=250, lr=0.005, w_cont_obj=1, glob_iter=200, save_history=False, w_kjpe=1,
                  caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False,
                  contact_norm_method=0, w_pen_cost=300, w_obj_rot=0, pen_it=0, contact_th=0.22):
    """Runs differentiable optimization to align the hand with the target contact map.
    Minimizes the loss between ground truth contact and contact calculated with DiffContact"""
    if coms is None:
        ## Normalized to origin
        coms = torch.zeros((obj_verts.shape[0], 3), device=obj_verts.device)

    batch_size = obj_verts.shape[0]
    global_pose = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=obj_verts.dtype, device=obj_verts.device)

    hand_opt_params = [mano_pose, mano_shape, global_pose, mano_trans]

    optimizer = torch.optim.Adam(hand_opt_params, lr=lr, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss(reduction='none')  # Benchmarked, L1 performs best vs MSE/SmoothL1
    # is_thin = mesh_is_thin(data['mesh_aug'].num_verts_per_mesh())
    # print('is thin', is_thin, data['mesh_aug'].num_verts_per_mesh())

    is_thin = torch.zeros(obj_verts.shape[0], device=obj_verts.device)
    if obj_name is not None:
        for i, n in enumerate(obj_name):
            is_thin[i] = 1 if n in ['toothbrush', 'knife', 'eyeglasses', 'mug', 'scissors'] else 0

    target_part_centre, weights, refined_part_mask, _, _ = key_joints_pos_and_weights(obj_verts, obj_normals, obj_contact, obj_parts, obj_pressure, coms, contact_th)
    ## Only modify trans and global rotation
    hand_verts, hand_joints, hand_faces = mano_layer.mesh_data({'rot_aa': global_pose, 'pose': mano_pose,
                                                                'shape': mano_shape, 'trans': mano_trans}, is_right=True)
    init_part_centre = joint2centre(hand_joints)
    root = hand_joints[:, 0:1]
    ## Transform the problem to mano coordinate system
    Rs, ts = cp_match(init_part_centre - root, target_part_centre - root, weights)

    global_pose = matrix_to_axis_angle(Rs)
    mano_trans = ts

    # trans_p1 = (init_part_centre - root) @ Rs.transpose(-1, -2) + ts.unsqueeze(1)  + root
    # hand_verts, hand_joints, hand_faces = mano_layer.mesh_data({'rot_aa': global_pose, 'pose': mano_pose,
    #                                                             'shape': mano_shape, 'trans': mano_trans}, is_right=True)
    # trans_p2 = joint2centre(hand_joints)
    # print(torch.sum(torch.norm(weights * (trans_p1 - trans_p2), dim=-1), dim=-1))
    # print(torch.det(Rs))

    mano_pose.requires_grad = True
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True

    for it in range(glob_iter):
        loss_info = ''
        loss = 0
        optimizer.zero_grad()

        hand_verts, hand_joints, hand_faces = mano_layer.mesh_data({'rot_aa': global_pose, 'pose': mano_pose,
                                                                    'shape': mano_shape, 'trans': mano_trans}, is_right=True)

        # if contact_norm_method != 0 and not caps_on_hand:
        pred_part_centre = joint2centre(hand_joints)
        kjpe_loss = F.mse_loss((pred_part_centre - target_part_centre) * weights, torch.zeros_like(target_part_centre, device=obj_verts.device), reduction='sum')
        loss += kjpe_loss
        loss_info += f'Keypoint loss: {kjpe_loss:.3f}'
        if it % 100 == 99:
            print(f'Init stage it {it}/{glob_iter}: {loss_info}')

        loss.backward()
        optimizer.step()

    for it in range(n_iter):
        loss_info = ''
        ## The force refinement works in two circumstances:
        ## 1. contact incorrect; 2. slight penetration -> hand deformation.
        loss = 0
        optimizer.zero_grad()

        hand_verts, hand_joints, hand_faces = mano_layer.mesh_data({'rot_aa': global_pose, 'pose': mano_pose,
                                                                    'shape': mano_shape, 'trans': mano_trans}, is_right=True)

        pred_part_centre = joint2centre(hand_joints)
        kjpe_loss = F.mse_loss((pred_part_centre - target_part_centre) * weights, torch.zeros_like(target_part_centre, device=obj_verts.device), reduction='sum')
        loss += kjpe_loss * w_kjpe
        loss_info += f'| Keypoint loss: {kjpe_loss:.3f} '
        with torch.no_grad():   # We need to calculate hand normals if using more complicated methods
            mano_mesh = Meshes(verts=hand_verts, faces=torch.as_tensor(hand_faces, device=obj_verts.device).repeat(batch_size, 1, 1))
            hand_normals = mano_mesh.verts_normals_padded()
        # else:
        #     hand_normals = torch.zeros(hand_verts.shape, device=device)

        pred_contact_obj = calculate_part_contact_capsule(hand_verts, hand_normals, obj_verts, obj_normals, refined_part_mask, mano_layer.part_id_right,
                                                          caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad, caps_on_hand=caps_on_hand,
                                                          contact_norm_method=contact_norm_method, fix_normal=True).squeeze(-1)
        # pred_contact_obj[obj_contact < contact_th] = 0

        # contact_obj_sub = obj_contact - pred_contact_obj.squeeze(-1)

        # contact_obj_sub = obj_contact_target - contact_obj
        # contact_obj_weighted = contact_obj_sub # + torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        # loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(0, 1))
        ## Compare w/ parts.

        # obj_part_contact = obj_parts * obj_contact.unsqueeze(-1) # B x N x 16
        # pred_part_contact = torch.zeros_like(obj_part_contact, device=obj_verts.device)
        # torch.gather(pred_part_contact, dim=3, index=mano_layer.part_id_right[nn_idx.detach().cpu().numpy()])
        # pred_part_contact[mano_layer.part_id_right[nn_idx.detach().cpu().numpy()]] = pred_contact_obj
        loss_contact = F.mse_loss(pred_contact_obj, obj_contact)

        loss += loss_contact * w_cont_obj
        loss_info += f'| Contact loss: {loss_contact:.3f} '

        if w_pen_cost > 0 and it >= pen_it:
            pen_cost = calculate_penetration_cost(hand_verts, hand_normals, obj_verts, obj_normals, is_thin, contact_norm_method)
            loss += pen_cost.mean() * w_pen_cost
            loss_info += f'| Penetration loss: {loss_contact:.3f} '

        ## Physics loss
        # if physics_solver is not None and it >= n_iter * 3 / 4:
        #     contact_obj, contact_hand, sdf_obj, nn_idx = calculate_contact.calculate_contact_signed_dist(
        #         hand_verts, hand_normals, obj_verts, obj_normals, object_meshes, hand_meshes, caps_top=caps_top,
        #         caps_bot=caps_bot, caps_rad=caps_rad)
        #     w_phys_reg = 0.1
        #     obj2hand_contact_target = torch.stack([hand_contact_target[i, nn_idx[i, :, 0]] for i in range(batch_size)], dim=0)
        #     phys_loss, optim_Fs, Fs = physics_solver.qp_physics_stable_loss(
        #         contact_obj.squeeze(-1), obj_verts, obj_normals, coms,
        #         obj_contact_target.squeeze(-1), obj2hand_contact_target.squeeze(-1))
        # phys_loss, Fs = physics_solver.diff_stable_loss(sdf_obj, obj_verts, obj_normals, coms, abs(caps_bot) / caps_rad,
        #                                                 obj_contact_target.squeeze(-1), obj2hand_contact_target.squeeze(-1))

        # loss += phys_loss * w_phys_reg

        # out_dict = {'loss': loss.detach().cpu()}
        # if save_history:
        #     out_dict['hand_verts'] = hand_verts.detach().cpu()#.numpy()
        #     out_dict['hand_joints'] = hand_joints.detach().cpu()#.numpy()
        #     out_dict['contact_obj'] = contact_obj.detach().cpu()#.numpy()
        #     out_dict['contact_hand'] = contact_hand.detach().cpu()#.numpy()
        #     if contact_weights is not None:
        #         out_dict['forces'] = contact_weights.unsqueeze(-1).detach().cpu()
                # out_dict['force_grad'] = Fs.grad.unsqueeze(-1).detach().cpu()
                # out_dict['real_forces'] = real_Fs.unsqueeze(-1).detach().cpu()
        # opt_state.append(out_dict)
        if it % 100 == 99:
            print(loss_info)

        loss.backward()
        optimizer.step()
    vis_params = {'refined_part_map': refined_part_mask, 'fc_part_mask': weights > 0, 'target_pts': target_part_centre}

    return global_pose, mano_pose, mano_shape, mano_trans, vis_params


def phy_optimize_sdf_hand(model, mano_layer, smpl_mano, obj_verts, obj_normals, obj_cmap, obj_partition, obj_pressure=None, coms=None,
                  w_contact=1e-1, w_pene=0.5, w_kp=0.2, w_pose_reg=1e-2, w_shape_reg=1e-2,
                  global_iter=200, pose_iter=1000, contact_th=0.11,
                  global_lr=5e-2, pose_lr=5e-3, eps=-1e-3, ret_history=False):
    model.eval()
    vis_history = []
    for param in model.parameters():
        param.requires_grad = False
    batch_size = obj_verts.shape[0]
    if coms is None:
        ## Normalized to origin
        coms = torch.zeros((obj_verts.shape[0], 3), device=obj_verts.device)
    global_pose = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)

    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=obj_verts.dtype, device=obj_verts.device)

    ## Do initial registration using closest point
    hand_vertices, hand_joints, _ = smpl_mano.mesh_data(
        {'rot_aa': global_pose, 'trans': mano_trans, 'pose': mano_pose, 'shape': mano_shape}, is_right=True)
    init_part_centre = joint2centre(hand_joints)
    root = hand_joints[:, 0:1]

    verts_history = [hand_vertices.detach().clone().cpu().numpy()]
    ## Transform the problem to mano coordinate system
    if obj_pressure is not None:
        target_part_centre, weights, refined_part_mask, part_pres, new_part_pres = key_joints_pos_and_weights(obj_verts, obj_normals, obj_cmap, obj_partition, obj_pressure, coms, contact_th)

        Rs, ts = cp_match(init_part_centre - root, target_part_centre - root, weights)
        global_pose = matrix_to_axis_angle(Rs)
        mano_trans = ts

    # mano_history = [{'rot_aa': global_pose.detach().clone(), 'trans': mano_trans.detach().clone(),
    #                  'pose': mano_pose.detach().clone(), 'shape': mano_shape.detach().clone()}]

    if ret_history:
        hand_vertices, hand_joints, _ = smpl_mano.mesh_data(
            {'rot_aa': global_pose, 'trans': mano_trans, 'pose': mano_pose, 'shape': mano_shape}, is_right=True)
        verts_history.append(hand_vertices.detach().clone().cpu().numpy())

    mano_pose.requires_grad = True
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [global_pose, mano_pose, mano_trans]
    global_optimizer = torch.optim.Adam(hand_opt_params, lr=global_lr)
    if w_kp == 0:
        global_iter = 0

    for it in range(global_iter):
        loss_info = ""
        loss = 0

        _, handJ, frames = mano_layer(torch.cat(
            (torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose), dim=1),
                               th_betas=mano_shape, th_trans=torch.zeros_like(mano_trans, device=mano_trans.device,
                                                                              dtype=mano_trans.dtype))
        hand_vertices, handJ1, _ = smpl_mano.mesh_data({'rot_aa': global_pose, 'trans': mano_trans, 'pose': mano_pose, 'shape': mano_shape}, is_right=True)
        pred_part_centre = joint2centre(handJ1)
        kp_loss = F.mse_loss(weights * (target_part_centre - pred_part_centre), torch.zeros_like(target_part_centre, device=weights.device), reduction='sum')
        loss += kp_loss
        loss_info += "keypoint loss: {:.3f}| ".format(kp_loss)

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        if it % 100 == 99:
            print("global iter {} | ".format(it) + loss_info)
        verts_history.append(hand_vertices.detach().clone().cpu().numpy())

    if ret_history:
        vis_history.append(hand_vertices)

    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [global_pose, mano_pose, mano_shape, mano_trans]
    pose_optimizer = torch.optim.Adam(hand_opt_params, lr=pose_lr)

    for it in range(pose_iter):
        loss_info = ""
        loss = 0
        _, handJ, frames = mano_layer(
            torch.cat(
                (torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose),
                dim=1), th_betas=mano_shape,
            th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1),
                                       global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)  # (B, Q)

        hand_vertices, handJ, _ = smpl_mano.mesh_data({'rot_aa': global_pose, 'trans': mano_trans, 'pose': mano_pose, 'shape': mano_shape}, is_right=True)
        if obj_pressure is not None:
            pred_part_centre = joint2centre(handJ)
            kp_loss = F.mse_loss(weights * (target_part_centre - pred_part_centre), torch.zeros_like(target_part_centre, device=weights.device), reduction='sum')
            loss += kp_loss * w_kp
            loss_info += "keypoint loss: {:.3f}| ".format(kp_loss)

        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        # obj_pressure = obj_pressure / torch.max(obj_pressure, dim=-1, keepdim=True)[0]
        # loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap * obj_pressure).sum(dim=-1).mean(dim=0)
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())
        #
        mask = pred_p_full < eps
        masked_value = pred_p_full[mask]
        if len(masked_value) > 0:
            loss_pene = w_pene * (-masked_value.sum()) / batch_size
            loss += loss_pene
            loss_info += "pene loss: {:.3f} | ".format(loss_pene.item())

        # hand_vertices, handJ, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)

        pose_reg_loss = w_pose_reg * (mano_pose ** 2).sum() / batch_size
        loss += pose_reg_loss
        loss_info += "pose reg loss: {:.3f} | ".format(pose_reg_loss.item())

        shape_reg_loss = w_shape_reg * (mano_shape ** 2).sum() / batch_size
        loss += shape_reg_loss
        loss_info += "shape reg loss: {:.3f}".format(shape_reg_loss.item())

        pose_optimizer.zero_grad()
        loss.backward()
        pose_optimizer.step()
        if it % 100 == 99:
            print("iter {} | ".format(it) + loss_info)

        verts_history.append(hand_vertices.detach().clone().cpu().numpy())

    if ret_history:
        vis_history.append(hand_vertices)

    if ret_history:
        vis_params = {'refined_part_map': refined_part_mask, 'fc_part_mask': weights > 0,
                      'target_pts': target_part_centre}
        return global_pose, mano_pose, mano_shape, mano_trans, verts_history, vis_params
    else:
        return global_pose, mano_pose, mano_shape, mano_trans, part_pres, torch.as_tensor(new_part_pres, device=obj_verts.device).float()


def key_joints_pos_and_weights(obj_verts, obj_normals, obj_contact, obj_parts, obj_pressure, coms, contact_th=0.22):
    contact_mask = obj_contact > contact_th
    part_pres = []
    part_contact_pt = []
    part_masks = []
    for i in range(16):
        part_mask = torch.logical_and(contact_mask, obj_parts == i) # B x N
        part_masks.append(part_mask)
        pres = torch.sum(obj_pressure.unsqueeze(-1) * -obj_normals * part_mask.unsqueeze(-1), dim=1) # B x 3
        contact_pt = torch.sum(obj_pressure.unsqueeze(-1) * obj_verts * part_mask.unsqueeze(-1), dim=1) / (
                torch.sum(obj_pressure * part_mask, dim=1, keepdim=True) + 1e-8) # B x 3
        part_pres.append(pres)
        part_contact_pt.append(contact_pt)
    part_pres = torch.stack(part_pres, dim=1)
    new_part_pres = part_pres.clone()
    part_masks = torch.stack(part_masks, dim=1)
    part_contact_pt = torch.stack(part_contact_pt, dim=1)
    coms = coms.detach().cpu().numpy()
    ## Calculate the score
    # part_pres = part_pres.detach().cpu().numpy()
    # part_contact_pt = part_contact_pt.detach().cpu().numpy()
    obj_radius = torch.max(torch.norm(obj_verts, dim=-1), dim=-1)[0].detach().cpu().numpy()
    weights = torch.zeros(*part_masks.shape[:2], 1, device=obj_verts.device)
    refine_method = 'physics'
    for b in range(obj_verts.shape[0]):
        all_contact_parts = [pid for pid in range(16) if part_masks[b, pid].any()]

        ## Refine the contact maps
        if refine_method == 'geometry':
            pc_masks = [[],] * 16
            determined = []
            undetermined = []
            for p in all_contact_parts:
                pcl = part_cluster(obj_verts[b].detach().cpu().numpy(), obj_normals[b].detach().cpu().numpy(),
                                                part_masks[b, p].detach().cpu().numpy())
                pc_masks[p] = pcl
                if len(pcl) > 1:
                    undetermined.append(p)
                else:
                    determined.append(p)

            ## Try to avoid distributions on both sides of the geometry
            for p in undetermined:
                scores = []
                tmp_vars = []
                if len(determined) > 0:
                    ref_pts = part_contact_pt[b, determined].view(-1, 3)
                    ref_pres = part_pres[b, determined].view(-1, 3)
                    for c in pc_masks[p]:
                        pm = torch.as_tensor(c, device=obj_verts.device)
                        tmp_part_pres = torch.sum(obj_pressure[b].unsqueeze(-1) * -obj_normals[b] * pm.unsqueeze(-1), dim=0)  # B x 3
                        tmp_contact_pt = torch.sum(obj_pressure[b].unsqueeze(-1) * obj_verts[b] * pm.unsqueeze(-1), dim=0) / (
                                torch.sum(obj_pressure[b] * pm, dim=0) + 1e-8)  # B x 3
                        dist = torch.norm(ref_pts - tmp_contact_pt.unsqueeze(0), dim=-1) + 0.1 * (
                                1 - torch.sum(normalize_vec_tensor(ref_pres) * normalize_vec_tensor(tmp_part_pres.unsqueeze(0)), dim=-1))
                        tmp_vars.append((pm, tmp_part_pres, tmp_contact_pt))
                        scores.append(torch.sum(dist).item())

                else:
                    for c in pc_masks[p]:
                        pm = torch.as_tensor(c, device=obj_verts.device)
                        tmp_part_pres = torch.sum(obj_pressure[b].unsqueeze(-1) * -obj_normals[b] * pm.unsqueeze(-1), dim=0)  # B x 3
                        tmp_contact_pt = torch.sum(obj_pressure[b].unsqueeze(-1) * obj_verts[b] * pm.unsqueeze(-1),
                                                   dim=0) / (torch.sum(obj_pressure[b] * pm, dim=0) + 1e-8)  # B x 3
                        tmp_vars.append((pm, tmp_part_pres, tmp_contact_pt))
                        scores.append(phyScore(tmp_part_pres.view(-1, 3).detach().cpu().numpy(), tmp_contact_pt.view(-1, 3).detach().cpu().numpy(), radius=obj_radius[b], obj_com=coms[b]))

                idx = np.argmin(np.array(scores))
                #     # Fs = part_pres[b, all_contact_parts].clone().view(-1, 3).detach().cpu().numpy()
                #     scores.append(phyScore(tmp_part_pres.detach().cpu().numpy(), tmp_contact_pt.detach().cpu().numpy(), coms[b]))
                # idx = np.argmin(np.array(scores))
                pm, tmp_part_pres, tmp_contact_pt = tmp_vars[idx]
                part_masks[b, p] = pm ## Update the part mask
                part_pres[b, p] = tmp_part_pres
                part_contact_pt[b, p] = tmp_contact_pt
                determined.append(p)
        else:
            for p in all_contact_parts:
                pcl = part_cluster(obj_verts[b].detach().cpu().numpy(), obj_normals[b].detach().cpu().numpy(),
                                                part_masks[b, p].detach().cpu().numpy())
                if len(pcl) > 1:
                    scores = []
                    tmp_vars = []
                    origin_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_verts[b, part_masks[b, p]].detach().cpu().numpy() - 0.1))
                    origin_pc.paint_uniform_color([0.8, 0.2, 0])
                    # vis_geoms = [origin_pc]
                    # part_cmap = plt.get_cmap('hsv')
                    for i, c in enumerate(pcl):
                        pm = torch.as_tensor(c, device=obj_verts.device)
                        tmp_part_pres = part_pres[b].clone().view(-1, 3)
                        tmp_contact_pt = part_contact_pt[b].clone().view(-1, 3)
                        tmp_part_pres[p] = torch.sum(obj_pressure[b].unsqueeze(-1) * -obj_normals[b] * pm.unsqueeze(-1), dim=0)  # B x 3
                        tmp_contact_pt[p] = torch.sum(obj_pressure[b].unsqueeze(-1) * obj_verts[b] * pm.unsqueeze(-1), dim=0) / (
                                torch.sum(obj_pressure[b] * pm, dim=0) + 1e-8)  # B x 3
                        # Fs = part_pres[b, all_contact_parts].clone().view(-1, 3).detach().cpu().numpy()
                        scores.append(phyScore(tmp_part_pres.detach().cpu().numpy(), tmp_contact_pt.detach().cpu().numpy(), radius=obj_radius[b], obj_com=coms[b]))
                        tmp_vars.append((pm, tmp_part_pres[p], tmp_contact_pt[p]))
                    #     pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_verts[b, pm].detach().cpu().numpy()))
                    #     pc.paint_uniform_color(part_cmap(i / len(pcl))[:3])
                    #     vis_geoms.append(pc)
                    # o3d.visualization.draw_geometries(vis_geoms)
                    idx = np.argmin(np.array(scores))
                    pm, tmp_part_pres, tmp_contact_pt = tmp_vars[idx]
                    part_masks[b, p] = pm ## Update the part mask
                    new_part_pres[b, p] = tmp_part_pres
                    part_contact_pt[b, p] = tmp_contact_pt
            # else:
            #     pm = torch.as_tensor(pcl[0], device=obj_verts.device)
            #     part_masks[b, p] = pm
            #     part_pres[b, p] = torch.sum(obj_pressure[b].unsqueeze(-1) * -obj_normals[b] * pm.unsqueeze(-1), dim=0)  # B x 3
            #     part_contact_pt[b, p] = torch.sum(obj_pressure[b].unsqueeze(-1) * obj_verts[b] * pm.unsqueeze(-1), dim=0) / (
            #         torch.sum(obj_pressure[b] * pm, dim=0) + 1e-8)  # B x 3

            ## First determine if ambiguous part maps appear.
    target_part_centre = part_contact_pt - new_part_pres / (torch.norm(new_part_pres, dim=-1, keepdim=True) + 1e-8) * 0.005
    part_contact_pt = part_contact_pt.detach().cpu().numpy()
    new_part_pres = new_part_pres.detach().cpu().numpy()
    new_part_pres[:] = 11.374 ## Test average value

    num_kps = 3
    for b in range(obj_verts.shape[0]):
        all_contact_parts = [pid for pid in range(16) if part_masks[b, pid].any()]
        ## Reduce the key points one-by-one to find the optimal closure set of 3 or 4 points.
        if len(all_contact_parts) < num_kps + 1:
            final_contacts = all_contact_parts
        else:
            kp_comb = []
            scores = []
            # for i1, p1 in enumerate(all_contact_parts[:-2]):
            #     for i2, p2 in enumerate(all_contact_parts[i1+1:-1]):
            #         for i3, p3 in enumerate(all_contact_parts[i2+i1+2:]):
            for tmp_part_list in combinations(all_contact_parts, num_kps):
                Fs = new_part_pres[b, tmp_part_list].reshape(-1, 3)
                pts = part_contact_pt[b, tmp_part_list].reshape(-1, 3)
                s = phyScore(Fs, pts, obj_radius[b], coms[b])
                kp_comb.append(tmp_part_list)
                scores.append(s)
            scores = np.array(scores)
            final_contacts = kp_comb[np.argmin(scores)]

            # print(f"The {b}th score is {s}")
        # selected_parts = []
        # opt_score = 100
        # while (len(selected_parts) < len(all_contact_parts) and opt_score > 1):
        #     selected_parts.append(0)
        #     scores = []
        #     parts = []
        #     for pid in all_contact_parts:
        #         if len(selected_parts) == 1 or pid not in selected_parts[:-1]:
        #             selected_parts[-1] = pid
        #             Fs = part_pres[b, selected_parts].reshape(-1, 3)
        #             pts = part_contact_pt[b, selected_parts].reshape(-1, 3)
        #             s = phyScore(Fs, pts, coms[b])
        #             scores.append(s)
        #             parts.append(pid)
        #     selected_parts[-1] = parts[np.argmin(np.array(scores))]
        #     opt_score = np.min(np.array(scores))
        weights[b, final_contacts] = 1
    weights[b, 0] *= 0.4 # lower the weight for palm due to the large and complex shape

    return target_part_centre, weights, part_masks, part_pres, new_part_pres


def phyScore(Fs: np.ndarray, pts: np.ndarray, radius, obj_com: np.ndarray=np.zeros(3)):
    """
    Stable score function.
    Fs, n x 6. Fu: 6 F[..., :3]: pressure; F[..., 3:]: contact pt
    com: 3
    return: score: float
    """
    n = Fs.shape[0]
    NF = Fs # n x 3
    pressure = np.linalg.norm(Fs, axis=1) # n x 1
    B, T = get_perpend_vecs(Fs / (pressure.reshape(n, 1) + 1e-8)) # n x 3
    BF = (B * pressure.reshape(n, 1)) # n x 3
    TF = (T * pressure.reshape(n, 1)) # n x 3
    mg_res = np.array([[0], [0], [-9.8], [0], [0], [0]])
    BT = np.concatenate((BF.T, TF.T), axis=1) # 3 x 2n

    ## Calculate the torque stability
    ls = pts - obj_com.reshape(1, 3)

    ## Moment of inertia approximation as a ball
    # radius = np.max(np.linalg.norm(ls, axis=-1))
    J = 0.4 * radius ** 2
    NFp = np.cross(ls, NF, axis=-1) / J
    BFp = np.cross(ls, BF, axis=-1) / J
    TFp = np.cross(ls, TF, axis=-1) / J
    BTp = np.concatenate((BFp.T, TFp.T), axis=1)
    ## min N + BT @ x s.t. |x| < 1
    N = np.sum(np.concatenate((NF.T, NFp.T), axis=0), axis=1, keepdims=True) # 6 x 1
    BT = np.concatenate((BT, BTp), axis=0) # 6 x 2n

    qp_params = {'P': csc_matrix(BT.T @ BT), 'q': BT.T @ (N + mg_res), 'lb': -np.ones(2*n), 'ub': np.ones(2*n), 'solver': 'clarabel'}
    sln = solve_qp(**qp_params)
    if sln is None:
        min_a = 9.81
    else:
        min_a = np.linalg.norm(N + BT @ sln.reshape(2*n, 1) + mg_res)
    return min_a


def joint2centre(handJ: torch.Tensor):
    """
    handJ: B x 21 x 3
    return: B x 16 x 3
    """
    centres = torch.zeros((handJ.shape[0], 16, 3), device=handJ.device)
    palm_js = [0]
    for ktree in kine_tree_w_tips.values():
        for i in range(3):
            centres[:, ktree[i+1]] = (handJ[:, ktree[i+1]] + handJ[:, ktree[i+2]]) / 2
        palm_js.append(ktree[1])
    centres[:, 0] = torch.mean(handJ[:, palm_js], dim=1)
    return centres


def part_cluster(obj_verts: np.array, obj_normals: np.array, part_mask: np.array):
    """
    obj_verts: N x 3
    obj_normals: N x 3
    part_mask: N
    return: list of torch.Tensor
    """
    n_pts = np.sum(part_mask)
    part_verts = obj_verts[part_mask]
    part_normals = obj_normals[part_mask]
    dist = np.linalg.norm(part_verts.reshape(n_pts, 1, 3) - part_verts.reshape(1, n_pts, 3), axis=-1) # n x n
    dot_prod = np.sum(part_normals.reshape(n_pts, 1, 3) * part_normals.reshape(1, n_pts, 3), axis=-1) # n x n
    dist_mat = dist + (1 - dot_prod) * 0.1
    if n_pts < 10 or np.max(dist_mat) < 0.08:
        return [part_mask]
    hier_cluster = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=0.08, linkage='average') # 5cm w/ a little difference
    hier_cluster.fit(dist_mat)
    masks = []
    for i in range(hier_cluster.n_clusters_):
        pm = part_mask.copy()
        pm[pm] = hier_cluster.labels_ == i
        masks.append(pm)

    return masks