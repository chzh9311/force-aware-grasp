# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contactopt.loader import *
import contactopt.util as util
from contactopt.hand_object import HandObject
import time
from open3d import io as o3dio
from open3d import geometry as o3dg
from copy import copy
import open3d as o3d
from open3d import utility as o3du
from open3d import visualization as o3dv
from common.utils.utils import linear_normalize
from common.utils.vis import AppWindow
import open3d.visualization.gui as gui


def show_compare_optim(data, opt_state, phy_state, hand_contact_target=None, obj_contact_target=None, is_video=False, delay=0.001):
    """Displays video/still frame of optimization process
    """

    gt_ho = HandObject()
    in_ho = HandObject()
    opt_ho = HandObject()
    phy_ho = HandObject()
    gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'],
                          data['obj_contact_gt'], data['mesh_gt'])
    in_ho.load_from_batch(data['hand_beta_aug'], data['hand_pose_aug'], data['hand_mTc_gt'], data['hand_contact_gt'],
                          data['obj_contact_gt'], data['mesh_aug'])
    opt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'],
                           data['obj_contact_gt'], data['mesh_aug'], obj_rot=opt_state[-1]['obj_rot'])
    phy_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'],
                           data['obj_contact_gt'], data['mesh_aug'], obj_rot=phy_state[-1]['obj_rot'])

    hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes()
    hand_mesh_aug, obj_mesh_aug = in_ho.get_o3d_meshes()
    hand_mesh_opt, obj_mesh_opt = opt_ho.get_o3d_meshes()
    hand_mesh_phy, obj_mesh_phy = phy_ho.get_o3d_meshes()

    obj_mesh_force = copy(obj_mesh_phy)
    obj_mesh_contactxforce = copy(obj_mesh_phy)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[-0.5, 0, 0])
    geom_list = [hand_mesh_gt, obj_mesh_gt, hand_mesh_aug, obj_mesh_aug, hand_mesh_opt, obj_mesh_opt, hand_mesh_phy, obj_mesh_phy,
                 obj_mesh_force, obj_mesh_contactxforce, coord_frame]

    ## Do visualization
    util.mesh_set_color(hand_contact_target, hand_mesh_opt)
    util.mesh_set_color(hand_contact_target, hand_mesh_phy)
    if obj_contact_target.shape[1] == util.SAMPLE_VERTS_NUM:
        obj_contact_target = upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], obj_contact_target)
    opt_pred_obj_contact = upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], opt_state[-1]['contact_obj'])
    phy_pred_obj_contact = upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], phy_state[-1]['contact_obj'])
    opt_pred_hand_contact = opt_state[-1]['contact_hand']
    phy_pred_hand_contact = phy_state[-1]['contact_hand']
    force_pred = upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], phy_state[-1]['forces'])

    ## Normalize force to paint color
    fcolor = linear_normalize(force_pred, 0.1, 1)
    hand_mesh_opt.vertices = o3du.Vector3dVector(opt_state[-1]['hand_verts'].squeeze())
    hand_mesh_opt.compute_vertex_normals()
    hand_mesh_phy.vertices = o3du.Vector3dVector(phy_state[-1]['hand_verts'].squeeze())
    hand_mesh_phy.compute_vertex_normals()

    vis_list = [{'label': 'GT', 'position': [0, 0.3, 0], 'hand_mesh': hand_mesh_gt,
                 'obj_mesh': obj_mesh_gt, 'obj_color': None, 'hand_color': None},
                {'label': 'Input', 'position': [0.3, 0.3, 0], 'hand_mesh': hand_mesh_aug,
                 'obj_mesh': obj_mesh_aug, 'obj_color': None, 'hand_color': None},
                {'label': 'ContactOpt', 'position': [0, 0, 0], 'hand_mesh': hand_mesh_opt,
                 'obj_mesh': obj_mesh_opt, 'obj_color': obj_contact_target, 'hand_color': hand_contact_target},
                {'label': 'ContactOpt Contacts', 'position': [0.3, 0, 0], 'hand_mesh': copy(hand_mesh_opt),
                 'obj_mesh': copy(obj_mesh_opt), 'obj_color': opt_pred_obj_contact, 'hand_color': opt_pred_hand_contact},
                {'label': 'PhysOpt', 'position': [0, -0.3, 0], 'hand_mesh': hand_mesh_phy,
                 'obj_mesh': obj_mesh_phy, 'obj_color': obj_contact_target, 'hand_color': hand_contact_target},
                {'label': 'PhysOpt Contacts', 'position': [0.3, -0.3, 0], 'hand_mesh': copy(hand_mesh_phy),
                 'obj_mesh': copy(obj_mesh_phy), 'obj_color': phy_pred_obj_contact, 'hand_color': phy_pred_hand_contact},
                {'label': 'Force', 'position': [0.6, -0.3, 0], 'hand_mesh': None,
                 'obj_mesh': obj_mesh_contactxforce, 'obj_color': fcolor, 'hand_color': None},
                ]

    def vis_geometry_list(vis_list):
        geo_list = []
        vis_dict = {}
        for vis_obj in vis_list:
            lbl_pc = util.text_3d(vis_obj['label'], pos=vis_obj['position'], font_size=20, density=2)
            geo_list.append(lbl_pc)
            vis_dict[vis_obj['label'] + '_label'] = lbl_pc
            if vis_obj['hand_mesh'] is not None:
                vis_obj['hand_mesh'].translate(vis_obj['position'])
                geo_list.append(vis_obj['hand_mesh'])
                vis_dict[vis_obj['label'] + ' hand mesh'] = vis_obj['hand_mesh']
            if vis_obj['obj_mesh'] is not None:
                vis_obj['obj_mesh'].translate(vis_obj['position'])
                geo_list.append(vis_obj['obj_mesh'])
                vis_dict[vis_obj['label'] + ' obj mesh'] = vis_obj['obj_mesh']
            if vis_obj['obj_color'] is not None:
                util.mesh_set_color(vis_obj['obj_color'], vis_obj['obj_mesh'])
            if vis_obj['hand_color'] is not None:
                util.mesh_set_color(vis_obj['hand_color'], vis_obj['hand_mesh'])
        # o3dv.draw_geometries(geo_list)
        return vis_dict

    templates = vis_geometry_list(vis_list)
    vis_seq = []
    for i in range(len(opt_state)):
        vis_seq.append([
            {'name': 'ContactOpt hand mesh',
             'vertices': o3d.utility.Vector3dVector(opt_state[i]['hand_verts'].squeeze().detach().cpu().numpy() + np.array([[0, 0, 0]])),
             'color': hand_contact_target},
            {'name': 'ContactOpt Contacts hand mesh',
             'vertices': o3d.utility.Vector3dVector(opt_state[i]['hand_verts'].squeeze().detach().cpu().numpy() + np.array([[0.3, 0, 0]])),
             'color': opt_state[i]['contact_hand']},
            {'name': 'ContactOpt Contacts obj mesh',
             'vertices': None, 'color': upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], opt_state[i]['contact_obj'])},
            # {'name': 'ContactOpt obj mesh', 'vertices': None,
            #  'color': obj_contact_target},
            {'name': 'PhysOpt hand mesh',
             'vertices': o3d.utility.Vector3dVector(phy_state[i]['hand_verts'].squeeze().detach().cpu().numpy() + np.array([[0, -0.3, 0]])),
             'color': hand_contact_target},
            # {'name': 'PhysOpt obj mesh', 'vertices': None,
            #  'color': obj_contact_target},
            {'name': 'PhysOpt Contacts hand mesh',
             'vertices': o3d.utility.Vector3dVector(phy_state[i]['hand_verts'].squeeze().detach().cpu().numpy() + np.array([[0.3, -0.3, 0]])),
             'color': phy_state[i]['contact_hand']},
            {'name': 'PhysOpt Contacts obj mesh', 'vertices': None,
             'color': upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], phy_state[i]['contact_obj'])},
            {'name': 'Force obj mesh', 'vertices': None,
             'color': linear_normalize(upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], phy_state[i]['forces']), 0.1, 1)}
        ])

    gui.Application.instance.initialize()
    window = AppWindow(2500, 1600, templates, vis_seq)
    gui.Application.instance.run()

    if not is_video:
        pass
        # vis_geometry_list(vis_list)
        # o3dv.draw_geometries(geom_list)
    # else:
    #     vis = o3dv.VisualizerWithKeyCallback()
    #     vis.create_window()
    #     for g in geom_list:
    #         vis.add_geometry(g)
    #
    #     for i in range(len(opt_state) * 2):
    #         out_dict1 = opt_state[i % len(opt_state)]
    #         if out_dict1['obj_rot'][0, 0, 0] < 1:
    #             obj_verts = util.apply_rot(out_dict1['obj_rot'], data['mesh_aug'].verts_padded(), around_centroid=True).squeeze()
    #             obj_mesh_opt.vertices = o3du.Vector3dVector(obj_verts)
    #
    #         hand_mesh_opt.vertices = o3du.Vector3dVector(out_dict1['hand_verts'].squeeze())
    #
    #         out_dict2 = phy_state[i % len(phy_state)]
    #         if out_dict2['obj_rot'][0, 0, 0] < 1:
    #             obj_verts = util.apply_rot(out_dict2['obj_rot'], data['mesh_aug'].verts_padded(), around_centroid=True).squeeze()
    #             obj_mesh_phy.vertices = o3du.Vector3dVector(obj_verts)
    #
    #         hand_mesh_phy.vertices = o3du.Vector3dVector(out_dict2['hand_verts'].squeeze())
    #
    #         vis.update_geometry(hand_mesh_opt)
    #         vis.update_geometry(obj_mesh_opt)
    #         vis.update_geometry(hand_mesh_phy)
    #         vis.update_geometry(obj_mesh_phy)
    #         vis.update_geometry(obj_mesh_force)
    #         vis.update_geometry(obj_mesh_contactxforce)
    #         vis.poll_events()
    #         vis.update_renderer()
    #
    #         if i % len(opt_state) == 0:
    #             time.sleep(2)
    #         # time.sleep(delay)
    #
    #     vis.destroy_window()

def show_optimization(data, opt_state, hand_contact_target=None, obj_contact_target=None, is_video=False, label=None, vis_method=1, delay=0.001):
    """Displays video/still frame of optimization process
    Contact visualization options:
    0 GT contact on opt
    1 Predicted contact on opt
    2 Live contact on opt hand
    3 Live contact on both
    4 No contact on any
    5 No hand contact, predicted obj contact
    """

    gt_ho = HandObject()
    opt_ho = HandObject()
    gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'])
    opt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], obj_rot=opt_state[-1]['obj_rot'])

    hand_mesh_gt, obj_mesh_gt = gt_ho.get_o3d_meshes()
    hand_mesh_opt, obj_mesh_opt = opt_ho.get_o3d_meshes()
    geom_list = [hand_mesh_gt, obj_mesh_gt, obj_mesh_opt, hand_mesh_opt]

    if vis_method == 1 or vis_method == 5:
        util.mesh_set_color(hand_contact_target, hand_mesh_opt)

        if obj_contact_target.shape[1] == util.SAMPLE_VERTS_NUM:
            obj_contact_target = upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], obj_contact_target)

        util.mesh_set_color(obj_contact_target, obj_mesh_opt)
    if vis_method == 2 or vis_method == 3:
        util.mesh_set_color(opt_state[-1]['contact_hand'].squeeze(), hand_mesh_opt)
        if opt_state[-1]['contact_obj'].shape[1] == util.SAMPLE_VERTS_NUM:
            c = upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], opt_state[-1]['contact_obj'])
            util.mesh_set_color(c, obj_mesh_opt)
        else:
            util.mesh_set_color(opt_state[-1]['contact_obj'].squeeze(), obj_mesh_opt)
    if vis_method == 4 or vis_method == 5:
        hand_mesh_gt.paint_uniform_color(np.asarray([150.0, 250.0, 150.0]) / 255)   # Green
        hand_mesh_opt.paint_uniform_color(np.asarray([250.0, 150.0, 150.0]) / 255)  # Red

    if vis_method == 4:
        obj_mesh_gt.paint_uniform_color(np.asarray([100.0, 100.0, 100.0]) / 255)   # Gray
        obj_mesh_opt.paint_uniform_color(np.asarray([100.0, 100.0, 100.0]) / 255)  # Gray

    if label is not None:
        lbl_verts = util.text_3d(label, pos=[0, 0.1, 0], font_size=20, density=2)
        geom_list.append(lbl_verts)

    hand_mesh_opt.vertices = o3du.Vector3dVector(opt_state[-1]['hand_verts'].squeeze())
    hand_mesh_opt.compute_vertex_normals()

    hand_mesh_gt.translate((0, 0.2, 0))
    obj_mesh_gt.translate((0, 0.2, 0))

    if not is_video:
        o3dv.draw_geometries(geom_list)
    else:
        vis = o3dv.VisualizerWithKeyCallback()
        vis.create_window()
        for g in geom_list:
            vis.add_geometry(g)

        for i in range(len(opt_state) * 2):
            out_dict = opt_state[i % len(opt_state)]

            if out_dict['obj_rot'][0, 0, 0] < 1:
                obj_verts = util.apply_rot(out_dict['obj_rot'], data['mesh_aug'].verts_padded(), around_centroid=True).squeeze()
                obj_mesh_opt.vertices = o3du.Vector3dVector(obj_verts)

            hand_mesh_opt.vertices = o3du.Vector3dVector(out_dict['hand_verts'].squeeze())

            if vis_method == 2 or vis_method == 3:
                util.mesh_set_color(out_dict['contact_hand'].squeeze(), hand_mesh_opt)
            if vis_method == 3:
                if out_dict['contact_obj'].shape[1] == util.SAMPLE_VERTS_NUM:
                    c = util.upscale_contact(data['mesh_aug'], data['obj_sampled_idx'], out_dict['contact_obj'])
                    util.mesh_set_color(c, obj_mesh_opt)
                else:
                    util.mesh_set_color(out_dict['contact_obj'].squeeze(), obj_mesh_opt)

            vis.update_geometry(hand_mesh_opt)
            vis.update_geometry(obj_mesh_opt)
            vis.poll_events()
            vis.update_renderer()

            if i % len(opt_state) == 0:
                time.sleep(2)
            # time.sleep(delay)

        vis.destroy_window()


