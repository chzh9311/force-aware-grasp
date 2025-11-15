import os
import os.path as osp
import numpy as np
import torch
import lightning as L
import trimesh
from colorsys import hsv_to_rgb
import pickle
from alive_progress import alive_bar
from isort.parse import normalize_line
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt

from trimesh.sample import sample_surface
from collections import defaultdict
from copy import deepcopy
import open3d as o3d
from copy import deepcopy
from torch.utils.data import DataLoader
from common.utils.geometry import read_obj
from common.utils.vis import o3dmesh_from_trimesh
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from common.utils.manolayer import ManoLayer
from common.utils.utils import trimesh2Mesh, linear_normalize
# from common.simulation.genesis_hand_object_simulator import GenesisHandObjectSimulator
from common.simulation.mujoco_hand_object_simulator import run_mujoco_simulate
from common.utils.geometry import compute_signed_distance_and_closest_goemetry
from common.model.hand_object import HandObject
from contactopt.diffcontact import calculate_contact_capsule
from contactopt.util import upscale_contact, mesh_set_color, batched_index_select
from contactopt.evaluation.mano_train.simulation.simulate import run_simulation, run_sim_parallel_interface

from multiprocessing.pool import Pool

import pybullet
import pybullet_utils.bullet_client as bc


def aggregate_part_contact_normals(contact_map: np.ndarray, sdf_obj: np.ndarray, part_map: np.ndarray, obj_normals: np.ndarray) -> tuple:
    """
    :param contact_map: B x N x 1
    :param part_map: B x N x 1
    :param obj_normals: B x N x 3
    :return: tuple of (B x N x 16 for average contact values; B x N x 16 for minimum contact distance; B x N x 16 x 3 which indicates normals)
    """
    part_normals, part_contacts, part_dists = [], [], []
    for i in range(16):
        mask = (part_map == i).astype(np.uint8)
        contact_i = np.sum(contact_map * mask, axis=1)
        part_contacts.append(contact_i)

        dist_i = np.min(sdf_obj * mask + 100 * (1 - mask), axis=1)
        part_dists.append(dist_i)

        normals_i = np.sum(mask * contact_map * obj_normals, axis=1)
        normals_i = normals_i / (np.linalg.norm(normals_i, axis=1, keepdims=True) + 1.0e-6)
        part_normals.append(normals_i)

    return np.stack(part_contacts, axis=1), np.stack(part_dists, axis=1), np.stack(part_normals, axis=1)


def get_force_from_simulation(param):
    label_cfg, dn, fn, hand_model, hand_joints, obj_name, obj_model, obj_hulls, fidx, part_id = (
        param['label_cfg'], param['dataset_name'], param['frame_name'], param['hand_model'], param['hand_joints'],
        param['obj_name'], param['obj_model'], param['obj_hulls'], param['idx'], param['part_id'])
    model_path = osp.join('tmp', dn, 'Ours', f'{fidx:04d}')
    ## TODO: modify the hand pose based on the contact distance.
    # hand_verts = np.asarray(hand_model.vertices, dtype=np.float32)
    # obj_scene = o3d.t.geometry.RaycastingScene()
    # obj_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3dmesh_from_trimesh(obj_model)))
    # hand_vert_sdfs, nn, nn_normals = compute_signed_distance_and_closest_goemetry(obj_scene, hand_verts)
    # part_normals = np.zeros((16, 3))
    # part_dists = np.zeros(16)
    # for p in range(16):
    #     mask = part_id == p
    #     part_sdf, nn_part_normals = hand_vert_sdfs[mask], nn_normals[mask]
    #     part_nn_idx = np.argmin(part_sdf)
    #     part_normals[p] = nn_part_normals[part_nn_idx]
    #     part_dists[p] = part_sdf[part_nn_idx]

    # self.visualize_ho_features(hand_models[i], obj_models[i], contact_hand[i], cmap_obj[i], hand_part, pmap_obj[i])
    # params = {}
    # self.simulator.model_path = osp.join('tmp', self.dataset_name, f'{idx:04d}')
    contact_info, obj_qposes = run_mujoco_simulate(
        label_cfg=label_cfg,
        model_path=model_path,
        hand_mesh=hand_model,
        hand_joints=hand_joints,
        obj_model=obj_model,
        obj_hulls=obj_hulls,
        part_ids=part_id,
        # part_normals=part_normals,
        # part_dists=part_dists
    )

    disps = [np.linalg.norm(qp[:3]) for qp in obj_qposes]
    stable = 1
    labelled = False
    label_idx = len(disps)
    for idx, disp in enumerate(disps):
        if disp > 0.01 and not labelled: # ensure the force are labelled within 1 cm of displacement.
            label_idx = idx + 1
            labelled = True
        if disp > 0.05: # 5cm as an empirical threshold.
            stable = 0
            break
    feasible_disps = obj_qposes[:label_idx]
    if label_idx > 10:
        ## Use the frame with the least acceleration.
        vec = feasible_disps[1:] - feasible_disps[:-1]
        acc = np.linalg.norm(np.stack(vec[1:] - vec[:-1], axis=0), axis=1)
        ref_frame = np.argmin(acc)+1
    else:
        ref_frame = label_idx
    label_contact = contact_info[ref_frame]
    frame_info = {'contacts': [], 'obj_disp': obj_qposes[-1], 'label_frame':ref_frame,
                  'label_obj_disp': obj_qposes[ref_frame], 'frame_name': fn, 'stable': stable}
    if len(label_contact):
        for c in label_contact:
            frame, part_id, force_vec, contact_pts = c['frame'], c['hand_part_id'], c['force'], c['pos']
            # force_vec = (frame.reshape(3, 3).T @ force_vec.reshape(3, 1)).reshape(3)
            frame_info['contacts'].append({'part_id': part_id, 'frame': frame, 'force': force_vec, 'contact_pt': contact_pts})
    # print(disps[-1])

    return frame_info

# hm, om = o3dmesh_from_trimesh(hand_models[i]) , o3dmesh_from_trimesh(obj_models[i])
# o3d.visualization.draw_geometries([hm, om])

class ForceLabellingModel:
    def __init__(self, config, dataloader, split, parallelize=True, vis=False):
        super().__init__()
        self.mano_path = config.data.mano_path
        self.manolayer = ManoLayer(mano_path=config.data.mano_path, flat_hand_mean=True)
        self.parallelize = parallelize
        self.split = split
        if parallelize:
            self.pool = Pool(min(config[split].batch_size, 16))
        self.vis = vis
        # self.simulator = GenesisHandObjectSimulator('gpu')

        self.dataset_name = config.data.dataset
        self.dataset = dataloader.dataset
        if self.dataset_name == 'ho3d':
            self.obj_model_path = config.data.obj_model_path

        # self.simulator = MujocoHandObjectSimulator(osp.join('tmp', self.dataset_name))
        self.dataloader = dataloader
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.label_cfg = config.label

    def run_labelling(self):
        simu_result = []
        label_dir = osp.join('force_labels', 'verify', f'{self.dataset_name}_{self.split}')
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        pointer = 0
        with alive_bar(len(self.dataloader)) as bar:
            for idx, batch in enumerate(self.dataloader):
                # if idx > 300:
                #     bar()
                #     break
                for k, v in batch.items():
                    if type(v) is torch.Tensor:
                        batch[k] = v.to(self.device)
                ho_batch = HandObject(self.device, self.manolayer.mano_f['right'], self.manolayer.part_id_right, normalize=True)
                ## Prepare data
                obj_templates, obj_hull_templates = [], []
                for obj_name in batch['objName']:
                    obj_templates.append(trimesh.Trimesh(self.dataset.obj_info[obj_name]['verts'], self.dataset.obj_info[obj_name]['faces']))
                    obj_hull_templates.append(self.dataset.obj_hulls[obj_name])

                ## Load to a hand-object object.
                flip_side = np.array([batch['handSide'][i] == 'left' for i in range(len(batch['handSide']))])
                inv_obj_rot = self.dataset_name == 'grab'
                ho_batch.load_from_batch(batch, obj_templates, obj_hull_templates,
                                                inv_obj_rot=inv_obj_rot)

                # hand_models, hand_joints, obj_models, obj_hulls = self.get_batch_hm_oh(batch, self.dataset_name)
                # hand_models, hand_joints, obj_models = self.get_batch_ho_models(batch, self.dataset_name)
                if self.vis:
                    obj_samples, obj_normals = [], []
                    hand_verts, hand_normals = [], []
                    vids = []
                    for i in range(len(ho_batch.hand_models)):
                        hm, om = ho_batch.hand_models[i], ho_batch.obj_models[i]
                        hand_verts.append(hm.vertices)
                        hand_normals.append(hm.vertex_normals)
                        # samples, fids = sample_surface(om, count=2048)
                        vid = np.random.randint(0, len(om.vertices), 2048)
                        vids.append(torch.from_numpy(vid))
                        obj_samples.append(om.vertices[vid])
                        obj_normals.append(om.vertex_normals[vid])
                    hand_verts, hand_normals, obj_samples, obj_normals = torch.from_numpy(np.stack(hand_verts, axis=0)).float(),\
                        torch.from_numpy(np.stack(hand_normals, axis=0)).float(), torch.from_numpy(np.stack(obj_samples, axis=0)).float(), torch.from_numpy(np.stack(obj_normals, axis=0)).float(),
                    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

                    ## For contact map & part map visualization
                    contact_obj, contact_hand, sdf_obj, sdf_hand, nn_idx = calculate_contact_capsule(hand_verts, hand_normals, obj_samples, obj_normals,
                        caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False, contact_norm_method=0)
                    obj_Meshes = trimesh2Mesh(ho_batch.obj_models, 'cpu')
                    cmap_obj = upscale_contact(obj_Meshes, torch.stack(vids, dim=0).long(), contact_obj)
                    hand_part = np.argmax(self.manolayer.W_right, axis=1)
                    obj_part = np.stack([hand_part[nn_idx[i]] for i in range(len(ho_batch.hand_models))], axis=0)
                    pmap_obj = upscale_contact(obj_Meshes, torch.stack(vids, dim=0).long(), torch.from_numpy(obj_part).int())
                    for i in range(len(ho_batch.hand_models)):
                        self.visualize_ho_features(ho_batch.hand_models[i], ho_batch.obj_models[i], contact_hand[i], cmap_obj[i], hand_part, pmap_obj[i])

                param_list = [{'label_cfg': deepcopy(self.label_cfg), 'dataset_name':self.dataset_name,
                               'frame_name':batch['frameName'][i], 'hand_model': ho_batch.hand_models[i],
                               'obj_name': batch['objName'][i], 'hand_joints':ho_batch.hand_joints[i],
                               'obj_model': ho_batch.obj_models[i], 'obj_hulls': ho_batch.obj_hulls[i],
                               'idx': i, 'part_id': self.manolayer.part_id_right} for i in range(len(ho_batch.hand_models))]
                if self.parallelize:
                    frame_info = self.pool.map(get_force_from_simulation, param_list)
                    simu_result.extend(frame_info)
                else:
                    for param in param_list:
                        frame_info = get_force_from_simulation(param)
                        ## Take only the last frame as label.
                        simu_result.append(frame_info)
                bar()
                if idx % 20 == 0:
                    print(f'Processing {idx}/{len(self.dataloader)} ...')
                    print(f'Stable rate: {sum([np.linalg.norm(r["obj_disp"][:3]) < 0.05 for r in simu_result]) / len(simu_result) * 100}%')
                log_gap = 300
                if idx % log_gap == log_gap - 1:
                    with open(osp.join(label_dir, f'{idx-log_gap+1:04d}-{idx:04d}.pkl'), 'wb') as f:
                        pickle.dump(simu_result, f)
                    pointer = idx + 1
                    simu_result = []
            with open(osp.join(label_dir, f'{pointer:04d}-{idx:04d}.pkl'), 'wb') as f:
                pickle.dump(simu_result, f)
        # self.pool.join()

    def run_pybullet_labelling(self):
        vhacd_exe = "/home/zxc417/Projects/Toolboxes/v-hacd/app/build/TestVHACD"
        with alive_bar(len(self.dataloader)) as bar:
            for idx, batch in enumerate(self.dataloader):
                for k, v in batch.items():
                    if type(v) is torch.Tensor:
                        batch[k] = v.to(self.device)
                hand_models, hand_joints, obj_models = self.get_batch_ho_models(batch, self.dataset_name)
                client = bc.BulletClient(connection_mode=pybullet.DIRECT)
                for i in range(len(batch['frameName'])):
                    fid = batch['frameName'][i].replace('/', '_')
                    disp = run_simulation(hand_models[i].vertices, hand_models[i].faces, obj_models[i].vertices,
                                   obj_models[i].faces, indicator=fid, client=client, simulation_step=0.0001, num_iterations=10000,
                                   hand_restitution=0.99, object_restitution=0.99, hand_friction=1, object_friction=1,
                                   vhacd_exe=vhacd_exe, save_video=False, save_video_path='tmp/pybullet/', use_gui=False)
                    print(disp)
                break

        # vis_contact_force()
        # pos_disp = np.linalg.norm(disps[:, :3], axis=1) * 1000
        # rot_disp = np.arccos(disps[:, 3]) * 2
        # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        # axes[0].hist(pos_disp, bins=100, density=True)
        # axes[0].set_xlabel('Positional displacement (mm)')
        # axes[1].hist(rot_disp, bins=100, density=True)
        # axes[1].set_xlabel('Rotation displacement (radian)')
        # fig.suptitle('Simulation displacements')
        # print(f"""Statistics:
        # avg position displacement: {np.mean(pos_disp):.2f} (mm); avg rotation displacement: {np.mean(rot_disp):.2f} (radian).
        # std position displacement: {np.std(pos_disp):.2f} (mm); std rotation displacement: {np.std(rot_disp):.2f} (radian).
        # pos_disp < 5cm & rot < 0.17: {np.sum(np.logical_and(pos_disp < 50, rot_disp < 0.17)) * 100 / len(pos_disp):.2f} %.
        # """)
        plt.show()

        # with alive_bar(len(self.dataloader)) as bar:
        #     for idx, batch in enumerate(self.dataloader):
        #         bar()

    #
    # def get_batch_ho_models(self, batch, dataset='ho3d'):
    #     """
    #     get the hand and object models in trimesh.Trimesh format.
    #     """
    #     # hmodel = load_model(osp.join(self.mano_path, 'models', 'MANO_RIGHT.pkl'), ncomps=6, flat_hand_mean=True)
    #     hand_models, obj_models = [], []
    #     if dataset == 'ho3d':
    #         self.manolayer.to(self.device)
    #         params = {'rot_aa': batch['handPose'][:, :3], 'pose': batch['handPose'][:, 3:],
    #                   'trans': batch['handTrans'], 'shape': batch['handBeta']}
    #         handV, handJ, handF = self.manolayer.mesh_data_np(params, is_right=True)
    #         if self.dataset_name == 'ho3d':
    #             camEx = batch['camEx'].detach().cpu().numpy()
    #             hand_joints = apply_transform_joint(handJ, camEx)
    #
    #         for i in range(len(batch['objName'])):
    #             hand_mesh = trimesh.Trimesh(handV[i], handF)
    #
    #             if self.dataset_name == 'ho3d':
    #                 omodel = read_obj(osp.join(self.obj_model_path, 'models', batch['objName'][i], 'textured_simple.obj'))
    #                 objR = axis_angle_to_matrix(batch['objRot'][i])
    #                 obj_verts = np.copy(omodel.v @ objR.detach().cpu().numpy().T + batch['objTrans'][i].detach().cpu().numpy())
    #                 obj_faces = np.copy(omodel.f)
    #             elif self.dataset_name == 'grab':
    #                 obj_verts, obj_faces = self.dataset.get_obj_mesh(batch['objName'][i], batch['objRot'][i],
    #                                                                  batch['objTrans'][i])
    #             obj_mesh = trimesh.Trimesh(obj_verts, obj_faces)
    #
    #             if self.dataset_name == 'ho3d':
    #                 hand_mesh.apply_transform(camEx[i])
    #                 obj_mesh.apply_transform(camEx[i])
    #             hand_models.append(hand_mesh)
    #             obj_models.append(obj_mesh)
    #     elif dataset == 'grab':
    #         hand_joints = []
    #         for i in range(len(batch['objName'])):
    #             obj_verts, obj_faces = self.dataset.get_obj_mesh(
    #                 batch['objName'][i], batch['objRot'][i], batch['objTrans'][i])
    #             side = batch['handSide'][i]
    #             handV = batch['handVerts'][i].squeeze().detach().cpu().numpy()
    #             handF = self.manolayer.mano_f[side].copy()
    #             if side == 'left':
    #                 handV[:, 0] *= -1
    #                 handF[:, [0, 1]] = handF[:, [1, 0]]
    #                 obj_verts[:, 0] *= -1
    #                 obj_faces[:, [0, 1]] = obj_faces[:, [1, 0]]
    #             hand_models.append(trimesh.Trimesh(handV, handF))
    #             obj_models.append(trimesh.Trimesh(obj_verts, obj_faces))
    #             hand_joints.append(batch['handJoints'][i, 0, :16].cpu().numpy())
    #         hand_joints = np.stack(hand_joints, axis=0)
    #
    #     return hand_models, hand_joints, obj_models
    #
    def visualize_ho_features(self, hand_mesh, obj_mesh, hand_contact, obj_contact, hand_part, obj_part, hand_pressure=None, obj_pressure=None):
        # 16 parts
        offset = np.array([0, 0.3, 0])
        offsetx = np.array([0.3, 0.3, 0])
        o3d_hand_mesh = o3dmesh_from_trimesh(hand_mesh)
        o3d_obj_mesh = o3dmesh_from_trimesh(obj_mesh)

        ## for contact
        mesh_set_color(hand_contact.detach().cpu().numpy(), o3d_hand_mesh)
        mesh_set_color(obj_contact.detach().cpu().numpy(), o3d_obj_mesh)

        ## for partmap
        hm, om = deepcopy(o3d_hand_mesh), deepcopy(o3d_obj_mesh)
        mesh_set_color(linear_normalize(hand_part, 0, 1), hm, cm['hsv'])
        # mask = obj_contact.squeeze() < 0.6
        mesh_set_color(linear_normalize(obj_part.detach().cpu().numpy(), 0, 1), om, cm['hsv'], brightness=obj_contact.detach().cpu().numpy())
        ## replace parts without contact using dark color.

        hm.translate(offset)
        om.translate(offset)

        obj_cmap = deepcopy(o3d_obj_mesh)
        obj_pmap = deepcopy(om)
        obj_cmap.translate(offsetx)
        obj_pmap.translate(offsetx)

        geometries = [o3d_hand_mesh, o3d_obj_mesh, hm, om, obj_cmap, obj_pmap]
        # print([np.array(m.vertex_colors).max() for m in geometries])
        o3d.visualization.draw_geometries(geometries)

        # part_colors = np.array([hsv_to_rgb(i / 16, 0.9, 0.7) for i in range(16)])
        # vert_colors = part_colors[hand_pid]
        # o3d_hand_mesh = o3dmesh_from_trimesh()
