import os
import os.path as osp
import pickle
import time
import numpy as np
import cv2
import torch
import coacd
import trimesh
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from copy import copy, deepcopy
import random
from collections import defaultdict

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torch.nn.utils.rnn import pad_sequence
from contactopt.diffcontact import calculate_contact_capsule
from common.utils.utils import trimesh2Mesh, linear_normalize
from common.utils.manolayer import get_part_meshes
from common.utils.geometry import quaternion2matrix, flip_x_axis, rodrigues_rot
from common.utils.vis import o3dmesh_from_trimesh, o3d_arrow, o3dmesh
import open3d as o3d
import open3d.visualization as vis
import matplotlib as mpl
from matplotlib import pyplot as plt


def apply_transform_joint(joints, T):
    """
    joints: B x N x 3
    T: B x 4 x 4
    """
    b, n, _ = joints.shape
    homo_joints = np.concatenate((joints, np.ones((b, n, 1))), axis=2)
    T_joints = homo_joints @ T.transpose(0, 2, 1)
    T_joints = T_joints[:, :, :3] / T_joints[:, :, 3:4]
    return T_joints

class HandObject:
    def __init__(self, device, hand_faces, hand_part_ids, pressure_quantise_splits=None, contact_th=0.11, normalize=True):
        self.device = device
        self.contact_th = contact_th
        self.hand_faces = hand_faces
        self.hand_part_ids = hand_part_ids
        self.pressure_quantise_splits = pressure_quantise_splits
        self.simu_disps = None
        self.simu_label_disp = None
        self.obj_verts = None
        self.contact_map = None
        self.part_map = None
        self.pressure_map = None
        self.quantised_pressure = None
        self.obj_normals = None
        self.onehot_pressure = None
        self.hand_joints = None
        self.hand_models = []
        self.obj_models = []
        self.obj_hulls = []
        self.obj_names = []
        self.normalize = normalize
        self.simu_force_vecs, self.simu_contact_pts, self.simu_part_ids = [], [], []
        # self.obj_mass = [] # default value. unit: kg
        self.gravity_direction = []

    def load_from_ho_mesh(self, hand_meshes, object_meshes):
        self.hand_meshes = hand_meshes
        self.object_meshes = object_meshes

    # def normalize_object(self):
    #     offsets = torch.mean(self.obj_verts, dim=1, keepdim=True)
    #     self.obj_verts = self.obj_verts - offsets
    #     for i, m in enumerate(self.obj_models):
    #         m.vertices -= offsets[i].detach().cpu().numpy()
    #     for i, hs in enumerate(self.obj_hulls):
    #         for h in hs:
    #             h.vertices -= offsets[i].detach().cpu().numpy()
    #
    def __copy__(self):
        new_ho = HandObject(device=self.device, hand_faces=self.hand_faces, hand_part_ids=self.hand_part_ids, pressure_quantise_splits=self.pressure_quantise_splits, normalize=self.normalize)
        new_ho.pressure_quantise_splits = copy(self.pressure_quantise_splits)
        new_ho.contact_th = self.contact_th
        # new_ho.obj_mass = copy(self.obj_mass)
        new_ho.obj_names = self.obj_names
        new_ho.gravity_direction = copy(self.gravity_direction)
        new_ho.hand_part_ids = copy(self.hand_part_ids)
        new_ho.simu_disps = copy(self.simu_disps)
        new_ho.simu_label_disp = copy(self.simu_label_disp)
        new_ho.obj_verts = copy(self.obj_verts)
        new_ho.contact_map = copy(self.contact_map)
        new_ho.part_map = copy(self.part_map)
        new_ho.obj_normals = copy(self.obj_normals)
        new_ho.hand_joints = copy(self.hand_joints)
        new_ho.quantised_pressure = copy(self.quantised_pressure)
        new_ho.onehot_pressure = copy(self.onehot_pressure)
        new_ho.hand_models = deepcopy(self.hand_models)
        new_ho.obj_models = deepcopy(self.obj_models)
        new_ho.obj_hulls = deepcopy(self.obj_hulls)
        new_ho.simu_force_vecs = deepcopy(self.simu_force_vecs)
        new_ho.simu_contact_pts = deepcopy(self.simu_contact_pts)
        new_ho.simu_part_ids = deepcopy(self.simu_part_ids)
        return new_ho

    def load_from_batch(self, batch, obj_templates=None, obj_hulls=None, inv_obj_rot=False):
        """
        Load the sampled vertices of objects from batched data. Used for training & testing.
        Force labels are also loaded.
        Used in training and validating
        """
        self.obj_verts = batch['objSamplePts'].clone().to(self.device)
        # self.obj_mass = batch['objMass'].to(self.device)
        self.obj_com = batch['objCoM'].clone().to(self.device)
        self.obj_names = batch['objName']
        self.gravity_direction = torch.zeros((self.obj_verts.shape[0], 3), device=self.device).float()
        self.gravity_direction[:, 2] = -1.0
        # self.contact_map = batch['contact'].to(self.device)
        # self.part_map = batch['contactPart'].to(self.device)
        self.obj_normals = batch['objSampleNormals'].clone().to(self.device)
        handV, handJ = batch['handVerts'].clone().cpu().numpy(), batch['handJoints'][:, :16].clone().cpu().numpy()
        ## Calculate contacts
        cmap, _, _, _, nn_idx = calculate_contact_capsule(batch['handVerts'], batch['handNormals'],
                                                          batch['objSamplePts'], batch['objSampleNormals'], fix_normal=True)
        self.contact_map = cmap.to(self.device).squeeze(-1)
        pmap = torch.as_tensor(self.hand_part_ids).to(self.device)[nn_idx].squeeze(-1)
        self.part_map = F.one_hot(pmap, 16).float()
        self.hand_models = []
        self.obj_models = []
        self.obj_hulls = []
        for b in range(handV.shape[0]):
            objR = axis_angle_to_matrix(batch['objRot'][b]).detach().cpu().numpy()
            objt = batch['objTrans'][b].detach().cpu().numpy()
            T = np.eye(4)
            T[:3, :3] = objR.T if inv_obj_rot else objR
            T[:3, 3] = objt
            if obj_templates is not None:
                obj_mesh = copy(obj_templates[b])
                obj_mesh.apply_transform(T)
                if batch['handSide'][b] == 'left':
                    flip_x_axis(obj_mesh)
                self.obj_models.append(obj_mesh)
            if obj_hulls is not None:
                ohs = []
                for h in obj_hulls[b]:
                    h0 = copy(h)
                    h0.apply_transform(T)
                    if batch['handSide'][b] == 'left':
                        flip_x_axis(h0)
                    ohs.append(h0)
                self.obj_hulls.append(ohs)

            if batch['handSide'][b] == 'left':
                ## flip all samples to the right side:
                handV[b, :, 0] *= -1
                handJ[b, :, 0] *= -1
                self.obj_verts[b, :, 0] *= -1
                self.obj_normals[b, :, 0] *= -1
                self.obj_com[b, 0] *= -1
                # for c in simu_labels[b]:
                #     c['frame'][[0, 3, 6]] *= -1
                #     c['contact_pt'][0] *= -1
                # tmpF = self.hand_faces.copy()
                # tmpF[:, [0, 1]] = tmpF[:, [1, 0]]
                self.hand_models.append(trimesh.Trimesh(handV[b], self.hand_faces.copy()))
            else:
                self.hand_models.append(trimesh.Trimesh(handV[b], self.hand_faces.copy()))

        self.hand_joints = handJ
        ## Calculate the pressure heatmap:
        # self.calculate_gaussian_pressure_map(handV, simu_labels, 0.005)
        obj_com_np = self.obj_com.detach().cpu().numpy().reshape(-1, 1, 3)
        self.obj_verts -= self.obj_com.view(-1, 1, 3)
        # self.obj_vert_centre = torch.mean(self.obj_verts, dim=1, keepdim=True)
        if self.normalize:
            # obj_ctr_npy = self.obj_vert_centre.detach().cpu().numpy()
            # self.obj_verts = self.obj_verts - self.obj_vert_centre
            for i in range(self.obj_verts.shape[0]):
                if len(self.obj_models):
                    self.obj_models[i].vertices -= obj_com_np[i]
                if len(self.obj_hulls):
                    for h in self.obj_hulls[i]:
                        h.vertices -= obj_com_np[i]
                self.hand_models[i].vertices -= obj_com_np[i]
                self.hand_joints[i] -= obj_com_np[i]
                # if 'simuContacts' in batch:
                #     for c in batch['simuContacts'][i]:
                #         c['contact_pt'] -= obj_com_np[i].flatten()

        # if len(self.obj_models):
        #     self.coms = torch.as_tensor(np.stack([om.center_mass for om in self.obj_models], axis=0), device=self.device).float()
        # else:
        #     self.coms = self.obj_vert_centre.squeeze(1)

        if 'simuContacts' in batch:
            self.calculate_pressure(batch['simuContacts'], batch['simuDisp'], batch['labelDisp'])

    def load_from_batch_obj_only(self, batch, obj_templates=None, obj_hulls=None, inv_obj_rot=False):
        """
        Load the data of objects only.
        """
        self.obj_verts = batch['objSamplePts'].clone().to(self.device)
        self.obj_com = batch['objCoM'].clone().to(self.device)
        self.obj_names = batch['objName']
        self.gravity_direction = torch.zeros((self.obj_verts.shape[0], 3), device=self.device).float()
        self.gravity_direction[:, 2] = -1.0
        self.obj_normals = batch['objSampleNormals'].clone().to(self.device)
        self.obj_models = []
        self.obj_hulls = []
        for b in range(self.obj_verts.shape[0]):
            objR = axis_angle_to_matrix(batch['objRot'][b]).detach().cpu().numpy()
            T = np.eye(4)
            T[:3, :3] = objR.T if inv_obj_rot else objR
            if obj_templates is not None:
                obj_mesh = copy(obj_templates[b])
                obj_mesh.apply_transform(T)
                self.obj_models.append(obj_mesh)
            if obj_hulls is not None:
                ohs = []
                for h in obj_hulls[b]:
                    h0 = copy(h)
                    h0.apply_transform(T)
                    ohs.append(h0)
                self.obj_hulls.append(ohs)

        ## Calculate the pressure heatmap:
        obj_com_np = self.obj_com.detach().cpu().numpy().reshape(-1, 1, 3)
        self.obj_verts -= self.obj_com.view(-1, 1, 3)
        if self.normalize:
            for i in range(self.obj_verts.shape[0]):
                if len(self.obj_models):
                    self.obj_models[i].vertices -= obj_com_np[i]
                if len(self.obj_hulls):
                    for h in self.obj_hulls[i]:
                        h.vertices -= obj_com_np[i]

    def update_contact_from_hand_models(self):
        handV = torch.as_tensor(np.stack([hm.vertices for hm in self.hand_models], axis=0)).to(self.device)
        handN = torch.as_tensor(np.stack([hm.vertex_normals for hm in self.hand_models], axis=0)).to(self.device)
        cmap, _, _, _, nn_idx = calculate_contact_capsule(handV, handN, self.obj_verts, self.obj_normals, fix_normal=True)
        self.contact_map = cmap.to(self.device).squeeze(-1)
        pmap = torch.as_tensor(self.hand_part_ids).to(self.device)[nn_idx].squeeze(-1)
        self.part_map = F.one_hot(pmap, 16).float()

    def calculate_pressure(self, simu_labels, simu_disps, simu_label_disp):
        self.simu_disps = simu_disps.to(self.device)
        self.simu_label_disp = simu_label_disp.to(self.device)
        self.calculate_even_pressure_map(simu_labels)
        self.simu_force_vecs, self.simu_contact_pts, self.simu_part_ids = [], [], []

        for b in range(self.obj_verts.shape[0]):
            simu_force_vecs, simu_contact_pts, simu_part_ids = [], [], []
            for contacts in simu_labels[b]:
                part_id, frame, force_vec, contact_pt = (
                    contacts['part_id'], torch.as_tensor(contacts['frame']).float().to(self.device),
                    torch.as_tensor(contacts['force']).float().to(self.device),
                    torch.as_tensor(contacts['contact_pt']).float().to(self.device))
                ## Tmp code for visualizing only normal forces
                frame[3:] = 0
                force_vec_glob = (frame.reshape(3, 3).T @ force_vec.reshape(3, 1)).reshape(3)
                simu_force_vecs.append(force_vec_glob)
                simu_contact_pts.append(contact_pt)
                simu_part_ids.append(part_id)
            if len(simu_force_vecs):
                simu_force_vecs = torch.stack(simu_force_vecs, dim=0)
            if len(simu_contact_pts):
                simu_contact_pts = torch.stack(simu_contact_pts, dim=0)
            # if len(simu_part_ids):
            #     simu_part_ids = torch.as_tensor(simu_part_ids, dtype=torch.uint8)
            self.simu_force_vecs.append(simu_force_vecs)
            self.simu_contact_pts.append(simu_contact_pts)
            self.simu_part_ids.append(simu_part_ids)

        ## Quantise the pressure map:
        pmap = torch.sum(self.pressure_map, dim=-1)  ## ignore part output;
        self.quantised_pressure = torch.zeros_like(pmap, device=self.device, dtype=torch.int64)
        for q in self.pressure_quantise_splits:
            mask = pmap > q
            self.quantised_pressure[mask] += 1
        self.onehot_pressure = F.one_hot(self.quantised_pressure, num_classes=len(self.pressure_quantise_splits) + 1).float()

    def calculate_gaussian_pressure_map(self, handV, simu_labels, sigma=0.005):
        """
        Alternative method to calculate pressure map based on point contacts
        """
        self.pressure_map = torch.zeros((*self.obj_verts.shape[:2], 16)).to(self.device)
        contact_mask = self.contact_map > 0
        for b in range(handV.shape[0]):
            obj_verts = self.obj_verts[b]
            obj_normals = self.obj_normals[b]
            for contacts in simu_labels[b]:
                part_id, frame, force_vec, contact_pt = (
                        contacts['part_id'], torch.as_tensor(contacts['frame']).float().to(self.device),
                        torch.as_tensor(contacts['force']).float().to(self.device),
                        torch.as_tensor(contacts['contact_pt']).float().to(self.device))
                dist = torch.norm(obj_verts - contact_pt.view(1, 3), dim=-1)
                dist_mask = dist < 3 * sigma
                dot_prod = - torch.sum(obj_normals * frame[0:3].reshape(1, 3), dim=1)
                dotprod_mask = dot_prod < 0
                # mask = torch.logical_and(dist_mask, dotprod_mask)
                # mask = torch.logical_and(mask, contact_mask[b])
                mask = dist_mask & dotprod_mask & contact_mask[b]
                dist = dist[mask]
                ## Generate pressure maps like Gaussian distributions
                pressure = force_vec[0]
                ratios = torch.exp(- dist ** 2 / (2 * sigma ** 2))
                ratios = ratios / ratios.sum()  # Normalize the sum to 1
                self.pressure_map[b, mask, part_id] += pressure * ratios

    def calculate_even_pressure_map(self, simu_labels):
        """
        Method to calculate the pressure evenly distributed over the contact area.
        """
        self.pressure_map = torch.zeros((*self.obj_verts.shape[:2], 16)).to(self.device)
        contact_mask = self.contact_map > self.contact_th
        pmap = (self.part_map @ torch.arange(16, device=self.device).view(1, 16, 1).float()).squeeze(-1)
        for b in range(self.obj_verts.shape[0]):
            obj_verts = self.obj_verts[b]
            ## Reorganize part id
            map_part2contact = defaultdict(list)
            for c in simu_labels[b]:
                map_part2contact[c['part_id']].append(c)
            for pid in range(16):
                if pid in map_part2contact:
                    part_mask = pmap[b] == pid
                    part_mask = torch.logical_and(part_mask, contact_mask[b])
                    if len(map_part2contact[pid]) == 1:
                        frame = torch.as_tensor(map_part2contact[pid][0]['frame']).float().to(self.device)
                        force_vec = torch.as_tensor(map_part2contact[pid][0]['force']).float().to(self.device)
                        pressure = force_vec[0]
                        dot_prod = torch.sum(self.obj_normals[b] * frame[0:3].reshape(1, 3), dim=1)
                        dotprod_mask = dot_prod < 0
                        mask = torch.logical_and(dotprod_mask, part_mask)
                        # mask = part_mask
                        if mask.sum().item() > 0:
                            self.pressure_map[b, mask, pid] = pressure / mask.sum()
                    else:
                        dists = []
                        point_pressure = []
                        masks = []
                        for c in map_part2contact[pid]:
                            frame = torch.as_tensor(c['frame']).float().to(self.device)
                            force_vec = torch.as_tensor(c['force']).float().to(self.device)
                            contact_pt = torch.as_tensor(c['contact_pt']).float().to(self.device)
                            dot_prod = torch.sum(self.obj_normals[b] * frame[0:3].reshape(1, 3), dim=1)
                            dotprod_mask = dot_prod < 0
                            point_pressure.append(force_vec[0])
                            masks.append(dotprod_mask & part_mask)
                            # masks.append(part_mask)
                            dists.append(torch.norm(obj_verts - contact_pt.view(1, 3), dim=-1))
                        ## Only the point with the least distance will be spread.
                        dists = torch.stack(dists, dim=0)
                        min_id = torch.min(dists, dim=0)[1]
                        for i, pressure in enumerate(point_pressure):
                            mask = masks[i] & (min_id == i)
                            if mask.sum().item() > 0:
                                self.pressure_map[b, mask, pid] = pressure / mask.sum()

    def get_phy_reps(self, use_features, random_rotate=False):
        """
        Obtain data directly used for training.
        The random rotations only affects the output of this function, not modifying the physics representation.
        """
        gravity_prod = torch.sum(self.gravity_direction.unsqueeze(1) * self.obj_normals, dim=-1, keepdim=True) # invariant in rotation
        if self.contact_map is not None:
            contact_map = self.contact_map.unsqueeze(-1)
        else:
            contact_map = None
        if random_rotate:
            rot_angle = np.random.rand(self.gravity_direction.shape[0]).reshape(-1, 1) * np.pi / 4
            theta = np.random.rand(self.gravity_direction.shape[0]) * 2 * np.pi
            gamma = np.random.rand(self.gravity_direction.shape[0]) * np.pi - np.pi / 2
            rot_axis = np.stack([np.cos(gamma) * np.cos(theta), np.cos(gamma) * np.sin(theta), np.sin(theta)], axis=1)
            # randrot = torch.as_tensor(R.random(self.gravity_direction.shape[0]).as_matrix()).to(self.device).float()
            randrot = torch.as_tensor(rodrigues_rot(rot_axis, rot_angle)).to(self.device).float()
            obj_verts = self.obj_verts @ randrot.transpose(-1, -2)
            obj_normals = self.obj_normals @ randrot.transpose(-1, -2)
            gravity_direction = self.gravity_direction.unsqueeze(1) @ randrot.transpose(-1, -2)
            coms = self.obj_com.unsqueeze(1) @ randrot.transpose(-1, -2)
            obj_features = {'normal': obj_normals, 'gravity_prod': gravity_prod, 'com_offsets': coms - obj_verts}
            obj_features = torch.cat([obj_features[k] for k in use_features], dim=-1)
            return obj_verts, obj_features, contact_map, self.part_map, self.onehot_pressure, gravity_direction.squeeze(1)
        else:
            obj_features = {'normal': self.obj_normals, 'gravity_prod': gravity_prod, 'com_offsets': self.obj_com.unsqueeze(1) - self.obj_verts}
            obj_features = torch.cat([obj_features[k] for k in use_features], dim=-1)
            return self.obj_verts, obj_features, contact_map, self.part_map, self.onehot_pressure, self.gravity_direction # torch.sum(self.pressure_map, dim=-1, keepdim=True)

    def calculate_pressure_from_onehot(self):
        return torch.sum(F.softmax(50 * self.onehot_pressure, dim=-1) * torch.arange(
            self.pressure_quantise_splits.shape[0] + 1, device=self.device).view(1, 1, -1) / 15, dim=-1)

    def load_from_batch_and_templates(self, batch: dict, obj_templates: list, obj_hulls: list, inv_obj_rot:bool=False, flip_side: np.ndarray=None):
        """
        This function loads exact object vertices & object hulls to the mesh model,
        particularly used for labelling.
        hand_params: rot_aa, trans, pose, shape;
        obj_templates: list of trimesh.Trimesh;
        obj_trans: object transformations: obj_rot, obj_trans
        Used in loading raw hand-object surface data.
        """
        batch_size = len(obj_templates)
        self.hand_models, self.obj_models, self.obj_hulls = [], [], []
        if flip_side is None:
            flip_side = np.array([False,] * batch_size)

        handV, handJ = batch['handVerts'].squeeze(1).cpu().numpy(), batch['handJoints'][:, 0, :16].cpu().numpy()
        if 'camEx' in batch:
            self.hand_joints = apply_transform_joint(handJ, batch['camEx'])
        else:
            self.hand_joints = handJ

        for i, temp in enumerate(obj_templates):
            hand_mesh = trimesh.Trimesh(handV[i], self.hand_faces)
            objR = axis_angle_to_matrix(batch['objRot'][i]).detach().cpu().numpy()
            objt = batch['objTrans'][i].detach().cpu().numpy()
            obj_mesh = copy(temp)
            T = np.eye(4)
            T[:3, :3] = objR.T if inv_obj_rot else objR
            T[:3, 3] = objt
            obj_mesh.apply_transform(T)
            obj_hull = deepcopy(obj_hulls[i])
            for j, h in enumerate(obj_hull):
                h.apply_transform(T)

            if 'camEx' in batch:
                hand_mesh.apply_transform(batch['camEx'][i])
                obj_mesh.apply_transform(batch['camEx'][i])
            if flip_side[i]:
                flip_x_axis(hand_mesh)
                flip_x_axis(obj_hull)
                # hand_mesh.vertices[:, 0] *= -1
                # hand_mesh.faces[:, [0, 1]] = hand_mesh.faces[:, [1, 0]]
                # obj_mesh.vertices[:, 0] *= -1
                # obj_mesh.faces[:, [0, 1]] = obj_mesh.faces[:, [1, 0]]
                for h in obj_hull:
                    flip_x_axis(h)
                    # h.vertices[:, 0] *= -1
                    # h.faces[:, [0, 1]] = h.faces[:, [1, 0]]
            self.hand_models.append(hand_mesh)
            self.obj_models.append(obj_mesh)
            self.obj_hulls.append(obj_hull)

    def aggregate_simu_forces(self):
        simu_forces = []
        simu_pts = []
        simu_pids = []
        for idx in range(len(self.obj_models)):
            tmp_simu_forces = []
            tmp_simu_pts = []
            tmp_simu_part_ids = []
            for pid in range(16):
                forces = []
                pts = []
                if len(self.simu_force_vecs[idx]):
                    for i in range(self.simu_force_vecs[idx].shape[0]):
                        if self.simu_part_ids[idx][i] == pid:
                            pt = self.simu_contact_pts[idx][i]
                            force = self.simu_force_vecs[idx][i]
                            forces.append(force)
                            pts.append(pt)
                if len(forces) > 0:
                    force = sum(forces)
                    pts = sum(pts) / len(pts)
                    tmp_simu_forces.append(force)
                    tmp_simu_pts.append(pts)
                    tmp_simu_part_ids.append(pid)
            simu_forces.append(torch.stack(tmp_simu_forces, dim=0))
            simu_pts.append(torch.stack(tmp_simu_pts, dim=0))
            simu_pids.append(tmp_simu_part_ids)

        self.simu_force_vecs = simu_forces
        self.simu_contact_pts = simu_pts
        self.simu_part_ids = simu_pids


    def get_vis_geoms(self, idx=0, draw_force_arrows=True, draw_multi_objs=True, show_procedure=False, show_optim_vars=False, draw_maps=True, **kwargs):
        default_mat = vis.rendering.MaterialRecord()
        default_mat.shader = 'defaultLit'
        hand_mat = vis.rendering.MaterialRecord()
        hand_mat.shader = "defaultLitTransparency"
        hand_mat.base_color = [0.8, 0.7, 0.5, 0.8]
        obj_mat = vis.rendering.MaterialRecord()
        obj_mat.shader = "defaultLitTransparency"
        obj_mat.base_color = [0.5, 0.5, 0.5, 0.9]
        obj_mat1 = vis.rendering.MaterialRecord()
        obj_mat1.shader = "defaultLitTransparency"
        obj_mat1.base_color = [0.5, 0.9, 0.4, 0.7]
        heat_cmap = plt.colormaps['inferno']
        part_cmap = plt.colormaps['hsv']

        vis_geoms = []
        ## Gravity
        grav_arrow = o3d_arrow(np.zeros(3), np.array([0, 0, -9.81]),
                          np.ones(3) * 0.1, scale=0.01, log=True)
        # vis_geoms.append({'name': 'coord_system', 'geometry': o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]), 'material': default_mat})
        if draw_force_arrows and len(self.simu_force_vecs[idx]) > 0:
            vis_geoms.append({'name': 'gravity', 'geometry': grav_arrow, 'material': default_mat})
            for i in range(self.simu_force_vecs[idx].shape[0]):
                if torch.norm(self.simu_force_vecs[idx][i]) > 1.0e-4:
                    arrow = o3d_arrow(self.simu_contact_pts[idx][i].detach().cpu().numpy(),
                                      self.simu_force_vecs[idx][i].detach().cpu().numpy(),
                                      part_cmap(self.simu_part_ids[idx][i] / 16)[:3], scale=0.01, log=True)
                    vis_geoms.append({'name': f'contact_{i}', 'geometry': arrow, 'material': default_mat})

        offset = np.array([0.40, 0, 0])
        hand = o3dmesh_from_trimesh(self.hand_models[idx], (0.8, 0.7, 0.5))
        hand_vcolor = part_cmap(self.hand_part_ids / 16)[:, :3] * 0.5 + 0.25
        # hand.vertex_colors = o3d.utility.Vector3dVector(hand_vcolor)
        obj_mesh = self.obj_models[idx]
        obj0 = o3dmesh_from_trimesh(obj_mesh, (0.5, 0.5, 0.5))
        # hand1 = copy(self.hand_models[idx])
        # hand1.apply_translation((-0.3, 0, 0))
        # hand1 = o3dmesh_from_trimesh(hand1, (0.8, 0.7, 0.5))
        om1 = copy(obj_mesh)
        om1.apply_translation(-offset*2)
        white_obj = o3dmesh_from_trimesh(om1, (0.5, 0.5, 0.5))
        hand1 = copy(hand).translate(-offset)
        obj1 = copy(obj0).translate(-offset)
        vis_geoms.extend([{'name': 'hand', 'geometry': hand, 'material': hand_mat},
                          {'name': 'hand1', 'geometry': hand1, 'material': default_mat},
                          {'name': 'obj1', 'geometry': obj1, 'material': default_mat},
                          # {'name': 'white_obj', 'geometry': white_obj, 'material': default_mat},
                          # {'name': 'hand1', 'geometry': hand1, 'material': default_mat},
                          {'name': 'object_before', 'geometry': obj0, 'material': obj_mat}])
        if draw_multi_objs:
            rot_mat = quaternion2matrix(self.simu_disps[idx, 3:].cpu().numpy().reshape(1, -1)).reshape(3, 3)
            T = np.eye(4)
            T[:3, :3] = rot_mat
            T[:3, 3] = self.simu_disps[idx, :3].cpu().numpy()
            obj_after = copy(obj_mesh)
            obj_after.apply_transform(T)
            T1 = np.eye(4)
            label_rot_mat = quaternion2matrix(self.simu_label_disp[idx, 3:].cpu().numpy().reshape(1, -1)).reshape(3, 3)
            T1[:3, :3] = label_rot_mat
            T1[:3, 3] = self.simu_label_disp[idx, :3].cpu().numpy()
            obj_label = copy(obj_mesh)
            obj_label.apply_transform(T1)
            obj1 = o3dmesh_from_trimesh(obj_after, (1.0, 0.2, 0.2))
            objlb = o3dmesh_from_trimesh(obj_label, (0.3, 0.5, 0.9))
            vis_geoms.extend([
                          {'name': 'object_after', 'geometry': obj1, 'material': obj_mat},
                          # {'name': 'object_label', 'geometry': objlb, 'material': obj_mat}
                          ])

        if draw_maps:
            comesh = copy(obj_mesh)
            comesh.apply_translation(offset)
            ## contact upscale
            dists = np.linalg.norm(self.obj_verts[idx].view(1, -1, 3).detach().cpu().numpy()
                                   - self.obj_models[idx].vertices.reshape(-1, 1, 3), axis=-1)
            nn_idx = np.argmin(dists, axis=1)
            up_contact = self.contact_map[idx, nn_idx]
            up_contact = linear_normalize(up_contact, 0, 1).detach().cpu().numpy()
            vertex_colors = heat_cmap(up_contact)[:, :3]
            vertex_colors[up_contact <= 0.1] = 0.1
            comesh = o3dmesh_from_trimesh(comesh, (0.5, 0.5, 0.5))
            comesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            pomesh = copy(obj_mesh)
            pomesh.apply_translation(offset*2)
            pomesh = o3dmesh_from_trimesh(pomesh)
            part_confs, part_ids = torch.max(self.part_map, dim=-1)
            up_part_ids = part_ids[idx, nn_idx].detach().cpu().numpy() / 16
            # up_part_confs = part_confs[idx, nn_idx].detach().cpu().numpy()
            vertex_colors = part_cmap(up_part_ids)[:, :3]
            vertex_colors *= up_contact.reshape(-1, 1) # Regularize by the contact maps
            # vertex_colors[up_part_confs <= 0.1] = 0.1
            pomesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            fomesh = copy(obj_mesh)
            fomesh.apply_translation(offset*3)
            fomesh = o3dmesh_from_trimesh(fomesh, (0.5, 0.5, 0.5))
            # quant_colors = heat_cmap(np.arange(16)/16)[:, :3]
            # vertex_colors = self.onehot_pressure[idx, nn_idx].detach().cpu().numpy() @ quant_colors
            self.quantised_pressure = torch.argmax(self.onehot_pressure, dim=-1)
            up_q_pressure = self.quantised_pressure[idx, nn_idx].detach().cpu().numpy() / 16
            vertex_colors = heat_cmap(up_q_pressure)[:, :3]
            vertex_colors *= up_contact.reshape(-1, 1) # Regularize by the contact maps
            # part_colors = part_cmap(np.arange(16)/16)[:, :3]
            # vertex_colors = linear_normalize(self.pressure_map[idx, nn_idx], 0, 1).detach().cpu().numpy() @ part_colors
            # vertex_colors = heat_cmap(up_fmap)[:, :3]
            # vertex_colors[vertex_colors <= 0.1] = 0.1
            fomesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            vis_geoms.extend([{'name': 'obj_contacts', 'geometry': comesh, 'material': default_mat},
                              {'name': 'obj_parts', 'geometry': pomesh, 'material': default_mat},
                              {'name': 'obj_force', 'geometry': fomesh, 'material': default_mat}])
            voffset = np.array([0, 0.2, 0])
        if show_procedure:
            ## split in parts:
            hand_part = get_part_meshes(self.hand_models[idx].vertices + np.array([[0.2, 0.2, 0]]), self.hand_models[idx].faces, self.hand_part_ids)
            for i, p in enumerate(hand_part):
                pm = o3dmesh_from_trimesh(p, color=part_cmap(i/16)[:3])
                p_hull = trimesh.convex.convex_hull(p)
                p_hull.apply_translation(np.array([0, 0.2, 0]))
                p_hull = o3dmesh_from_trimesh(p_hull, color=part_cmap(i/16)[:3])
                vis_geoms.extend([{'name': f'hand_part_{i}', 'geometry': pm, 'material': default_mat},
                                  {'name': f'hand_hull_{i}', 'geometry': p_hull, 'material': default_mat},])
                simu_ph = copy(p_hull)
                simu_ph.translate(np.array([0, 0.2, 0]))
                vis_geoms.append({'name': f'simu_ph_{i}', 'geometry': simu_ph, 'material': default_mat})

            obj_coacd_mesh = coacd.Mesh(obj_mesh.vertices + np.array([[0.4, 0.2, 0]]), obj_mesh.faces)
            obj_hulls = coacd.run_coacd(obj_coacd_mesh, threshold=0.05)
            for i, h in enumerate(obj_hulls):
                o_hull = o3dmesh(h[0], h[1])
                vis_geoms.append({'name': f'obj_hull_{i}', 'geometry': o_hull, 'material': default_mat})
                simu_oh = copy(o_hull)
                simu_oh.translate(np.array([-0.2, 0.4, 0]))
                vis_geoms.append({'name': f'simu_oh_{i}', 'geometry': simu_oh, 'material': default_mat})
        if 'history' in kwargs:
            ## visualize optimization history
            for i, hand_verts in enumerate(kwargs['history']):
                om = copy(obj_mesh)
                om.apply_translation(voffset * (i+1))
                # vis_geoms.append({'name': f'obj_history_{i}', 'geometry': o3dmesh_from_trimesh(om, (0.5, 0.5, 0.5)), 'material': default_mat})
                hm = trimesh.Trimesh(hand_verts[idx].detach().cpu().numpy(), self.hand_faces)
                hm.apply_translation(voffset * (i+1))
                hm = o3dmesh_from_trimesh(hm, (0.8, 0.7, 0.5))
                hm.vertex_colors = o3d.utility.Vector3dVector(hand_vcolor)
                vis_geoms.append({'name': f'hand_history_{i}', 'geometry': hm, 'material': default_mat})

        if show_optim_vars:
            refined_part_map = kwargs['refined_part_map'].detach().cpu().numpy() # B x N x 16
            rpomesh = copy(obj_mesh)
            rpomesh.apply_translation(voffset + offset*2)
            part_ids = np.sum(refined_part_map * np.arange(16).reshape(1, 16, 1), axis=1)
            up_part_ids = part_ids[idx, nn_idx] / 16
            vertex_colors = part_cmap(up_part_ids)[:, :3]
            vertex_colors *= refined_part_map[idx, :, nn_idx].sum(axis=1).reshape(-1, 1)
            rpomesh = o3dmesh_from_trimesh(rpomesh)
            rpomesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            fc_mask = kwargs['fc_part_mask'].detach().cpu().numpy() # parts that forms a stable grasp
            kp_ids = np.arange(16)[fc_mask[idx].flatten()]
            kp_maps = refined_part_map * fc_mask.reshape(-1, 16, 1)
            kp_mask = np.logical_not(np.sum(kp_maps, axis=1))
            vertex_colors[kp_mask[idx, nn_idx]] = 0.1
            kpomesh = copy(obj_mesh)
            kpomesh.apply_translation(voffset*2 + offset*2)
            kpomesh = o3dmesh_from_trimesh(kpomesh)
            kpomesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            target_pts = kwargs['target_pts'][idx].detach().cpu().numpy() + (offset * 2 + voffset * 2).reshape(1, 3)
            target_pts = target_pts[fc_mask[idx].flatten()]
            for i, pt in enumerate(target_pts):
                kp_circle = o3d.geometry.TriangleMesh.create_sphere(0.005)
                kp_circle.translate((pt[0], pt[1], pt[2]))
                kp_circle.paint_uniform_color(part_cmap(kp_ids[i]/16)[:3])
                # kp_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.detach().cpu().numpy()))
                vis_geoms.append({'name': f'keypoints_{i}', 'geometry': kp_circle})

            vis_geoms.extend([{'name': 'refined_part_map', 'geometry': rpomesh, 'material': default_mat},
                              {'name': 'key_part_map', 'geometry': kpomesh, 'material': default_mat}])

        return vis_geoms
        ## Vis force & displacements

    def vis_frame(self, idx=0, draw_force_arrows=True, draw_multi_objs=True, show_optim_vars=False, show_procedure=False, **kwargs):
        vis_geoms = self.get_vis_geoms(idx, draw_force_arrows=draw_force_arrows, draw_multi_objs=draw_multi_objs, show_procedure=show_procedure, show_optim_vars=show_optim_vars, **kwargs)
        o3d.visualization.draw(vis_geoms, show_skybox=False, lookat=[0, 1, 0], eye=[0, -1, 0], up=[0, 0, 1])

    def vis_history(self, idx, history, fc_part_mask, target_pts, **kwargs):
        vis = o3d.visualization.Visualizer()
        vis.create_window('history', 800, 600)
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([0, 1, 0])
        out_dir = osp.join('tmp', 'grab_video_out',f'{self.obj_names[idx]}-{idx}')
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        heat_cmap = plt.colormaps['inferno']
        part_cmap = plt.colormaps['hsv']
        obj_mesh = self.obj_models[idx]
        obj0 = o3dmesh_from_trimesh(obj_mesh, (0.7, 0.7, 0.7))
        vis_geoms = [obj0]

        ## Object point cloud
        obj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.obj_verts[idx].detach().cpu().numpy()))
        vis.add_geometry(obj_pc)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'point_cloud.png'))
        vis.remove_geometry(obj_pc)

        ## Only object:
        vis.add_geometry(obj0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'white_obj.png'))

        dists = np.linalg.norm(self.obj_verts[idx].view(1, -1, 3).detach().cpu().numpy()
                               - self.obj_models[idx].vertices.reshape(-1, 1, 3), axis=-1)
        nn_idx = np.argmin(dists, axis=1)
        up_contact = self.contact_map[idx, nn_idx]
        up_contact = linear_normalize(up_contact, 0, 1).detach().cpu().numpy()

        contact_colors = heat_cmap(up_contact)[:, :3]
        obj0.vertex_colors = o3d.utility.Vector3dVector(contact_colors)
        vis.update_geometry(obj0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'obj_contact.png'))

        self.quantised_pressure = torch.argmax(self.onehot_pressure, dim=-1)
        up_q_pressure = self.quantised_pressure[idx, nn_idx].detach().cpu().numpy() / 16
        vertex_colors = heat_cmap(up_q_pressure)[:, :3]
        vertex_colors *= up_contact.reshape(-1, 1)  # Regularize by the contact maps
        obj0.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        vis.update_geometry(obj0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'obj_pressure.png'))

        part_confs, part_ids = torch.max(self.part_map, dim=-1)
        up_part_ids = part_ids[idx, nn_idx].detach().cpu().numpy() / 16
        vertex_colors = part_cmap(up_part_ids)[:, :3]
        part_vertex_colors = vertex_colors * up_contact.reshape(-1, 1)  # Regularize by the contact maps
        obj0.vertex_colors = o3d.utility.Vector3dVector(part_vertex_colors)
        vis.update_geometry(obj0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'obj_part.png'))

        refined_part_map = kwargs['refined_part_map'].detach().cpu().numpy()  # B x 16 x N
        part_ids = np.sum(refined_part_map * np.arange(16).reshape(1, 16, 1), axis=1)
        up_part_ids = part_ids[idx, nn_idx] / 16
        vertex_colors = part_cmap(up_part_ids)[:, :3]
        vertex_colors *= refined_part_map[idx, :, nn_idx].sum(axis=1).reshape(-1, 1)
        obj0.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        vis.update_geometry(obj0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'refined_obj_parts.png'))

        fc_mask = fc_part_mask.detach().cpu().numpy()  # parts that forms a stable grasp
        kp_ids = np.arange(16)[fc_mask[idx].flatten()]
        target_pts = target_pts[idx].detach().cpu().numpy()
        target_pts = target_pts[fc_mask[idx].flatten()]
        for i, pt in enumerate(target_pts):
            kp_circle = o3d.geometry.TriangleMesh.create_sphere(0.005)
            kp_circle.translate((pt[0], pt[1], pt[2]))
            kp_circle.paint_uniform_color(part_cmap(kp_ids[i] / 16)[:3])
            vis_geoms.append(kp_circle)
        for geom in vis_geoms:
            vis.add_geometry(geom)

        fc_part_map = refined_part_map[idx, fc_mask[idx].flatten()]
        part_ids = np.sum(fc_part_map * kp_ids.reshape(1, -1, 1), axis=1)
        up_part_ids = part_ids[0, nn_idx] / 16
        vertex_colors = part_cmap(up_part_ids)[:, :3]
        vertex_colors *= refined_part_map[idx, :, nn_idx].sum(axis=1).reshape(-1, 1)
        obj0.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        vis.update_geometry(obj0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(osp.join(out_dir, 'fc_obj_parts.png'))

        hand = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(history[0][idx]), o3d.utility.Vector3iVector(self.hand_faces))
        hand_vcolor = part_cmap(self.hand_part_ids / 16)[:, :3] * 0.5 + 0.25
        hand.vertex_colors = o3d.utility.Vector3dVector(hand_vcolor)
        vis.add_geometry(hand)

        # Optimization Process
        obj0.vertex_colors = o3d.utility.Vector3dVector(part_vertex_colors)
        vis.update_geometry(obj0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(out_dir, 'opt_process.mp4'), fourcc, 30.0, (800, 600))

        for i in range(len(history)):
            hand.vertices = o3d.utility.Vector3dVector(history[i][idx])
            vis.update_geometry(hand)
            vis.poll_events()
            vis.update_renderer()
            image = np.asarray(vis.capture_screen_float_buffer())
            image = (255 * image).astype(np.uint8)
            # mpl.pyplot.imshow(image)
            # mpl.pyplot.show()
            out.write(image[:, :, ::-1])
        out.release()


    def vis_img(self, idx:int, h:int=600, w:int=800, pts:torch.Tensor=None) -> np.ndarray:
        """
        Visualize the hand-object as an image.
        pts: N x 3
        returns an image array of h x w x 3
        """
        vis_geoms = self.get_vis_geoms(idx, draw_force_arrows=False, draw_multi_objs=False, draw_maps=True)
        if pts is not None:
            kp_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.detach().cpu().numpy()))
            vis_geoms.append({'name': 'keypoints', 'geometry': kp_pc})
        vis = o3d.visualization.Visualizer()
        vis.create_window('Open3D', w, h)
        for geom in vis_geoms:
            vis.add_geometry(geom['geometry'])
        ctr = vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([0, 1, 0])
        # front_params = o3d.io.read_pinhole_camera_parameters('data/misc/viewpoints/front.json')
        # ctr.convert_from_pinhole_camera_parameters(front_params)
        result_imgs = []
        for i in range(4):
            if i > 0:
                ctr.rotate(0, h, xo=h/2)
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer()
            result_imgs.append(np.asarray(image))
        vis.destroy_window()
        return np.concatenate(result_imgs, axis=0)
        # o3d.visualization.draw(vis_geoms, show_skybox=False, on_animation_frame=rotate_view)

