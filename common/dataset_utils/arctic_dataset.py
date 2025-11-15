import os
import os.path as osp
import json
import loguru
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_axis_angle
import pickle
import open3d as o3d
from collections import defaultdict
from tqdm import tqdm

from common.dataset_utils.arctic_objects import ObjectTensors
from common.utils.utils import iter_merge_dict, sample_slice_from_gap, calc_num_gaps
from common.utils.geometry import calc_contacts, get_contact_map
from common.utils.manolayer import ManoLayer

import trimesh


def get_num_images(split, num_images):
    if split in ["train", "val", "test"]:
        return num_images

    if split == "minitrain":
        return 300

    if split == "minival":
        return 80

    if split == "minitest":
        return 200

    assert False, f"Invalid split {split}"


class ArcticMeshDataset(Dataset):
    def __init__(self, data_cfg, split, logger=None, normalize=True, **kwargs):
        # split: train, val, test
        super(ArcticMeshDataset, self).__init__()
        self.root = data_cfg.dataset_path
        self.split = split
        if logger is not None:
            self.logger = logger
        else:
            self.logger = loguru.logger

        with open(osp.join(self.root, 'data', 'arctic_data', 'data', 'splits_json', 'protocol_p1.json'), "r") as jsf:
            for sp in ['train', 'val', 'test']:
                if sp in self.split:
                    self.act_names = json.load(jsf)[sp]
                    break
        assert len(self.act_names), f"No split named {self.split}."
        self.data = {'left_hand': defaultdict(list), 'right_hand': defaultdict(list),
                     'object': defaultdict(list), 'meta': defaultdict(list)}
        self.obj_sample = data_cfg.object_sample

        self.mano_layers = ManoLayer(data_cfg.mano_path)

        for act_pth in self.act_names:
            subj, act = act_pth.split('/')
            obj_name = act.split('_')[0]
            act_data = self._load_frame_data(subj, act, obj_name)

            self.data = iter_merge_dict(self.data, act_data)

        for k in ['left_hand', 'right_hand', 'object']:
            for kk, vv in self.data[k].items():
                self.data[k][kk] = torch.stack(vv, dim=0)

        lh_v, lh_j, lh_f = self.mano_layers.rel_mano_forward(self.data['left_hand'], is_right=False,
                                                             ref_rot=self.data['object']['rot_aa'],
                                                             ref_trans=self.data['object']['trans'])
        rh_v, rh_j, rh_f = self.mano_layers.rel_mano_forward(self.data['right_hand'], is_right=True,
                                                             ref_rot=self.data['object']['rot_aa'],
                                                             ref_trans=self.data['object']['trans'])
        self.data['left_hand']['vertices'] = lh_v
        self.data['right_hand']['vertices'] = rh_v
        self.mano_left_f = lh_f
        self.mano_right_f = rh_f

        with open(osp.join('data', 'misc', 'arctic_train_mean_std.pkl'), 'rb') as f:
            self.train_mean_std = pickle.load(f)

        for k in ['left_hand', 'right_hand']:
            for kk, vv in self.train_mean_std[k].items():
                self.train_mean_std[k][kk] = torch.from_numpy(vv).float()
            self.train_mean_std[k]['param_mean'] = torch.cat((
                self.train_mean_std[k]['pose_mean'], self.train_mean_std[k]['shape_mean'],
                self.train_mean_std[k]['rot_6d_mean'], self.train_mean_std[k]['trans_mean'],
            ), dim=-1)

            self.train_mean_std[k]['param_std'] = torch.cat((
                self.train_mean_std[k]['pose_std'], self.train_mean_std[k]['shape_std'],
                self.train_mean_std[k]['rot_6d_std'], self.train_mean_std[k]['trans_std'],
            ), dim=-1)

        ## Normalize
        if normalize:
            self.data['left_hand'] = self.normalize_dict(self.data['left_hand'], 'left')
            self.data['right_hand'] = self.normalize_dict(self.data['right_hand'], 'right')

        ## The object contacts
        # obj_contact_cache_path = osp.join('data', 'misc', f'object_contacts_{split}_{obj_sample}.pkl')
        # if osp.exists(obj_contact_cache_path):
        #     self.logger.info(f"Loading contact data from {obj_contact_cache_path} ...")
        #     with open(obj_contact_cache_path, 'rb') as f:
        #         obj_contacts = pickle.load(f)
        #         for k in ['sampled_pts', 'lh_sdf', 'rh_sdf']:
        #             self.data['object'][k] = torch.from_numpy(obj_contacts[k]).float()
        #     self.logger.info("Done.")
        # else:
        #     self.logger.info(f"Processing contacts ...")
        #     for i in tqdm(range(self.data['object']['arti'].shape[0])):
        #         ## 10000 objects are processed together
        #         process_batch = 10000
        #         if i % process_batch == 0:
        #             ## process w/ object tensors in batch
        #             obj_out = self.object_tensors(self.data['object']['arti'][i:i+process_batch],
        #                                           self.data['object']['rot_aa'][i:i+process_batch],
        #                                           self.data['object']['trans'][i:i+process_batch],
        #                                           self.data['meta']['object_name'][i:i+process_batch])
        #         v = obj_out['v'][i % process_batch]
        #         f = obj_out['f'][i % process_batch]
        #         mesh = trimesh.Trimesh(vertices=v, faces=f)
        #         sampled_pts = trimesh.sample.sample_surface(mesh, self.obj_sample)[0]
        #         self.data['object']['sampled_pts'].append(torch.from_numpy(sampled_pts).float())
        #         # left hand
        #         lh_mesh = trimesh.Trimesh(vertices=self.data['left_hand']['vertices'][i].cpu().numpy(),
        #                                   faces=self.mano_left_f)
        #         # right hand
        #         rh_mesh = trimesh.Trimesh(vertices=self.data['right_hand']['vertices'][i].cpu().numpy(),
        #                                   faces=self.mano_right_f)
        #         lh_contacts, rh_contacts = calc_signed_distance(sampled_pts, lh_mesh, rh_mesh)
        #         self.data['object']['lh_contacts'].append(lh_contacts)
        #         self.data['object']['rh_contacts'].append(rh_contacts)
        #     res_dict = {}
        #     for k in ['sampled_pts', 'lh_contacts', 'rh_contacts']:
        #         self.data['object'][k] = torch.stack(self.data['object'][k], dim=0).float()
        #         res_dict[k] = self.data['object'][k]
        #
        #     with open(obj_contact_cache_path, 'wb') as f:
        #         pickle.dump(res_dict, f)
        #     self.logger.info(f"Dumped contact info cache to {obj_contact_cache_path}.")

    def _load_frame_data(self, subj, act, obj_name, rel_to_obj=True):
        # Use meter as the unit.
        frame_data = {'left_hand': {}, 'right_hand': {}, 'object': {}, 'meta': {}}
        obj_params = torch.from_numpy(
            np.load(osp.join(self.root, 'data', 'arctic_data', 'data', 'raw_seqs', subj, act+'.object.npy')))
        obj_rot = obj_params[:, 1:4]
        obj_trans = obj_params[:, 4:] / 1000
        n_sample = obj_params.shape[0]
        frame_data['object']['arti'] = obj_params[:, 0:1] # in radian
        frame_data['object']['rot_aa'] = obj_rot
        frame_data['object']['trans'] = obj_trans

        mano_params = np.load(osp.join(self.root, 'data', 'arctic_data', 'data', 'raw_seqs', subj, act+'.mano.npy'),
                              allow_pickle=True).item()
        n_samples = mano_params['left']['pose'].shape[0]
        frame_data['left_hand']['pose'] = torch.from_numpy(mano_params['left']['pose'])
        frame_data['left_hand']['shape'] = torch.from_numpy(mano_params['left']['shape']).repeat(n_samples, 1)
        lh_rot = axis_angle_to_matrix(torch.from_numpy(mano_params['left']['rot']))
        lh_trans = torch.from_numpy(mano_params['left']['trans'])
        if rel_to_obj:
            lh_rot = torch.linalg.solve(axis_angle_to_matrix(obj_rot), lh_rot)
            lh_trans -= obj_trans
        frame_data['left_hand']['rot_6d'] = matrix_to_rotation_6d(lh_rot)
        frame_data['left_hand']['rot_aa'] = matrix_to_axis_angle(lh_rot)
        frame_data['left_hand']['trans'] = lh_trans

        frame_data['right_hand']['pose'] = torch.from_numpy(mano_params['right']['pose'])
        frame_data['right_hand']['shape'] = torch.from_numpy(mano_params['right']['shape']).repeat(n_samples, 1)
        rh_rot = axis_angle_to_matrix(torch.from_numpy(mano_params['right']['rot']))
        rh_trans = torch.from_numpy(mano_params['right']['trans'])
        if rel_to_obj:
            rh_rot = torch.linalg.solve(axis_angle_to_matrix(obj_rot), rh_rot)
            rh_trans -= obj_trans
        frame_data['right_hand']['rot_6d'] = matrix_to_rotation_6d(rh_rot)
        frame_data['right_hand']['rot_aa'] = matrix_to_axis_angle(rh_rot)
        frame_data['right_hand']['trans'] = rh_trans

        ## These are surfaces in global coordinates
        # data = np.load(seq_p, allow_pickle=True).item()['world_coord']
        # frame_data['left_hand']['vertices'] += list(torch.from_numpy(data['verts.left']))
        # frame_data['right_hand']['vertices'] += list(torch.from_numpy(data['verts.right']))
        # frame_data['object']['faces'] += list(torch.from_numpy(data['f']))
        # frame_data['object']['vertices'] += list(torch.from_numpy(data['verts.object']))

        ## metadata
        # n_sample = data['verts.left'].shape[0]
        frame_data['meta']["subject"] = [subj] * n_sample
        frame_data['meta']["action"] = [act] * n_sample
        frame_data['meta']["object_name"] = [obj_name] * n_sample
        return frame_data

    def normalize_dict(self, dct, side, seq=False):
        # normalize:
        for p in ['pose', 'shape', 'rot_6d', 'trans']:
            if seq:
                dct[p] = (dct[p] - self.train_mean_std[side + '_hand'][p+'_mean'].reshape(1, 1, -1)) \
                            / self.train_mean_std[side + '_hand'][p+'_std'].reshape(1, 1, -1)
            else:
                dct[p] = (dct[p] - self.train_mean_std[side + '_hand'][p+'_mean'].unsqueeze(0)) \
                            / self.train_mean_std[side + '_hand'][p+'_std'].unsqueeze(0)
        return dct

    def denormalize_vector(self, params, side):
        """
        params: bs x size
        used in tensors
        """
        params = params * self.train_mean_std[side+'_hand']['param_std'].unsqueeze(0).to(params.device) \
                 + self.train_mean_std[side+'_hand']['param_mean'].unsqueeze(0).to(params.device)

        return params

    def denormalize_dict(self, hand_dict, side):
        for p in ['pose', 'shape', 'rot_6d', 'trans']:
            hand_dict[p] = hand_dict[p] * self.train_mean_std[side + '_hand'][p + '_std'].to(hand_dict[p].device) \
                           + self.train_mean_std[side + '_hand'][p+'_mean'].to(hand_dict[p].device)

        return hand_dict

    def __len__(self):
        num_imgs = self.data['left_hand']['pose'].shape[0]
        return get_num_images(self.split, num_imgs)

    def __getitem__(self, idx):
        out = {}
        for k in self.data.keys():
            out[k] = {}
            for kk, vv in self.data[k].items():
                out[k][kk] = vv[idx]

        return out


def get_samples_and_contacts(obj_param_dict: dict, hand_vs_dict:dict,
                             object_tensors: ObjectTensors, mano_layers: ManoLayer,
                             n_obj_sampled_pts, contact_dist, contact_th) -> dict:
    """
    :param obj_param_dict: {'arti', 'rot_aa', 'trans', 'object_name'}
    :param hand_vf_dict: {'lh_vs', 'rh_vs', 'lh_f', 'rh_f'}
    """
    obj_out = object_tensors(obj_param_dict['arti'], obj_param_dict['rot_aa'],
                             obj_param_dict['trans'], obj_param_dict['object_name'])
    v = obj_out['v'].cpu().numpy()
    f = obj_out['f'].cpu().numpy()
    bs = v.shape[0]
    mesh_data = defaultdict(list)
    lh_vs = hand_vs_dict['lh_vs']
    rh_vs = hand_vs_dict['rh_vs']
    for i in range(bs):
        mesh = trimesh.Trimesh(vertices=v[i], faces=f[i])
        sampled_pts, fidx = trimesh.sample.sample_surface(mesh, n_obj_sampled_pts)
        pts_normals = mesh.face_normals[fidx]
        mesh_data['sampled_pts'].append(sampled_pts)
        mesh_data['sampled_pts_normals'].append(pts_normals)
        # mesh_data['com'].append(mesh.center_mass)
        # left hand
        lh_mesh = trimesh.Trimesh(vertices=lh_vs[i], faces=mano_layers.mano_f['left'])
        # right hand
        rh_mesh = trimesh.Trimesh(vertices=rh_vs[i], faces=mano_layers.mano_f['right'])
        lh_contacts, _, rh_contacts, _ = calc_contacts(sampled_pts, lh_mesh, rh_mesh, contact_dist)

        contacts = get_contact_map(lh_contacts, rh_contacts, contact_th)
        mesh_data['contacts'].append(contacts)
        mesh_data['coms'].append(mesh.center_mass)

    return mesh_data

if __name__ == "__main__":
    pass
    # data = np.load(osp.join('/media/zxc417/data/arctic', seq_p), allow_pickle=True).item()
    # cam_data = data["world_coord"]
    # for k, v in cam_data.items():
    #     try:
    #         print(k, v.shape)
    #     except AttributeError:
    #         print(k, v)

# outputs of ArcticDataset
# img # torch.Size([3, 224, 224])
# mano.pose.r # torch.Size([48])
# mano.pose.l # torch.Size([48])
# mano.beta.r # torch.Size([10])
# mano.beta.l # torch.Size([10])
# mano.j2d.norm.r # torch.Size([21, 2])
# mano.j2d.norm.l # torch.Size([21, 2])
# object.kp3d.full.b # torch.Size([16, 3])
# object.kp2d.norm.b # torch.Size([16, 2])
# object.kp3d.full.t # torch.Size([16, 3])
# object.kp2d.norm.t # torch.Size([16, 2])
# object.bbox3d.full.b # torch.Size([8, 3])
# object.bbox2d.norm.b # torch.Size([8, 2])
# object.bbox3d.full.t # torch.Size([8, 3])
# object.bbox2d.norm.t # torch.Size([8, 2])
# object.radian # torch.Size([])
# object.kp2d.norm # torch.Size([32, 2])
# object.bbox2d.norm # torch.Size([16, 2])
# object.rot # torch.Size([1, 3])
# mano.j3d.full.r # torch.Size([21, 3])
# mano.j3d.full.l # torch.Size([21, 3])
# is_valid # 1.0
# left_valid # 1.0
# right_valid # 1.0
# joints_valid_r(21, )
# joints_valid_l(21, )
# imgname. / data / arctic_data / data / cropped_images / s09 / capsulemachine_use_01 / 4 / 00224.jpg
# kp3d.cano # torch.Size([16, 3])
# query_names # capsulemachine
# window_size # torch.Size([1])
# intrinsics # torch.Size([3, 3])
# dist # torch.Size([8])
# center(2, ) # is_flipped
# 0 # rot_angle()
# dataset = ArcticDataset('/media/zxc417/data/arctic', 'p1', 'minitrain', None, True) # data_slice = dataset[0]
# 10 viewpoints.

# outputs of seq_p = "outputs/processed_verts/seqs/s01/box_grab_01.npy"
# verts.right (732, 778, 3)
# joints.right (732, 21, 3)
# verts.left (732, 778, 3)
# joints.left (732, 21, 3)
# diameter (732,)
# f (732, 7998, 3)
# f_len (732,)
# v_len (732,)
# mask (732, 3947)
# parts_ids (732, 3947)
# bbox3d (732, 16, 3)
# kp3d (732, 32, 3)
# verts.object (732, 3947, 3)
# verts.smplx (732, 10475, 3)
# joints.smplx (732, 127, 3)
# rot_r (732, 3)
# rot_l (732, 3)
# obj_rot (732, 3)


