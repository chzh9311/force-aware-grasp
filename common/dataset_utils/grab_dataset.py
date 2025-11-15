## Largely from https://github.com/shreyashampali/ho3d.git
## Implement LightningDataModule API for HO3Dv3
import sys

import trimesh
from trimesh import sample

for p in ['.', '..']:
    sys.path.append(p)
import os
import os.path as osp
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from lightning import LightningDataModule
from copy import copy, deepcopy
import math
import cv2
import smplx
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
from common.utils.manolayer import ManoLayer
from common.utils.geometry import rodrigues_rot
from common.utils.utils import force_padding_collate_fn
from easydict import EasyDict as edict
# from manopth.manolayer import ManoLayer
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from scipy.spatial.transform import Rotation as R

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

test_objects = ['wineglass', 'fryingpan', 'mug', 'toothpaste', 'camera', 'binoculars']

contact_ids={'Body': 1,
             'L_Thigh': 2,
             'R_Thigh': 3,
             'Spine': 4,
             'L_Calf': 5,
             'R_Calf': 6,
             'Spine1': 7,
             'L_Foot': 8,
             'R_Foot': 9,
             'Spine2': 10,
             'L_Toes': 11,
             'R_Toes': 12,
             'Neck': 13,
             'L_Shoulder': 14,
             'R_Shoulder': 15,
             'Head': 16,
             'L_UpperArm': 17,
             'R_UpperArm': 18,
             'L_ForeArm': 19,
             'R_ForeArm': 20,
             'L_Hand': 21,
             'R_Hand': 22,
             'Jaw': 23,
             'L_Eye': 24,
             'R_Eye': 25,
             'L_Index1': 26,
             'L_Index2': 27,
             'L_Index3': 28,
             'L_Middle1': 29,
             'L_Middle2': 30,
             'L_Middle3': 31,
             'L_Pinky1': 32,
             'L_Pinky2': 33,
             'L_Pinky3': 34,
             'L_Ring1': 35,
             'L_Ring2': 36,
             'L_Ring3': 37,
             'L_Thumb1': 38,
             'L_Thumb2': 39,
             'L_Thumb3': 40,
             'R_Index1': 41,
             'R_Index2': 42,
             'R_Index3': 43,
             'R_Middle1': 44,
             'R_Middle2': 45,
             'R_Middle3': 46,
             'R_Pinky1': 47,
             'R_Pinky2': 48,
             'R_Pinky3': 49,
             'R_Ring1': 50,
             'R_Ring2': 51,
             'R_Ring3': 52,
             'R_Thumb1': 53,
             'R_Thumb2': 54,
             'R_Thumb3': 55}

key2part_id = ['Hand', 'Index1', 'Index2', 'Index3', 'Middle1', 'Middle2', 'Middle3', 'Pinky1', 'Pinky2', 'Pinky3',
               'Ring1', 'Ring2', 'Ring3', 'Thumb1', 'Thumb2', 'Thumb3']

def regularize_part_id(contacts: torch.Tensor, hand_side: str):
    c2p = []
    for pstr in key2part_id:
        if hand_side == 'left':
            c2p.append(contact_ids['L_' + pstr])
        else:
            c2p.append(contact_ids['R_' + pstr])
    out_contacts = torch.zeros_like(contacts, dtype=torch.int64)
    for idx, cid in enumerate(c2p):
        out_contacts[contacts==cid] = idx + 1

    out_contacts = F.one_hot(out_contacts, num_classes=16 + 1)
    out_contacts = out_contacts[..., 1:] # no contact are labelled as all 0 vectors.

    return out_contacts

class GRABDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.randrot = True #not cfg.model.name == 'external'

    def prepare_data(self):
        """
        Calculate and cache the MANO & Object models
        """
        pass

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = GRABDataset(self.cfg, 'train', True)
            self.validate_set = GRABDataset(self.cfg, 'val', True)
        elif stage == 'validate':
            self.validate_set = GRABDataset(self.cfg, 'val', True)
        elif stage == 'test':
            self.test_set = GRABDataset(self.cfg, 'test', False, randrot=self.randrot)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=8, collate_fn=force_padding_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.validate_set, batch_size=self.val_batch_size, shuffle=False, num_workers=8, collate_fn=force_padding_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=8, collate_fn=force_padding_collate_fn)

class GRABDataset(Dataset):
    def __init__(self, cfg: edict, split: str, load_force: bool = False, randrot:bool = True):
        super().__init__()
        self.data_dir = cfg.dataset_path
        self.force_label_dir = osp.join(cfg.force_label_dir, f'grab_{split}')
        # self.force_label_dir = osp.join(cfg.force_label_dir, f'grab_train')
        self.pick_every_n_frames = cfg.pick_every_n_frames
        self.split = split
        self.n_samples = cfg.object_sample
        self.randrot = randrot

        ## load data
        self.obj_info = np.load(osp.join(self.data_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(osp.join(self.data_dir, 'sbj_info.npy'), allow_pickle=True).item()
        self.lh_data = torch.load(osp.join(self.data_dir, self.split, 'lhand_data.pt'), weights_only=True)
        self.rh_data = torch.load(osp.join(self.data_dir, self.split, 'rhand_data.pt'), weights_only=True)
        self.object_data = torch.load(osp.join(self.data_dir, self.split, 'object_data.pt'), weights_only=True)
        self.frame_info = np.load(osp.join(self.data_dir, self.split, 'frame_names.npz'))
        self.frame_names = self.frame_info['frame_names']
        self.hand_sides = self.frame_info['hand_sides']
        if self.pick_every_n_frames > 1:
            self.lh_data = self.sample_frames(self.lh_data)
            self.rh_data = self.sample_frames(self.rh_data)
            self.object_data = self.sample_frames(self.object_data)
            self.frame_names = self.sample_frames(self.frame_names)
            self.hand_sides = self.sample_frames(self.hand_sides)
        if split == 'test':
            self.num_samples = cfg.test_samples
            self.test_objects = test_objects

        ## get object surface normals
        for k, v in self.obj_info.items():
            mesh = trimesh.Trimesh(v['verts'], v['faces'], process=False)
            sample_pts, fid = sample.sample_surface(mesh, self.n_samples)
            v['samples'] = np.asarray(sample_pts)
            v['sample_normals'] = np.asarray(mesh.face_normals[fid])

        if os.path.exists(self.force_label_dir) and load_force:
            self.force_labels = []
            for label_file in sorted(os.listdir(self.force_label_dir)):
                with open(osp.join(self.force_label_dir, label_file), 'rb') as ff:
                    self.force_labels.extend(pickle.load(ff))
            # assert len(self.force_labels) == len(self.frame_names)

        ## Object hulls for simulation.
        self.obj_hulls = {}
        # self.obj_mass = {}
        self.obj_model = {}
        for obj_name in self.obj_info.keys():
            hulls = []
            hull_path = osp.join(self.data_dir, 'obj_hulls', obj_name)
            for i in range(len(os.listdir(hull_path))):
                hulls.append(trimesh.load(osp.join(hull_path, f'hull_{i}.stl')))

            self.obj_hulls[obj_name] = hulls
            obj_model = trimesh.Trimesh(self.obj_info[obj_name]['verts'], self.obj_info[obj_name]['faces'])
            self.obj_model[obj_name] = obj_model
            # self.obj_mass[obj_name] = obj_model.volume * 1000 # sum([h.volume for h in hulls]) * 1000 # 1000 kg / m^3

        self.rh_models = {}
        self.lh_models = {}
        for sbj_id in self.sbj_info.keys():
            self.rh_models[sbj_id] = smplx.create(model_path='./data/mano_v1_2',
                                                  model_type='mano',
                                                  is_rhand=True,
                                                  v_template=self.sbj_info[sbj_id]['rh_vtemp'],
                                                  num_pca_comps=45,
                                                  flat_hand_mean=True,
                                                  batch_size=1)
            self.lh_models[sbj_id] = smplx.create(model_path='./data/mano_v1_2',
                                                  model_type='mano',
                                                  is_rhand=False,
                                                  v_template=self.sbj_info[sbj_id]['lh_vtemp'],
                                                  num_pca_comps=45,
                                                  flat_hand_mean=True,
                                                  batch_size=1)
        # self.randrotmats = np.load(osp.join('data', 'misc', 'rand_rots.npy'))

    def sample_frames(self, data):
        if type(data) is dict:
            for k, v in data.items():
                data[k] = v[::self.pick_every_n_frames]
        else:
            data = data[::self.pick_every_n_frames]

        return data

    def __len__(self):
        # return 9000
        if self.split == 'test':
            return len(self.test_objects) * self.num_samples
        else:
            return len(self.frame_names)

    def __getitem__(self, idx):
        if self.split in ['train', 'val']:
            fname_path = self.frame_names[idx].split('/')
            sbj_id = fname_path[-2]
            obj_name = fname_path[-1].split('_')[0]
            obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3
            obj_com = self.obj_model[obj_name].center_mass
            obj_rot = self.object_data['global_orient'][idx]
            obj_trans = self.object_data['transl'][idx]

            ## Transform the vertices:
            objR = axis_angle_to_matrix(obj_rot).detach().cpu().numpy()
            objt = obj_trans.detach().cpu().numpy()
            obj_sample_pts = obj_sample_pts @ objR + objt
            obj_sample_normals = obj_sample_normals @ objR
            obj_com = obj_com.reshape(1, 3) @ objR + objt

            # mass = 0
            # for q in self.mass_mid_pts:
            #     if self.obj_mass[obj_name] > q:
            #         mass += 1
            # mass_embed = torch.zeros(16)
            # mass_embed[mass] = 1
            hand_side = self.hand_sides[idx]
            sample = {
                'frameName': '/'.join(fname_path[-2:]),
                'objName': obj_name,
                'sbjId': sbj_id,
                'objSamplePts': obj_sample_pts,
                'objSampleNormals': obj_sample_normals,
                'objTrans': obj_trans,
                'objRot': obj_rot,
                'handSide': hand_side,
                # 'objMass': float(self.obj_mass[obj_name]),
                'objCoM': obj_com.reshape(3),
                # 'contact': torch.clip(self.object_data['contact'][idx], 0, 1),
                # 'contactPart': regularize_part_id(self.object_data['contact'][idx], hand_side) # 0 - 15
            }

            if hasattr(self, 'force_labels'):
                flabel = self.force_labels[idx]
                # if flabel['frame_name'] == sample['frameName']:
                sample['simuContacts'] = flabel['contacts']
                sample['simuDisp'] = flabel['obj_disp']
                sample['labelDisp'] = flabel['label_obj_disp']

            if hand_side == 'left':
                # change the hand
                hmodels = self.lh_models
                hdata = self.lh_data
            else:
                hmodels = self.rh_models
                hdata = self.rh_data
            hand_out = hmodels[sbj_id](
                global_orient=hdata['global_orient'][idx:idx+1],
                hand_pose=hdata['fullpose'][idx:idx+1],
                transl=hdata['transl'][idx:idx+1])
            handV, handJ = hand_out.vertices[0].detach().cpu().numpy(), hand_out.joints[0].detach().cpu().numpy()
            handN = trimesh.Trimesh(handV, hmodels[sbj_id].faces).vertex_normals
            sample.update({
                'handRot': hdata['global_orient'][idx],
                'handPose': hdata['fullpose'][idx],
                'handTrans': hdata['transl'][idx],
                'handVerts': handV,
                'handJoints': handJ,
                'handNormals': np.stack(handN, axis=0),
            })
        else:
            obj_name = self.test_objects[int(idx // self.num_samples)]
            obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3
            obj_com = self.obj_model[obj_name].center_mass
            ## For testing, return object with random rotations. The rng is set to make sure
            ## each idx generates fixed rotation.
            if self.randrot:
                # objR = self.randrotmats[idx]
                # obj_rot = R.from_matrix(objR)
                obj_rot = R.random(rng=idx)
                objR = obj_rot.as_matrix()
            else:
                obj_rot = R.identity()
                objR = obj_rot.as_matrix()
            obj_sample_pts = obj_sample_pts @ objR
            obj_sample_normals = obj_sample_normals @ objR
            obj_com = obj_com.reshape(1, 3) @ objR
            sample = {
                'objName': obj_name,
                'objSamplePts': obj_sample_pts,
                'objSampleNormals': obj_sample_normals,
                'objRot': obj_rot.as_rotvec(),
                'objCoM': obj_com.reshape(3),
                # 'contact': torch.clip(self.object_data['contact'][idx], 0, 1),
                # 'contactPart': regularize_part_id(self.object_data['contact'][idx], hand_side) # 0 - 15
            }

            # hand_out = self.rh_models[sbj_id](
            #     global_orient=self.rh_data['global_orient'][idx:idx+1],
            #     hand_pose=self.rh_data['fullpose'][idx:idx+1],
            #     transl=self.rh_data['transl'][idx:idx+1],
            # )
            # handV, handJ = hand_out.vertices[0].detach().cpu().numpy(), hand_out.joints[0].detach().cpu().numpy()
            # handN = trimesh.Trimesh(handV, self.rh_models[sbj_id].faces).vertex_normals
            # sample.update({
            #     'handRot': self.rh_data['global_orient'][idx],
            #     'handPose': self.rh_data['fullpose'][idx],
            #     'handTrans': self.rh_data['transl'][idx],
            #     'handVerts': handV,
            #     'handJoints': handJ,
            #     'handNormals': np.stack(handN, axis=0),
            # })
        return sample

    def get_obj_mesh(self, obj_name, obj_rot, obj_trans):
        obj_verts = self.obj_info[obj_name]['verts']
        obj_faces = self.obj_info[obj_name]['faces']
        obj_verts = transform_obj(obj_verts, obj_rot.cpu().numpy(), obj_trans.cpu().numpy())
        return obj_verts, obj_faces

    def get_obj_hulls(self, obj_name, obj_rot, obj_trans):
        obj_hull = deepcopy(self.obj_hulls[obj_name])
        for i, h in enumerate(obj_hull):
            obj_hull[i].vertices = transform_obj(h.vertices, obj_rot.cpu().numpy(), obj_trans.cpu().numpy())

        return obj_hull


def transform_obj(verts, rot_aa, trans):
    """
    verts: N x 3
    """
    angle = np.linalg.norm(rot_aa) + 1.0e-8
    axis = rot_aa / angle
    R = rodrigues_rot(axis, angle)
    verts = verts @ R
    return verts + trans.reshape(1, 3)


def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                           thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):

            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn

def project_pts(pts: np.ndarray, cam_mat: np.ndarray):
    """
    Project global 3D coordinate to 2D plane.
    Used only for HO3D
    """
    pts = copy(pts)
    pts[:, 1:] *= -1
    homo_jt2d = pts @ cam_mat.T
    pts2d = np.stack((homo_jt2d[:, 0] / homo_jt2d[:, 2], homo_jt2d[:, 1] / homo_jt2d[:, 2]), axis=1)
    return pts2d


def global2local(x: np.ndarray, T: np.ndarray):
    """
    param x: the points to transform (N x 3)
    param T: the 4 x 4 transformation matrix of local coordinate system from global.
    """
    xh_global = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    T_inv = np.zeros_like(T)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = - T[:3, 3]
    T_inv[3, 3] = 1
    xh_local = (T_inv @ xh_global.T).T
    x_local = xh_local[:, :3] / xh_local[:, 3:4]
    return x_local


def vis_hand_on_img(sample):
    mano_model_path = './mano/models/MANO_RIGHT.pkl'
    manolayer = ManoLayer(mano_path='./data/mano_v1_2', use_pca=False, ncomps=6, flat_hand_mean=True)
    params = {'rot_aa': sample['handPose'][:, :3], 'pose': sample['handPose'][:, 3:],
              'trans': sample['handTrans'], 'shape': sample['handBeta']}
    handV, handJ, handF = manolayer.mesh_data_np(params, is_right=True)
    # handV, handJ = manolayer(sample['handPose'], sample['handBeta'])
    # handJ = handJ.detach().cpu().numpy()
    for i in range(4):
        # model = load_model(mano_model_path, ncomps=6, flat_hand_mean=True)
        # model.fullpose[:] = sample['handPose'][i]
        # model.trans[:] = sample['handTrans'][i]
        # model.betas[:] = sample['handBeta'][i]
        # handJ = model.J_transformed.r
        camMat = sample['camMat'][i].detach().cpu().numpy()
        glob_coord = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0) * 0.1
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        pts2d = project_pts(handJ[i, jointsMapManoToSimple], camMat)
        camEx = sample['camEx'][i].detach().cpu().numpy()
        g2l_coord = global2local(glob_coord, camEx)
        g2l_coord += np.mean(handJ[i], axis=0, keepdims=True) - g2l_coord[0:1]
        glob_proj = project_pts(g2l_coord, camMat).astype(np.int16)
        # img = sample['img'][i].permute(1, 2, 0).detach().cpu().numpy()
        img = cv2.imread(sample['imgPath'][i])[:, :, ::-1]
        annot_img = showHandJoints(img, pts2d)
        for i in range(3):
            cv2.line(annot_img, tuple(glob_proj[0]), tuple(glob_proj[i + 1]), colors[i], 2)
        plt.imshow(annot_img)
        plt.show()



