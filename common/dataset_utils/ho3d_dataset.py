## Largely from https://github.com/shreyashampali/ho3d.git
## Implement LightningDataModule API for HO3Dv3
import sys
for p in ['.', '..']:
    sys.path.append(p)
import os
import os.path as osp
import numpy as np
import pickle
import trimesh
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from lightning import LightningDataModule
from copy import copy
import math
import cv2
from common.utils.geometry import read_obj
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
from common.utils.manolayer import ManoLayer
from common.utils.utils import force_padding_collate_fn
# from manopth.manolayer import ManoLayer

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

class HO3DDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.ycb_dir = cfg.data.obj_model_path
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.randrot = True # not cfg.model.name == 'external'

    def prepare_data(self):
        """
        Calculate and cache the MANO & Object models
        """
        pass

    def setup(self, stage: str):
        if stage == 'fit' or stage == 'validate':
            ho3d_full = HO3DDataset(self.data_cfg, 'train')
            self.train_set, self.val_set = random_split(ho3d_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        else:
            self.test_set = HO3DDataset(self.data_cfg, 'test', randrot=self.randrot)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=8, collate_fn=force_padding_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=8, collate_fn=force_padding_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=8, collate_fn=force_padding_collate_fn)


class HO3DDataset(Dataset):
    def __init__(self, cfg, split: str, randrot: bool=False):
        super().__init__()
        self.data_dir = cfg.dataset_path
        self.ycb_dir = cfg.obj_model_path
        self.n_samples = cfg.object_sample
        self.split = split
        self.randrot = randrot
        self.num_samples = cfg.test_samples
        self.randrotmats = np.load(osp.join('data', 'misc', 'rand_rots.npy'))

        if not split == 'test':
            self.annots = self._load_annot(self.data_dir, self.ycb_dir, split)

        self.obj_info = {}
        # self.obj_names = sorted(os.listdir(osp.join(self.ycb_dir, 'models')))
        self.obj_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                             '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors',
                             '004_sugar_box']
        for obj_name in self.obj_names:
            omodel = read_obj(osp.join(self.ycb_dir, 'models', obj_name, 'textured_simple.obj'))
            omesh = trimesh.Trimesh(np.copy(omodel.v), np.copy(omodel.f))
            samples, fid = trimesh.sample.sample_surface(omesh, count=self.n_samples)
            self.obj_info[obj_name] = {'samples': np.asarray(samples), 'sample_normals':np.asarray(omesh.face_normals[fid]),
                                       'com': omesh.center_mass, 'verts': np.copy(omodel.v), 'faces': np.copy(omodel.f)}

        self.obj_hulls = {}
        # self.obj_mass = {}
        for obj_name in self.obj_info.keys():
            hulls = []
            hull_path = osp.join(self.ycb_dir, 'obj_hulls', obj_name)
            for i in range(len(os.listdir(hull_path))):
                hulls.append(trimesh.load(osp.join(hull_path, f'hull_{i}.stl')))

            self.obj_hulls[obj_name] = hulls

    def _load_annot(self, data_dir, ycb_dir, split):
        annots = []
        seq_cams = os.listdir(os.path.join(data_dir, split))
        calib_dir = os.path.join(data_dir, 'calibration')

        for seq_cam in seq_cams:
            for img in os.listdir(os.path.join(data_dir, split, seq_cam, 'rgb')):
                ## Load only seqs with camera extrinsics
                seqs = os.listdir(os.path.join(data_dir, 'calibration'))
                if seq_cam[:-1] in seqs:
                    annot = {'seq': seq_cam[:-1], 'cam_id': seq_cam[-1], 'id': img.split('.')[0],
                             'imgPath': os.path.join(data_dir, split, seq_cam, 'rgb', img)}
                    T = np.loadtxt(os.path.join(data_dir, 'calibration', seq_cam[:-1], 'calibration', f'trans_{seq_cam[-1]}.txt'))
                    annot['camEx'] = T
                    meta_file = os.path.join(data_dir, split, seq_cam, 'meta', annot['id'] + '.pkl')
                    assert os.path.exists(meta_file)
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                        try:
                            for k in ['handPose', 'handBeta', 'handTrans', 'objRot', 'objTrans', ]:
                                annot[k] = meta[k].flatten()
                            for k in ['objCorners3DRest', 'camMat', 'objName']:
                                annot[k] = meta[k]
                            annots.append(annot)
                        except AttributeError as e:
                            # print(f'Error loading annotation file {meta_file} due to {e}, ignored.')
                            continue
                # pose = meta['handPose']
                # glob_rot, theta = pose[:3], pose[3:]
                # annot['handRot'] = glob_rot
                # annot['handTheta'] = theta

        return annots

    def __len__(self):
        if self.split == 'test':
            return len(self.obj_names) * self.num_samples
        else:
            return len(self.annots)

    def __getitem__(self, idx):
        sample = {}
        if self.split in ['train', 'val']:
            get_keys = ['handBeta', 'handPose', 'handTrans', 'objRot', 'objTrans', 'objCorners3DRest', 'camEx', 'objName']
            if self.get_img:
                # img_path = os.path.join(self.data_dir, self.split, self.annots[idx]['seq'], 'rgb', self.annots[idx]['id']+'.png')
                # img = cv2.imread(img_path)
                # assert img is not None, f'{img_path} not found'
                # sample['camMat'] = self.annots[idx]['camMat']
                # sample['imgPath'] = self.annots[idx]['imgPath']
                get_keys.extend(['camMat', 'imgPath'])

            for k in get_keys:
                sample[k] = self.annots[idx][k]
        else:
            obj_name = self.obj_names[int(idx // self.num_samples)]
            obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3
            obj_com = self.obj_info[obj_name]['com']
            ## For testing, return object with random rotations. The rng is set to make sure
            ## each idx generates fixed rotation.
            if self.randrot:
                objR = self.randrotmats[idx]
                obj_rot = R.from_matrix(objR)
                # obj_rot = R.random(rng=idx)
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

        return sample

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


