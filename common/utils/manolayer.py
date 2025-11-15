import os.path as osp
import torch
import numpy as np
import pickle
import open3d as o3d
import trimesh
from smplx import MANO
from colorsys import hsv_to_rgb
from pytorch3d.transforms import rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle

class ManoLayer(object):
    """
    The mano_layer class based on MANO from smplx
    """
    def __init__(self, mano_path, use_pca=False, ncomps=6, flat_hand_mean=False):
        with open(osp.join(mano_path, "models", "MANO_LEFT.pkl"), "rb") as f:
            mano_left = pickle.load(f, encoding='latin1')
        with open(osp.join(mano_path, "models", "MANO_RIGHT.pkl"), "rb") as f:
            mano_right = pickle.load(f, encoding='latin1')
        self.mano_f = {}
        self.mano_f['left'] = np.array(mano_left['f'], dtype=np.int32)
        self.mano_f['right'] = np.array(mano_right['f'], dtype=np.int32)
        self.ncomps = ncomps

        self.W_right = mano_right['weights']
        self.W_left = mano_left['weights']
        self.part_id_right = np.argmax(self.W_right, axis=1)
        self.part_id_left = np.argmax(self.W_right, axis=1)
        # self.part_color = {}
        # self.part_color['right'] = self.get_part_colors(self.W_right, hard=True)

        self.mano_layer = {}
        # self.mano_layer['left'] = ManoLayer(mano_root='./data/mano_v1_2/models', side='left', use_pca=False)
        # self.mano_layer['right'] = ManoLayer(mano_root='./data/mano_v1_2/models', side='right', use_pca=False)
        self.mano_layer['left'] = MANO('./data/mano_v1_2/models', create_tarnsl=False, use_pca=use_pca,
                                       num_pca_comps=ncomps, flat_hand_mean=flat_hand_mean, is_rhand=False)
        self.mano_layer['right'] = MANO('./data/mano_v1_2/models', create_tarnsl=False, use_pca=use_pca,
                                        num_pca_comps=ncomps, flat_hand_mean=flat_hand_mean, is_rhand=True)
        self.filled_palm_faces = np.load(osp.join('data', 'misc', 'filled_palm_faces.npy'))

    # def get_part_colors(self, W, hard=False):
    #     """W: 778 x 16"""
    #     part_colors = np.array([hsv_to_rgb(i / 16, 0.9, 0.7) for i in range(16)])
    #     if hard:
    #         pidx = np.argmax(W, axis=1)
    #         vert_colors = part_colors[pidx]
    #     else:
    #         vert_colors = np.clip(W @ part_colors, 0, 1)
    #     return vert_colors

    def mesh_data(self, params, is_right):
        side = 'right' if is_right else 'left'
        out = self.mano_layer[side](global_orient=params['rot_aa'], hand_pose=params['pose'], betas=params['shape'],
                                    transl=params['trans'])
        v = out.vertices
        j = out.joints
        f = self.mano_f[side]
        return v, j, f

    def mesh_data_np(self, params, is_right):
        v, j, f = self.mesh_data(params, is_right)
        return v.detach().cpu().numpy(), j.detach().cpu().numpy(), f

    def o3dmeshes(self, params, is_right):
        v, _, f = self.mesh_data_np(params, is_right)
        vs = [o3d.utility.Vector3dVector(v[i]) for i in range(v.shape[0])]
        f = o3d.utility.Vector3iVector(f)
        return [o3d.geometry.TriangleMesh(vertices=v, faces=f) for v in vs]

    def to(self, device):
        self.mano_layer['left'].to(device)
        self.mano_layer['right'].to(device)

    def rel_mano_forward(self, params, is_right, ref_rot=None, ref_trans=None):
        if type(params) is torch.Tensor:
            pose = params[:, :-19]
            shape = params[:, -19:-9]
            rot = rotation_6d_to_matrix(params[:, -9:-3])
            trans = params[:, -3:]
        else:
            pose = params['pose']
            shape = params['shape']
            trans = params['trans']
            rot = rotation_6d_to_matrix(params['rot_6d'])

        if ref_rot is None:
            ref_rot = torch.eye(3).view(1, 3, 3)
        else:
            ref_rot = axis_angle_to_matrix(ref_rot)
        if ref_trans is None:
            ref_trans = torch.zeros(1, 3)

        hand_params = {'rot_aa': matrix_to_axis_angle(ref_rot @ rot), 'trans': trans,
                       'pose': pose.float(), 'shape': shape.float()}
        v, j, f = self.mesh_data(hand_params, is_right)

        # Transform back to meter.
        handV = v + ref_trans.unsqueeze(1)
        handJ = j + ref_trans.unsqueeze(1)

        return handV, handJ, f


def get_part_meshes(verts: np.ndarray, faces: np.ndarray, pid: np.ndarray) -> list:
    """
    verts: 778 x 3
    return a list of trimesh
    """
    parts = []
    vert_ids = np.arange(verts.shape[0])
    for i in range(16):
        face_i = []
        mask = pid == i
        vert_i = verts[mask]
        vert_id_i = vert_ids[mask]
        for (t1, t2, t3) in faces:
            try:
                t1 = np.where(vert_id_i == t1)[0].item()
                t2 = np.where(vert_id_i == t2)[0].item()
                t3 = np.where(vert_id_i == t3)[0].item()
                face_i.append([t1, t2, t3])
            except:
                continue

        face_i = np.array(face_i, dtype=np.int32)
        ## Fill the hole of palm:
        # if i == 0:
        #     face_i = self.filled_palm_faces

        parts.append(trimesh.Trimesh(vert_i, face_i))

    return parts
