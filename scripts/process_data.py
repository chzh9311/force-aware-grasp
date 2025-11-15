import os
import sys
import yaml
import os.path as osp
import torch
from torch.utils.data import DataLoader
import numpy as np
import coacd
import trimesh
from easydict import EasyDict as edict
from progress.bar import Bar
import pickle

for p in ['.', '..']:
    sys.path.append(p)
from common.dataset_utils.arctic_dataset import ArcticMeshDataset
from common.utils.geometry import read_obj

def export_arctic_mean_var():
    with open(osp.join('configs', 'base.yaml'), 'r') as yf:
        cfg = edict(yaml.load(yf, Loader=yaml.FullLoader))
    dataset = ArcticMeshDataset(cfg.data, 'train', 0, False, normalize=False)
    all_data = {}
    for k in ['left_hand', 'right_hand']:
        all_data[k] = {}
        for kk in ['pose', 'shape', 'trans', 'rot_6d']:
            vv = dataset.data[k][kk]
            stats = vv.detach().cpu().numpy() # n_samples x ...
            all_data[k][kk+'_mean'] = stats.mean(axis=0)
            all_data[k][kk+'_std'] = stats.std(axis=0)
    all_data['object'] = {}
    for kk in ['arti', 'rot_aa', 'trans']:
        vv = dataset.data['object'][kk]
        stats = vv.detach().cpu().numpy()  # n_samples x ...
        all_data['object'][kk + '_mean'] = stats.mean(axis=0)
        all_data['object'][kk + '_std'] = stats.std(axis=0)

    with open(osp.join("data", "misc", "arctic_train_mean_std.pkl"), "wb") as f:
        pickle.dump(all_data, f)


def save_object_samples_and_contacts():
    ## TODO: fix the sampled points regarding the articulated parts.
    with open(osp.join('configs', 'base.yaml'), 'r') as yf:
        cfg = edict(yaml.load(yf, Loader=yaml.FullLoader))
    for sp in ['train', 'val']:
        dataset = ArcticMeshDataset(cfg.data, sp, 2048, True)
        loader = DataLoader(dataset, )


def export_obj_coacd_hulls():
    in_path = '/media/zxc417/data/data/GRAB/tools/object_meshes/contact_meshes/'
    out_path = '/media/zxc417/data/data/GRAB/processed/GRAB_V00/obj_hulls'
    bar = Bar('Processing object mesh', max=len(os.listdir(in_path)))
    for obj in os.listdir(in_path):
        obj_name = obj.split('.')[0]
        os.makedirs(os.path.join(out_path, obj_name), exist_ok=True)
        obj_mesh = trimesh.load(os.path.join(in_path, obj))
        mesh = coacd.Mesh(obj_mesh.vertices, obj_mesh.faces)
        obj_hulls = coacd.run_coacd(mesh, threshold=0.05)
        for i, h in enumerate(obj_hulls):
            obj_hull_mesh = trimesh.Trimesh(h[0], h[1])
            obj_hull_mesh.export(os.path.join(out_path, obj_name, f'hull_{i}.stl'))

        bar.next()


def export_ho3d_obj_coacd_hulls():
    in_path = 'data/YCB_Video/models'
    out_path = 'data/YCB_Video/obj_hulls'
    bar = Bar('Processing object mesh', max=len(os.listdir(in_path)))
    for obj_name in os.listdir(in_path):
        os.makedirs(os.path.join(out_path, obj_name), exist_ok=True)
        obj_mesh = read_obj(os.path.join(in_path, obj_name, 'textured_simple.obj'))
        mesh = coacd.Mesh(np.copy(obj_mesh.v), np.copy(obj_mesh.f))
        obj_hulls = coacd.run_coacd(mesh, threshold=0.05)
        for i, h in enumerate(obj_hulls):
            obj_hull_mesh = trimesh.Trimesh(h[0], h[1])
            obj_hull_mesh.export(os.path.join(out_path, obj_name, f'hull_{i}.stl'))

        bar.next()



if __name__ == "__main__":
    # export_arctic_mean_var()
    # export_obj_coacd_hulls()
    export_ho3d_obj_coacd_hulls()
