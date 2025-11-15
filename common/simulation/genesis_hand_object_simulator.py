import pickle
import os
import os.path as osp
import argparse
import trimesh
from trimesh.exchange.obj import export_obj
import tempfile
import open3d as o3d

import numpy as np

import genesis as gs

class GenesisHandObjectSimulator:
    def __init__(self, device):
        ########################## init ##########################
        gs.init(backend=gs.cpu if device=='cpu' else gs.gpu)
        ########################## create a scene ##########################
        self.viewer_options = gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        )

        self.mat = gs.materials.Hybrid(
            mat_rigid=gs.materials.Rigid(
                gravity_compensation=1,
            ),
            mat_soft=gs.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                rho=10000,
                model='neohooken',
            ),
            thickness=0.001,
            damping=1000.,
            func_instantiate_rigid_from_soft=None,
            func_instantiate_soft_from_rigid=None,
            func_instantiate_rigid_soft_association=None,
        )
        self.coacd_options = gs.options.CoacdOptions(threshold=0.1, preprocess_resolution=100)

    def simulate(self, hand_mesh: trimesh.Trimesh | str, obj_mesh: trimesh.Trimesh | str):
        dt = 0.0001
        self.scene = gs.Scene(
            viewer_options=self.viewer_options,
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                gravity=(0, 0, -9.8),
                # use_contact_island=True,
                # use_hibernation=True,
                # hibernation_thresh_vel=0.00001,
                # hibernation_thresh_acc = 0.0001
            ),
            vis_options=gs.options.VisOptions(
                show_link_frame=False,
            ),
            mpm_options=gs.options.MPMOptions(
                dt=dt,
                grid_density=200,
                lower_bound=(-1.0, -1.0, -0.2),
                upper_bound=(1.0, 1.0, 1.0),
                gravity=(0, 0, 0)
            ),
        )

        # self.plane = self.scene.add_entity(gs.morphs.Plane(),)
        self.mu = 0.8
        # self.plane.set_friction(self.mu)

        ########################## file io  #########################
        tmp_dir = 'tmp/ho3dv3'
        if type(hand_mesh) == str:
            hand_tmp_fname = hand_mesh
        else:
            os.makedirs(tmp_dir, exist_ok=True)
            hand_tmp_fname = tempfile.mktemp(suffix=".obj", dir=tmp_dir)
            hand_mesh.export(hand_tmp_fname)
            print(f'saving hand mesh to {hand_tmp_fname}')
        if type(obj_mesh) == str:
            obj_tmp_fname = obj_mesh
        else:
            obj_tmp_fname = tempfile.mktemp(suffix=".obj", dir=tmp_dir)
            obj_mesh.export(obj_tmp_fname)
            print(f'saving object mesh to {obj_tmp_fname}')

        ########################## entities ##########################
        hand = self.scene.add_entity(
            gs.morphs.Mesh(file=hand_tmp_fname, convexify=True, decompose_nonconvex=True, fixed=True),
                           # coacd_options=self.coacd_options, fixed=True),
            # vis_mode='collision'
            # material=mat,
            # surface=gs.surfaces.Default()
        )
        hand.set_friction(self.mu)
        obj = self.scene.add_entity(
            gs.morphs.Mesh(file=obj_tmp_fname, convexify=True),
            visualize_contact=True,
            # vis_mode='collision'
        )
        obj.set_friction(self.mu)

        ########################## build ##########################
        self.scene.build()

        while(True):
            self.scene.step()


def saving_data():
    ### load data
    data = np.load('box_use_02.npy', allow_pickle=True).item()['world_coord']
    mano_path = '/home/zxc417/data/mano_v1_2'
    with open(osp.join(mano_path, "models", "MANO_LEFT.pkl"), "rb") as f:
        mano_left = pickle.load(f, encoding='latin1')
    with open(osp.join(mano_path, "models", "MANO_RIGHT.pkl"), "rb") as f:
        mano_right = pickle.load(f, encoding='latin1')
    idx = 450
    left_vert = data['verts.left'][idx] - np.array([0, 0, 0.6]).reshape(1, 3)
    left_f = np.array(mano_left['f'], dtype=np.int32)
    left_hand = trimesh.Trimesh(vertices=left_vert, faces=left_f)
    left_hand.export(osp.join('egmeshes', 'left_hand.obj'))
    right_vert = data['verts.right'][idx] - np.array([0, 0, 0.6]).reshape(1, 3)
    right_f = np.array(mano_right['f'], dtype=np.int32)
    right_hand = trimesh.Trimesh(vertices=right_vert, faces=right_f)
    right_hand.export(osp.join('egmeshes', 'right_hand.obj'))
    obj_vert = data['verts.object'][idx] - np.array([0, 0, 0.6]).reshape(1, 3)
    obj_f = data['f'][idx]
    object = trimesh.Trimesh(vertices=obj_vert, faces=obj_f)
    object.export(osp.join('egmeshes', 'object.obj'))



if __name__ == "__main__":
    simulator = HandObjectSimulator(device='gpu')
    hand_file = osp.join('tmp', 'ho3dv3', 'hand.obj')
    obj_file = osp.join('tmp', 'ho3dv3', 'object.obj')
    simulator.simulate(hand_file, obj_file)
    # hand_mesh = o3d.io.read_triangle_mesh(hand_file)
    # hand_mesh.compute_vertex_normals()
    # obj_mesh = o3d.io.read_triangle_mesh(obj_file)
    # obj_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([hand_mesh, obj_mesh])
    # main()
