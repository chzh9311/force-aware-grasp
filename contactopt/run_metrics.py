# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

for p in ['.', '..']:
    sys.path.append(p)
import pickle
from open3d import visualization as o3dv
import random
import argparse
import numpy as np
import time
import util as util
import geometric_eval as geometric_eval
import pprint
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics
import trimesh
import os
import open3d as o3d
import json
import yaml
from contactopt.hand_object import HandObject
from manopth.manolayer import ManoLayer
import torch
from contactopt.evaluation.converter import transform_to_canonical, convert_joints
from contactopt.evaluation.displacement import grasp_displacement, diversity
from contactopt.evaluation.halo import intersection_eval
from contactopt.evaluation.vis import vis_dataset
import statistics
from contactopt.evaluation.mano_train.simulation.simulate import run_simulation, run_sim_parallel_interface

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from multiprocessing import Pipe
from multiprocessing.pool import ThreadPool, Pool

import pybullet
import pybullet_utils.bullet_client as bc

SAVE_OBJ_FOLDER = "eval/saveobj"


def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    """
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    """
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


def calc_mean_dicts(all_dicts, phase=""):
    keys = all_dicts[0].keys()
    mean_dict = dict()
    stds = ["pen_vol"]

    for k in keys:
        l = list()
        for d in all_dicts:
            l.append(d[k])
        mean_dict[k] = np.array(l).mean().item()

        if k in stds:
            mean_dict[k + "_std"] = np.array(l).std().item()

    return mean_dict


def calc_sample(ho_test, ho_gt, idx, phase="nophase"):
    stats = geometric_eval.geometric_eval(ho_test, ho_gt)

    return stats


def process_sample(sample, idx):
    gt_ho, in_ho = sample['gt_ho'], sample['in_ho']
    if 'opt_ho' in sample:
        opt_ho = sample['opt_ho']
        opt_stats = calc_sample(opt_ho, gt_ho, idx, 'after ContactOpt')
    else:
        opt_stats = None

    if 'phy_ho' in sample:
        phy_ho = sample['phy_ho']
        phy_stats = calc_sample(phy_ho, gt_ho, idx, 'after PhysOpt')
    else:
        phy_stats = None
    in_stats = calc_sample(in_ho, gt_ho, idx, 'before ContactOpt')

    return in_stats, opt_stats, phy_stats


def metrics_parallel_interface(param_dict):
    method, s, idx, rh_mano, device = (param_dict["method"], param_dict["s"], param_dict["idx"],
                                               param_dict['rh_mano'], param_dict['device'])
    pose_tensor = torch.Tensor(s[method + "_ho"].hand_pose).unsqueeze(0).to(device)
    beta_tensor = torch.Tensor(s[method + "_ho"].hand_beta).unsqueeze(0).to(device)
    tform_tensor = torch.Tensor(s[method + "_ho"].hand_mTc).unsqueeze(0).to(device)
    mano_verts, mano_joints = util.forward_mano(rh_mano, pose_tensor, beta_tensor, [tform_tensor])

    obj_mesh = trimesh.Trimesh(vertices=s["gt_ho"].obj_verts, faces=s["gt_ho"].obj_faces)  # obj
    hand_mesh = trimesh.Trimesh(
        vertices=mano_verts.to("cpu").squeeze(dim=0).numpy(), faces=HandObject.closed_faces
    )  # hand

    """cluster1"""
    # kps = mano_joints.clone()
    # for count, kps_i in enumerate(kps):
    #     cluster.append(kps_i.detach().reshape(-1).cpu().numpy())

    """cluster2"""
    hand_kps = mano_joints.clone()
    is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

    hand_kps = convert_joints(hand_kps, source="mano", target="biomech")

    hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
    # hand_kps_after = convert_joints(hand_kps_after, source="biomech", target="mano")

    # for count, kps_flat in enumerate(hand_kps_after):
    #     cluster2.append(kps_flat.detach().reshape(-1).cpu().numpy())

    # penetration volume
    intersection_volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    # contact
    penetration_tol = 0.005
    result_close, result_distance, _ = trimesh.proximity.closest_point(
        obj_mesh, mano_verts.to("cpu").squeeze(dim=0).numpy()
    )
    sign = mesh_vert_int_exts(obj_mesh, mano_verts.to("cpu").squeeze(dim=0).numpy())
    nonzero = result_distance > penetration_tol
    exterior = [sign == -1][0] & nonzero
    contact = ~exterior
    sample_contact = contact.sum() > 0

    # simulation displacement
    # install the V-HACD for building the simulation of grasp displacement.
    # https://github.com/kmammou/v-hacd
    # you need change you path in here and here (must absolute path)
    vhacd_exe = "/home/zxc417/Projects/Toolboxes/v-hacd/app/build/TestVHACD"
    ## Run in parallel in batches
    # param_dict_list.append(
    #     {"hand_verts": mano_verts.to("cpu").squeeze(dim=0).numpy(), "hand_faces": rh_mano.th_faces.cpu().numpy(),
    #      "obj_verts": s["gt_ho"].obj_verts, "obj_faces": s["gt_ho"].obj_faces, 'vhacd_exe': vhacd_exe,
    #      "indicator": f"{method}_{idx}", 'save_video': False
    #      })
    #
    client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    simulation_displacement = run_simulation(
        mano_verts.to("cpu").squeeze(dim=0).numpy(),
        rh_mano.th_faces.cpu().numpy(),
        s["gt_ho"].obj_verts,
        s["gt_ho"].obj_faces,
        client=client,
        vhacd_exe=vhacd_exe,
        indicator=f"{method}_{idx}",
        save_video=False,
        save_video_path="logs/simulations/",
    )
    client.disconnect()
    return intersection_volume, sample_contact, simulation_displacement


def run_metrics(args, rh_mano, device):
    # in_file = "data/optimized_{}.pkl".format(args.split)
    # in_file = 'data/qp_optimized_im.pkl'
    methods = ['opt']
    runs = {}
    # args.partial = 200
    if 'opt' in methods:
        in_file_opt = 'data/opt_im.pkl'
        runs['opt'] = pickle.load(open(in_file_opt, "rb"))
        print("Loaded {} len {}".format(in_file_opt, len(runs['opt'])))
        if args.partial > 0:
            runs['opt'] = runs['opt'][: args.partial]
    if 'phy' in methods:
        in_file_phy = 'data/phy_im.pkl'
        runs['phy'] = pickle.load(open(in_file_phy, "rb"))
        print("Loaded {} len {}".format(in_file_phy, len(runs['phy'])))
        if args.partial > 0:
            runs['phy'] = runs['phy'][: args.partial]

    metrics = {}
    # if args.vis or args.physics:
    #     print('Shuffling!!!')
    #     random.shuffle(runs)

    all_data = []  # Do non-parallel

    cluster = []
    cluster2 = []
    batch_size = 20
    prev_idx = 0

    # childPipes = []
    # parentPipes = []
    # for pr in range(batch_size):
    #     parentPipe, childPipe = Pipe()
    #     parentPipes.append(parentPipe)
    #     childPipes.append(childPipe)
    # pb_clients = [bc.BulletClient(connection_mode=pybullet.DIRECT) for i in range(batch_size)]

    pool = Pool(batch_size)
    for method in methods:
        simulation_displacements_list = []  # 获取 std
        intersection_volumes_list = []
        sample_contact_list = []

        param_dict_list = []
        for idx, s in enumerate(tqdm(runs[method])):
            param_dict_list.append({'method': method, 's': s, 'idx': idx, 'rh_mano': rh_mano, 'device': device})
            results = []
            processes = []
            if idx % batch_size == batch_size - 1 or idx == len(runs) - 1:
                results = pool.map(metrics_parallel_interface, param_dict_list)
                # for params in param_dict_list:
                #     results.append(metrics_parallel_interface(params))
                # simulation_displacement = pool.map(run_sim_parallel_interface, param_dict_list)
                param_dict_list = []
                iv = [r[0] for r in results]
                sc = [r[1] for r in results]
                sd = [r[2] for r in results]

                intersection_volumes_list.extend(iv)
                sample_contact_list.extend(sc)
                simulation_displacements_list.extend(sd)

        # cluster_array = np.array(cluster)
        # entropy, cluster_size = diversity(cluster_array, cls_num=20)
        #
        # cluster_array_2 = np.array(cluster2)
        # entropy_2, cluster_size_2 = diversity(cluster_array_2, cls_num=20)

        std_simulation_displacement = statistics.stdev(simulation_displacements_list)

        mean_simulation_displacement = sum(simulation_displacements_list) / len(simulation_displacements_list)

        mean_intersection_volume = sum(intersection_volumes_list) / len(intersection_volumes_list)

        print(f"{method} metrics:"
            f"mean_simulation_displacement : {mean_simulation_displacement * 1e2:.4f}e-02\n"
            f"std_simulation_displacement : {std_simulation_displacement * 1e2:.4f}e-02\n"
            f"mean_intersection_volume : {mean_intersection_volume * 1e6:.4f}e-06\n"
            f"contact_ratio : {np.mean(sample_contact_list) * 1e2 :.4f}e-02\n"
            # f"entropy :, {entropy} \n"
            # f"cluster_size : {cluster_size}\n"
            # f"entropy2 : {entropy_2} \n",
            # f"cluster_size2 : {cluster_size_2} \n",
        )
        metrics[method] = {
            'Mean simulation displacement': mean_simulation_displacement.item(),
            'Std simulation displacement': std_simulation_displacement,
            'Mean intersection volume': mean_intersection_volume.item(),
            'Contact ratio': np.mean(sample_contact_list).item(),
        }

    for m, rs in runs.items():
        all_data = []
        for idx, s in enumerate(rs):
            all_data.append(process_sample(s, idx))
        in_all = [item[0] for item in all_data]
        if m == 'opt':
            opt_all = [item[1] for item in all_data]
        elif m == 'phy':
            phy_all = [item[2] for item in all_data]

    mean_in = calc_mean_dicts(in_all, "In vs GT")
    print("In vs GT\n", pprint.pformat(mean_in))
    if 'opt' in methods:
        mean_opt = calc_mean_dicts(opt_all, "Opt vs GT")
        print("Opt vs GT\n", pprint.pformat(mean_opt))
        for k, v in mean_opt.items():
            metrics['opt'][k] = v
    if 'phy' in methods:
        mean_phy = calc_mean_dicts(phy_all, "Phy vs GT")
        print("Phy vs GT\n", pprint.pformat(mean_phy))
        for k, v in mean_phy.items():
            metrics['phy'][k] = v
    print("all saved !!!")
    with open(os.path.join('logs', 'simulations', 'metrics', f'{"+".join(methods)}.yaml'), 'w') as yf:
        yaml.dump(metrics, yf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eval on fitted pkl")
    parser.add_argument("--split", default="aug", type=str)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--contact_f1", action="store_true")
    parser.add_argument("--pen", action="store_true")
    parser.add_argument("--saveobj", action="store_true")
    parser.add_argument("--partial", default=-1, type=int, help="Only run for n samples")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)

    rh_mano = ManoLayer(mano_root="data/mano_v1_2/models", use_pca=True, ncomps=15, side="right", flat_hand_mean=False).to(device)

    start_time = time.time()
    run_metrics(args, rh_mano, device)
    print("Eval time", time.time() - start_time)
