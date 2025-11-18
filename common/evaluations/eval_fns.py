import numpy as np
import torch
from pysdf import SDF
import scipy
from scipy.stats import entropy

from common.evaluations.bullet_simulation import run_simulation
from common.utils.converter import transform_to_canonical, convert_joints
from common.model.force_labelling_trainer import get_force_from_simulation

import pybullet
import pybullet_utils.bullet_client as bc


def diversity(params_list, cls_num=20):
    # k-means
    params_list = scipy.cluster.vq.whiten(params_list)
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences  count: [20]
    ee = entropy(counts)
    return ee, np.mean(dist)


def get_force(pressure, verts):
    pressure_val, pressure_pos = torch.max(pressure, dim=1)
    pressure_3d_pos = torch.stack([verts[b][pressure_pos[b]] for b in range(pressure_pos.shape[0])], dim=0)
    return pressure_3d_pos, pressure_val


def pos_pressure_error(pr1: torch.Tensor, pr2: torch.Tensor, verts: torch.Tensor):
    """
    pr1/2: <Batch x n_points x n_parts>;
    verts: <Batch x n_points x 3>;
    error = (pos1 x value1 - pos2 x value2)
    """
    pos1, value1 = get_force(pr1, verts)
    pos2, value2 = get_force(pr2, verts)
    ## Scale to mm
    return torch.mean(torch.norm(pos1 * value1.unsqueeze(-1) - pos2 * value2.unsqueeze(-1), dim=-1)) * 1000


def pressure_value_error(cmask, gt_pressure, pred_pressure):
    """
    cmask: <Batch x n_points>;
    gt_pressure: <Batch x n_points>;
    pred_pressure: <Batch x n_points>;
    """
    return torch.abs(gt_pressure - pred_pressure)[cmask].mean()


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



def calc_diversity(hand_joints):
    cluster = []
    cluster2 = []
    kps = hand_joints.copy()
    for count, kps_i in enumerate(kps):
        cluster.append(kps_i.flatten())

    """cluster2"""
    hand_kps = torch.as_tensor(kps.copy()).float()
    is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

    hand_kps = convert_joints(hand_kps, source="mano", target="biomech")

    hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
    hand_kps_after = convert_joints(hand_kps_after, source="biomech", target="mano")

    for count, kps_flat in enumerate(hand_kps_after):
        cluster2.append(kps_flat.detach().reshape(-1).cpu().numpy())

    cluster_array = np.array(cluster)
    entropy, cluster_size = diversity(cluster_array, cls_num=20)

    cluster_array_2 = np.array(cluster2)
    entropy_2, cluster_size_2 = diversity(cluster_array_2, cls_num=20)

    return entropy, cluster_size, entropy_2, cluster_size_2


def parallel_calculate_metrics(params:dict):
    metrics = params['metrics']
    result = get_force_from_simulation(params)
    if "PyBullet SimuDisp" in metrics:
        pb_disp = pybullet_parallel_interface(params) * 100
        result["PyBullet SimuDisp"] = pb_disp
        if "Pybullet Stable Rate" in metrics:
            result["Pybullet Stable Rate"] = pb_disp < 2
    if "Intersection Volume" in metrics:
        int_vol = intersect_vox(params['obj_model'], params['hand_model'], pitch=0.005) * 1000000 # turn to cm3
        result["Intersection Volume"] = int_vol
    if "Contact Ratio" in metrics:
        penetration_tol = 0.005
        hand_verts = params['hand_model'].vertices
        obj_sdf = SDF(params['obj_model'].vertices, params['obj_model'].faces)
        hv_sds = obj_sdf(hand_verts)

        contact = hv_sds > - penetration_tol
        sample_contact = contact.sum() > 0
        result["Contact Ratio"] = sample_contact

    return result


def pybullet_parallel_interface(params:dict):
    client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    hand_verts, hand_faces, obj_verts, obj_faces, fid, obj_hulls = (
        params['hand_model'].vertices, params['hand_model'].faces, params['obj_model'].vertices,
        params['obj_model'].faces, params['idx'], params['obj_hulls'])
    disp = run_simulation(hand_verts, hand_faces, obj_verts, obj_faces, indicator=fid, client=client, obj_hulls=obj_hulls, save_video=False)
    return disp


def calculate_metrics(param_list, pool=None, metrics=[], reduction='mean'):
    for p in param_list:
        p["metrics"] = metrics

    if pool is not None:
        result_list = pool.map(parallel_calculate_metrics, param_list)
    else:
        result_list = [parallel_calculate_metrics(p) for p in param_list]
    result = {}
    for k in result_list[0].keys():
        result[k] = [rit[k] for rit in result_list]
    for m in metrics:
        if reduction == 'mean':
            result[m] = np.mean(np.asarray(result[m])).item()
        elif reduction == 'sum':
            result[m] = np.sum(np.asarray(result[m])).item()
    return result