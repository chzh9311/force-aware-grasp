import torch
import trimesh
import numpy as np
from pysdf import SDF
from easydict import EasyDict as edict
from pytorch3d.structures import Meshes

def rel_tf(rot1, trans1, rot2, trans2):
    """
    The relative transformation of 1 based on 2.
    rot<1,2> are axis-angle format.
    trans<1,2> are Euclidean translations.
    """
    trans = trans1 - trans2

def calc_num_gaps(traj_len, gap, max_len, min_len):
    if traj_len < min_len * gap:
        gap = traj_len // min_len
    elif traj_len > max_len * gap:
        gap = (traj_len + max_len - 1) // max_len
    return gap

def sample_slice_from_gap(traj: np.ndarray, gap: int, max_len: int, min_len: int):
    res = []
    res_len = []
    # process
    # the first dimension is to slice!
    traj_len = int(traj.shape[0])
    # determine mode: using gap, using max_len or using min_len
    gap = calc_num_gaps(traj_len, gap, max_len, min_len)
    # slice
    for _offset in range(gap):
        sliced_traj = traj[_offset::gap]
        sliced_traj_len = sliced_traj.shape[0]
        assert min_len <= sliced_traj_len <= max_len
        # pad to max_len
        if sliced_traj_len < max_len:
            pad_len = max_len - sliced_traj_len
            pad = np.zeros((pad_len, *sliced_traj.shape[1:]), dtype=sliced_traj.dtype)
            sliced_traj = np.concatenate([sliced_traj, pad], axis=0)
        res.append(sliced_traj)
        res_len.append(sliced_traj_len)
    return res, res_len


def iter_merge_dict(dict1, dict2):
    for k, v in dict2.items():
        if type(v) is dict:
            dict1[k] = iter_merge_dict(dict1[k], v)
        else:
            dict1[k] += dict2[k]

    return dict1


def mano_array2dict(mano_array: (np.ndarray, torch.Tensor), rot_type:str) -> dict:
    if rot_type == 'aa':
        h_dict = {'pose': mano_array[:, :45], 'shape': mano_array[:, 45:55],
                  'rot_aa': mano_array[:, 55:58], 'trans': mano_array[:, 58:]}
    elif rot_type == '6d':
        h_dict = {'pose': mano_array[:, :45], 'shape': mano_array[:, 45:55],
                  'rot_6d': mano_array[:, 55:61], 'trans': mano_array[:, 61:]}
    else:
        raise ValueError(f'Unknown rotation type: {rot_type}.')
    return h_dict


def mano_dict2array(mano_dict: dict, rot_type) -> (np.ndarray, torch.Tensor):
    if type(mano_dict['pose']) is np.ndarray:
        return np.concatenate((mano_dict['pose'], mano_dict['shape'], mano_dict['rot_' + rot_type], mano_dict['trans']),axis=1)
    else:
        return torch.cat((mano_dict['pose'], mano_dict['shape'], mano_dict['rot_' + rot_type], mano_dict['trans']),dim=1)


def update_config(dict1: dict, dict2: dict) -> dict:
    for k, v in dict2.items():
        if type(v) is dict and k in dict1:
            dict1[k] = update_config(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def linear_normalize(x: torch.Tensor, lower_bound: float, upper_bound:float, lower_th:float|None=None, upper_th:float|None=None) -> torch.Tensor:
    """
    linearly map the variable x to the range of (lower_bound, upper_bound).
    :param x: 1-D tensor
    :param th: cutting edges of upper & lower limits.
    """
    if lower_th is not None:
        lower_th = lower_bound + lower_th * (upper_bound - lower_bound)
        x[x < lower_th] = lower_bound
    if upper_th is not None:
        upper_th = lower_bound + upper_th * (upper_bound - lower_bound)
        x[x > upper_th] = upper_bound

    x = (x - x.min()) / (x.max() - x.min() + 1.0e-8)
    x = lower_bound + x * (upper_bound - lower_bound)

    return x


def update_list_dict(base_lod: list, new_lod:list, idx: int) -> list:
    """
    lod: list of dicts where the values are batch_size x ...
    idx: the index of batch
    """
    output = []
    for i in range(len(base_lod)):
        for k, v in new_lod[i].items():
            base_lod[i][k][idx] = v[idx]

    return base_lod

def trimesh2Mesh(trimesh_list: list, device) -> Meshes:
    """
    trimesh_list: list of trimesh.Trimesh objects
    """
    verts = [torch.from_numpy(m.vertices).float().to(device) for m in trimesh_list]
    faces = [torch.from_numpy(m.faces).float().to(device) for m in trimesh_list]
    return Meshes(verts=verts, faces=faces)

def force_padding_collate_fn(data):
    # output = {'contact_frames': [], 'force_vecs': [], 'contact_pts': []}
    # for d in data:
    #     ## separate for each hand part.
    #     frames, force_vecs, contact_pts = [[]]*16, [[]]*16, [[]]*16
    #     for contacts in d['simuContacts']:
    #         idx = contacts['part_id']
    #         frames[idx].append(torch.tensor(contacts['frame']))
    #         force_vecs[idx].append(torch.tensor(contacts['force']))
    #         contact_pts[idx].append(torch.tensor(contacts['contact_pt']))
        # frames = pad_sequence([torch.stack(fr, dim=0) if len(fr) else torch.tensor(fr) for fr in frames])
        # force_vecs = pad_sequence([torch.stack(fv, dim=0) if len(fv) else torch.tensor(fv) for fv in force_vecs])
        # contact_pts = pad_sequence([torch.stack(cp, dim=0) if len(cp) else torch.tensor(cp) for cp in contact_pts])
        # output['contact_frames'].append(frames)
        # output['force_vecs'].append(force_vecs)
        # output['contact_pts'].append(contact_pts)
    output = {}

    for k in data[0].keys():
        if type(data[0][k]) in [np.ndarray, torch.Tensor]:
            output[k] = torch.stack([torch.as_tensor(d[k]).float() for d in data])
        elif type(data[0][k]) in [int, float]:
            output[k] = torch.as_tensor([d[k] for d in data])
        else:
            output[k] = [d[k] for d in data]

    return output
