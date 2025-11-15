import numpy as np
import torch
import trimesh
import open3d as o3d
from pysdf import SDF

def compute_signed_distance_and_closest_goemetry(scene: o3d.t.geometry.RaycastingScene, query_points: np.ndarray):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points['primitive_ids'].numpy(), closest_points['primitive_normals'].numpy()

def get_perpend_vecs_tensor(vs: torch.Tensor, device='cpu') -> tuple:
    """
    :param vs: (batch_size x N x 3) a set of 3D vectors;
    :return: tuple of (batch_size x N x 3) -> the 2 perpendicular vector of the current vector
    if n = [a, b, c], then one perpendicular vector will be: [0, -c, b];
    By cross product, we can get the second vector as: [b^2+c^2, -ab, -ac]
    """
    n1 = torch.zeros_like(vs, device=device)
    n1[:, :, 1] = -vs[:, :, 2]
    n1[:, :, 2] = vs[:, :, 1]
    n2 = torch.cross(vs, n1, dim=-1)
    return n1, n2

def flip_x_axis(mesh):
    mesh.vertices[:, 0] *= -1
    mesh.faces[:, [0, 1]] = mesh.faces[:, [1, 0]]

def get_perpend_vecs(vs: np.ndarray) -> tuple:
    """
    :param vs: (N x 3) a set of 3D vectors;
    :return: tuple of (N x 3) -> the 2 perpendicular vector of the current vector
    if n = [a, b, c], then one perpendicular vector will be: [0, -c, b];
    By cross product, we can get the second vector as: [b^2+c^2, -ab, -ac]
    """
    n1 = np.zeros_like(vs)
    n1[..., 1] = -vs[..., 2]
    n1[..., 2] = vs[..., 1]
    n2 = np.cross(vs, n1, axis=-1)
    return n1, n2

def calc_contacts(object_pts: np.ndarray,
                  left_hand: (trimesh.Trimesh, None)=None,
                  right_hand: (trimesh.Trimesh, None)=None,
                  thresh: float=4e-2,
                  ) -> tuple:
    # pts = sample_surface_even(object, 1024)
    lh_sdf = SDF(left_hand.vertices, left_hand.faces)
    lh_sds = lh_sdf(object_pts)
    lh_nn = lh_sdf.nn(object_pts)
    lh_contact_map = np.clip(lh_sds / thresh + 1, 0, 1)
    rh_sdf = SDF(right_hand.vertices, right_hand.faces)
    rh_sds = rh_sdf(object_pts)
    rh_nn = rh_sdf.nn(object_pts)
    rh_contact_map = np.clip(rh_sds / thresh + 1, 0, 1)
    return lh_contact_map, lh_nn, rh_contact_map, rh_nn

def get_v2v_rot(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix from n1 to n2;
    """
    ax = np.cross(n1, n2)
    if np.linalg.norm(ax) < 1e-8:
        if n1[0] < 1e-8:
            ax[1] = n1[2]
            ax[2] = -n1[1]
        else:
            ax[0] = n1[1]
            ax[1] = -n1[0]
    ax = normalize_vec(ax)
    ang = np.arccos(np.dot(n1, n2) /
                    (np.linalg.norm(n1, axis=-1, keepdims=True) * np.linalg.norm(n2, axis=-1, keepdims=True)))
    R = rodrigues_rot(ax, ang)
    return R

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """
    vec: ... x N
    """
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)

def normalize_vec_tensor(vec: torch.Tensor) -> torch.Tensor:
    return vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)

def rodrigues_rot(axis, angle):
    """
    axis: ... x 3
    angle: ... x 1
    """
    angle = angle.reshape(angle.shape + (1,))
    axx = np.zeros(axis.shape+(3,)) # ... x 3 x 3
    axx[..., [2, 0, 1], [1, 2, 0]] = axis
    axx[..., [1, 2, 0], [2, 0, 1]] = -axis
    I = np.zeros_like(axx)
    I[..., [0, 1, 2], [0, 1, 2]] = 1
    R = I + np.sin(angle) * axx + (1 - np.cos(angle)) * axx @ axx
    return R

def get_contact_map(lc, rc, contact_th):
    contacts = lc + rc
    if contact_th is not None:
        ## binarize the contact
        contacts[contacts < contact_th] = 0
        contacts[contacts >= contact_th] = 1
    else:
        contacts = np.clip(contacts, 0, 1)
    return contacts

def get_seperate_contact_maps(lc, rc, contact_th):
    lc, rc = lc.copy(), rc.copy()
    if contact_th is not None:
        ## binarize the contact
        lc[lc < contact_th] = 0
        lc[lc >= contact_th] = 1
        rc[rc < contact_th] = 0
        rc[rc >= contact_th] = 1
    else:
        lc = np.clip(lc, 0, 1)
        rc = np.clip(rc, 0, 1)
    return lc, rc


def axisangle2matrix(rot_aa: np.ndarray) -> np.ndarray:
    """
    rot_aa: B x 3
    """
    angle = np.linalg.norm(rot_aa, axis=-1, keepdims=True)
    axis = rot_aa / angle
    return rodrigues_rot(axis, angle)


def quaternion2matrix(q:np.ndarray) -> np.ndarray:
    """
    q: N x 4 unit quaternion
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    angle = np.arccos(q[:, 0]) * 2
    axis = q[:, 1:] / (np.sin(angle / 2).reshape(-1, 1) + 1.0e-6)
    R = rodrigues_rot(axis, angle)
    return R


# @profile
def normal_matched_nn(mesh_verts, mesh_normals, queries, query_normals, mesh_trimesh=None, query_trimesh=None, signed_dist=False):
    """
    :param mesh_verts: (Batch_size x N x 3) The (sampled) mesh vertices
    :param mesh_normals: (Batch_size x N x 3) the vertex normals of the mesh
    :param queries: (Batch_size x M x 3) query points;
    :param query_normals: (Batch_size x M x 3) query normals
    """
    normal_coefs = torch.sum(mesh_normals.unsqueeze(1) * query_normals.unsqueeze(2), dim=-1) # B x M x N
    vectors = queries.unsqueeze(2) - mesh_verts.unsqueeze(1) # B x M x N x 3
    dists = torch.norm(vectors, dim=-1) # B x M x N
    # epsilon = 0.1 # in meter
    # nn_indicator = -normal_coefs / (dists + epsilon)
    normal_coefs[normal_coefs < 0] = 0
    nn_value, nn_idx = torch.min(dists + 0.1 * normal_coefs, dim=1) # B x N
    # for i in range(queries.shape[0]):
    #     dists[i, nn_idx[i], range(queries.shape[1])]
    bs, N = mesh_verts.shape[:2]
    nn_dists = torch.stack([dists[i, nn_idx[i], range(N)] for i in range(dists.shape[0])], dim=0) # B x N
    nn_pos = torch.stack([queries[i, nn_idx[i]] for i in range(bs)], dim=0)
    is_in_mesh_gt = []
    if signed_dist:
        ## The following calculation for sdf is slow
        # for i in range(bs):
        #     is_in_mesh_gt.append(mesh_trimesh[i].contains(nn_pos[i].detach().cpu().numpy()))
        #     nn_dists[i, is_in_mesh_gt[-1]] *= -1
        #     # is_in_query = query_trimesh[i].contains(mesh_verts[i].detach().cpu().numpy())
        #     # nn_vectors = torch.stack([vectors[i, nn_idx[i], range(N)] for i in range(dists.shape[0])]) # B x N x 3
        #     # distance_indicator = torch.sum(nn_vectors * mesh_normals)
        ## Use this as approximation
        nn_vectors = torch.stack([vectors[i, nn_idx[i], range(N)] for i in range(dists.shape[0])], dim=0) # B x N x 3
        is_in_mesh = torch.sum(nn_vectors * mesh_normals, dim=-1) < 0 # B x N
        nn_dists[is_in_mesh] *= -1
    return nn_dists, nn_idx, nn_pos


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2])-1 for l in spl[:3]])])

        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])

    for k, v in d.items():
        if k in ['v','vn','f','vt','ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


def cp_match(pts1: torch.Tensor, pts2: torch.Tensor, weight: torch.Tensor=None):
    """
    Do closest point matching for point clouds pts1 and pts2.
    The points are in correspondence according to the order.
    """
    if weight is None:
        weight = torch.ones(*pts1.shape[:-1], 1)
    mean1 = torch.sum(pts1 * weight, dim=1, keepdim=True) / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
    pts1 = pts1 - mean1
    mean2 = torch.sum(pts2 * weight, dim=1, keepdim=True) / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
    pts2 = pts2 - mean2
    W = torch.sum((pts2.unsqueeze(-1) @ pts1.unsqueeze(-2)) * weight.unsqueeze(-1), dim=1)
    U, S, Vh = torch.linalg.svd(W)
    ## Because the two point sets are of the same plane, then the third dimension of singular value could be ignored
    R = U @ Vh
    I = torch.eye(3, device=pts1.device).view(1, 3, 3).repeat(pts1.shape[0], 1, 1).float()
    I[:, 2, 2] = torch.det(R)
    R = U @ I @ Vh
    t = (mean2.transpose(-1, -2) - R @ mean1.transpose(-1, -2)).view(-1, 3)
    return R, t