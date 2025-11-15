import torch
import torch.nn.functional as F
from common.utils.geometry import get_perpend_vecs_tensor
from pytorch3d.transforms import axis_angle_to_matrix

def axis_angle_loss(rot1, rot2):
    """
    The relative loss between two rotation vectors in axis-angle form.
    loss = ||R1 @ R2.T - I||^2_2
    :param rot1, rot2: (batch_size, 3)
    """
    R1 = axis_angle_to_matrix(rot1)
    R2 = axis_angle_to_matrix(rot2)
    losses = torch.sum(torch.square(R1 @ R2.transpose(1, 2)
                     - torch.eye(3, device=rot1.device).unsqueeze(0)).view(-1, 9), dim=-1)
    loss = torch.mean(losses)

    return loss


def calc_stable_loss(obj_verts: torch.Tensor, obj_normal: torch.Tensor, masked_pressure: torch.Tensor, gravity_direction:torch.Tensor):
    """
    Regularizes the VAE output so that the predicted forces are capable of keeping the object stable.
    The center of mass is in (0, 0, 0), the mass of the object is 1kg
    obj_verts: (batch_size, num_sampled_pts, 3)
    obj_normal: (batch_size, num_sampled_pts, 3)
    masked_pressure: (batch_size, num_sampled_pressures)
    gravity_direction: (batch_size, 3)
    """
    N = - obj_normal
    B, T = get_perpend_vecs_tensor(obj_normal, device=obj_verts.device)
    Fub, Flb = calc_ub_lb(N, B, T, mu=1)

    ## Calculate the torque stability
    ls = obj_verts
    radius = torch.max(torch.norm(ls, dim=-1))
    J = 0.4 * radius ** 2
    Np = torch.cross(ls, N, dim=-1) / J
    Bp = torch.cross(ls, B, dim=-1) / J
    Tp = torch.cross(ls, T, dim=-1) / J
    Mub, Mlb = calc_ub_lb(Np, Bp, Tp, mu=1)

    Aub = torch.cat((Fub, Mub), dim=-1)
    Alb = torch.cat((Flb, Mlb), dim=-1)

    gravity = gravity_direction * 9.81
    acc = torch.zeros((obj_verts.shape[0], 6, 1), device=obj_verts.device).float()
    acc[:, :3] = gravity.view(-1, 3, 1)
    up_residual = F.relu(-(acc + Aub.transpose(-1, -2) @ masked_pressure.unsqueeze(-1))).squeeze(-1)
    low_residual = F.relu(acc + Alb.transpose(-1, -2) @ masked_pressure.unsqueeze(-1)).squeeze(-1)

    loss = torch.log(torch.sum(up_residual + low_residual, dim=-1) + 1)
    return loss


def part_cluster_loss(obj_verts: torch.Tensor, soft_part_mask: torch.Tensor):
    """
    Regularise the part mask to avoid it from spreading all over the object
    """
    avg = torch.sum(obj_verts.unsqueeze(2) * soft_part_mask.unsqueeze(-1), dim=1) / torch.sum(
            soft_part_mask.unsqueeze(-1), dim=1)  # B x 16 x 3
    var = torch.sum(soft_part_mask * torch.norm(obj_verts.unsqueeze(2) - avg.unsqueeze(1), dim=-1), dim=1) / (torch.sum(soft_part_mask, dim=1) + 1e-8)
    return torch.mean(var, dim=-1)

def calc_aggregated_stable_loss(obj_verts: torch.Tensor, obj_normal: torch.Tensor, soft_part_mask: torch.Tensor,
                     pressure: torch.Tensor, gravity_direction: torch.Tensor):
    """
    Regularizes the VAE output so that the predicted forces are capable of keeping the object stable.
    The center of mass is in (0, 0, 0), the mass of the object is 1kg
    obj_verts: (batch_size, num_sampled_pts, 3)
    obj_normal: (batch_size, num_sampled_pts, 3)
    soft_part_mask: (batch_size, num_sampled_pts, 16) represent the probability of each part in contact with each point.
    pressure: (batch_size, num_sampled_pressures) the real value of the predicted pressure.
    gravity_direction: (batch_size, 3)
    """
    N = - torch.sum(obj_normal.unsqueeze(2) * soft_part_mask.unsqueeze(-1), dim=1)  # b x 16 x 3
    N = N / (torch.norm(N, dim=-1, keepdim=True) + 1e-8)
    B, T = get_perpend_vecs_tensor(N, device=obj_verts.device)
    Fub, Flb = calc_ub_lb(N, B, T, mu=1)

    ## Calculate the torque stability
    ls = torch.sum(obj_verts.unsqueeze(2) * soft_part_mask.unsqueeze(-1), dim=1) / torch.sum(
        soft_part_mask.unsqueeze(-1), dim=1)  # B x 16 x 3
    radius = torch.max(torch.norm(obj_verts, dim=-1))
    J = 0.4 * radius ** 2
    Np = torch.cross(ls, N, dim=-1) / J
    Bp = torch.cross(ls, B, dim=-1) / J
    Tp = torch.cross(ls, T, dim=-1) / J
    Mub, Mlb = calc_ub_lb(Np, Bp, Tp, mu=1)

    Aub = torch.cat((Fub, Mub), dim=-1)
    Alb = torch.cat((Flb, Mlb), dim=-1)

    ## Aggregate the pressure
    pressure = torch.sum(soft_part_mask * pressure.unsqueeze(-1), dim=1)  # B x 16

    gravity = gravity_direction * 9.81
    acc = torch.zeros((obj_verts.shape[0], 6, 1), device=obj_verts.device).float()
    acc[:, :3] = gravity.view(-1, 3, 1)
    up_residual = F.relu(-(acc + Aub.transpose(-1, -2) @ pressure.unsqueeze(-1))).squeeze(-1)
    low_residual = F.relu(acc + Alb.transpose(-1, -2) @ pressure.unsqueeze(-1)).squeeze(-1)

    loss = torch.log(torch.sum(up_residual + low_residual, dim=-1) + 1)
    return loss


def calc_ub_lb(N: torch.Tensor, B: torch.Tensor, T: torch.Tensor, mu: float, sumbt=False):
    """
    Calculate the upper & lower bound of the Newton's Law II equations
    """
    quantB = torch.zeros_like(B, device=B.device)
    quantB[B > 0] = mu
    quantB[B < 0] = -mu

    quantT = torch.zeros_like(T, device=T.device)
    quantT[T > 0] = mu
    quantT[T < 0] = -mu

    BqB = B * quantB
    TqT = T * quantT
    if sumbt:
        BqB = torch.sum(BqB, dim=-1)
        TqT = torch.sum(TqT, dim=-1)

    ub = N + BqB + TqT
    lb = N - BqB - TqT
    return ub, lb
