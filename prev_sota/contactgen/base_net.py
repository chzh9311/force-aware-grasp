import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.linalg import inverse_transformation

from prev_sota.contactgen.manopth import rodrigues_layer
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

def optimize_pose(model, mano_layer, obj_verts, obj_cmap, obj_partition, obj_uv,
                  w_contact=1e-1, w_pene=3.0, w_uv=1e-2, w_pose_reg=1e-2, w_shape_reg=1e-2,
                  global_iter=200, pose_iter=1000,
                  global_lr=5e-2, pose_lr=5e-3, eps=-1e-3):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    batch_size = obj_verts.shape[0]
    global_pose = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)

    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=obj_verts.dtype, device=obj_verts.device)

    mano_pose.requires_grad = False
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [global_pose, mano_trans]
    global_optimizer = torch.optim.Adam(hand_opt_params, lr=global_lr)

    for it in range(global_iter):
        loss_info = ""
        loss = 0

        _, _, frames = mano_layer(torch.cat((torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose), dim=1),
                               th_betas=mano_shape, th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1), global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)

        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())

        _, _, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=1.0+obj_cmap)
        loss += uv_loss
        loss_info += "uv loss: {:.3f}".format(uv_loss.item())

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        if it % 100 == 99:
            print("global iter {} | ".format(it) + loss_info)

    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [global_pose, mano_pose, mano_shape, mano_trans]
    pose_optimizer = torch.optim.Adam(hand_opt_params, lr=pose_lr)

    for it in range(pose_iter):
        loss_info = ""
        loss = 0
        _, _, frames = mano_layer(
            torch.cat((torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose),
                      dim=1),
            th_betas=mano_shape,
            th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1), global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)  # (B, Q)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())

        mask = pred_p_full < eps
        masked_value = pred_p_full[mask]
        if len(masked_value) > 0:
            loss_pene = w_pene * (-masked_value.sum()) / batch_size
            loss += loss_pene
            loss_info += "pene loss: {:.3f} | ".format(loss_pene.item())

        _, _, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=1+obj_cmap)
        loss += uv_loss
        loss_info += "uv loss: {:.3f} | ".format(uv_loss.item())

        pose_reg_loss = w_pose_reg * (mano_pose ** 2).sum() / batch_size
        loss += pose_reg_loss
        loss_info += "pose reg loss: {:.3f} | ".format(pose_reg_loss.item())

        shape_reg_loss = w_shape_reg * (mano_shape ** 2).sum() / batch_size
        loss += shape_reg_loss
        loss_info += "shape reg loss: {:.3f}".format(shape_reg_loss.item())

        pose_optimizer.zero_grad()
        loss.backward()
        pose_optimizer.step()
        if it % 300 == 299:
            print("iter {} | ".format(it) + loss_info)

    return global_pose, mano_pose, mano_shape, mano_trans


def normalize_vector(v):
    batch, n_points = v.shape[:2]
    v_mag = torch.norm(v, p=2, dim=-1)

    eps = torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v_mag.device))
    valid_mask = (v_mag > eps).float().view(batch, n_points, 1)
    backup = torch.tensor([1.0, 0.0, 0.0]).float().to(v.device).view(1, 1, 3).expand(batch, n_points, 3)
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, n_points, 1).expand(batch, n_points, v.shape[2])
    v = v / v_mag
    ret = v * valid_mask + backup * (1 - valid_mask)

    return ret

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, final_nl=False):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        x_out = x_s + dx
        if final_nl:
            return F.leaky_relu(x_out, negative_slope=0.2)
        return x_out


class Pointnet2(nn.Module):
    ''' PointNet++-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''

    def __init__(self, in_dim=6, hidden_dim=128, out_dim=3):
        super().__init__()
        assert in_dim == 6, "input (xyz, norm), channel=6"
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_dim,
                                          mlp=[hidden_dim, hidden_dim * 2], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=hidden_dim * 2 + 3,
                                          mlp=[hidden_dim * 2, hidden_dim * 4], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=hidden_dim * 4 + 3,
                                          mlp=[hidden_dim * 4, hidden_dim * 8], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=hidden_dim * 8 + hidden_dim * 4,
                                              mlp=[hidden_dim * 8, hidden_dim * 4])
        self.fp2 = PointNetFeaturePropagation(in_channel=hidden_dim * 4 + hidden_dim * 2,
                                              mlp=[hidden_dim * 4, hidden_dim * 2])
        self.fp1 = PointNetFeaturePropagation(in_channel=hidden_dim * 2 + in_dim,
                                              mlp=[hidden_dim * 2, hidden_dim])
        self.conv1 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :] # B x 3 x N
        l0_points = xyz[:, 3:, :] # B x (C-3) x N
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # xyz: B x 3 x 512; points: B x 128 x 512
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # xyz: B x 3 x 128; points: B x 256 x 128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # xyz: B x 3 x 1; points: B x 512 x 1

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # B x 256 x 128
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # B x 128 x 512
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points) # B x 64 x N
        feat = F.relu(self.bn1(self.conv1(l0_points))) # B x 64 x N
        return feat.permute(0, 2, 1)


class Pointnet(nn.Module):
    ''' PointNet-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(2 * hidden_dim, 4 * hidden_dim, 1)
        self.conv4 = torch.nn.Conv1d(4 * hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        ## global_vec: b x 3
        x = x.permute(0, 2, 1) # b x c x 2048
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.permute(0, 2, 1)

        return x, self.pool(x, dim=1)


class LetentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        self.fc_mean = nn.Linear(dim, out_dim)
        self.fc_logstd = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mean(x), self.fc_logstd(x)


def compute_uv(hand_frames, obj_verts, obj_parts):
    B, N, P = obj_verts.shape[0], obj_verts.shape[1], hand_frames.shape[1]
    inv_hand_frames = inverse_transformation(hand_frames.reshape(-1, 4, 4))
    part_label = F.one_hot(obj_parts.reshape(-1), num_classes=P).reshape(B, N, P).transpose(1, 2)
    obj_verts = obj_verts.unsqueeze(dim=1).expand(B, P, N, 3).reshape(-1, N, 3)
    local_verts = torch.bmm(obj_verts, inv_hand_frames[:, :3, :3].transpose(1, 2)) + inv_hand_frames[:, None, :3, 3]
    local_verts = local_verts.reshape(B, P, N, 3)
    local_verts = (part_label[:, :, :, None] * local_verts).sum(dim=1)
    uv_pred = local_verts / (torch.norm(local_verts, dim=2, keepdim=True) + 1e-10)
    return uv_pred

def compute_uv_loss(pred_uv, target_uv, weight=None):
    loss = 1 - torch.cosine_similarity(pred_uv, target_uv, dim=-1)
    if weight is not None:
        loss = loss * weight
    return loss.sum(dim=1).mean(dim=0)
