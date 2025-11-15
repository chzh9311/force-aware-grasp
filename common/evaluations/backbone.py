import torch
import lightning as L
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from common.utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from common.utils.manolayer import ManoLayer
from common.utils.vis import o3dmesh
from common.model.losses import axis_angle_loss

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix, matrix_to_axis_angle

# class HandFeaturePNEncoder(L.LightningModule):
#     def __init__(self, metric_cfg):
#         super().__init__()
#         self.model = PoinetNetCLSMSG(metric_cfg)
#
#     def forward(self, x):
#

## From https://github.com/jyunlee/InterHandGen
class TwoHandFeaturePNEncoder(L.LightningModule):
    def __init__(self,num_class=42*3,normal_channel=True):
        super(TwoHandFeaturePNEncoder, self).__init__()
        normal_channel = True

        in_channel = 2 if normal_channel else 0

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 2048], True)
        self.fc1 = nn.Linear(2048, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, return_feat=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        act = x = self.bn2(self.fc2(x))
        x = self.drop2(F.relu(x))
        x = self.fc3(x)

        if return_feat:
            return x, l3_points, act
        else:
            return x,l3_points


## From https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_ssg.py
class PointNetCLSMSG(L.LightningModule):
    def __init__(self, eval_cfg, normal_channel=True):
        super(PointNetCLSMSG, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, eval_cfg.num_class)

        self.mano_layers = ManoLayer('data/mano_v1_2')
        self.n_sampled_pts = eval_cfg.n_sampled_pts
        self.lr = eval_cfg.train.lr
        self.optim_step = eval_cfg.train.optim_step
        self.optim_gamma = eval_cfg.train.gamma
        self.save_hyperparameters()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        feat = self.bn2(self.fc2(x))
        x = self.drop2(F.relu(feat))
        x = self.fc3(x)

        return x, l3_points, feat

    def training_step(self, batch, batch_idx):
        loss = self.batch_test(batch)
        self.log('Train loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.batch_test(batch)

        self.log('Val rotation Error', loss)
        return loss

    def batch_test(self, batch):
        """
        Fit the model to a simple pose regression task.
        """
        lpose = batch['left_hand']['pose']
        lrot_aa = batch['left_hand']['rot_aa']
        rpose = batch['right_hand']['pose']
        rrot_aa = batch['right_hand']['rot_aa']
        ## flip at x=0 plane:
        mirror_T = torch.eye(3, device=self.device)
        mirror_T[0, 0] = -1
        l2rrot_aa = matrix_to_axis_angle(mirror_T.unsqueeze(0) @ axis_angle_to_matrix(lrot_aa))
        right_params = torch.cat((rpose, rrot_aa), dim=-1) # 54
        l2r_params = torch.cat((lpose, l2rrot_aa), dim=-1)
        params = torch.cat((right_params, l2r_params), dim=0)

        ## calculate hand mesh:
        ## TODO: do transformations & training.
        lv = batch['left_hand']['vertices']
        rv = batch['right_hand']['vertices'].detach().cpu().numpy()
        l2rv = (lv @ mirror_T.unsqueeze(0)).detach().cpu().numpy()

        ## Do mesh sampling
        samples = []
        batch_size = lpose.shape[0]
        for i in range(batch_size):
            rmesh = o3dmesh(rv[i], self.mano_layers.mano_f['right'])
            samples.append(torch.from_numpy(np.array(rmesh.sample_points_uniformly(self.n_sampled_pts).points)))
        for i in range(batch_size):
            l2rmesh = o3dmesh(l2rv[i], self.mano_layers.mano_f['left'])
            samples.append(torch.tensor(np.array(l2rmesh.sample_points_uniformly(self.n_sampled_pts).points)))

        sampled_pts = torch.stack(samples, dim=0).to(self.device).float()
        sampled_pts = self.pc_normalize(sampled_pts).permute(0, 2, 1)

        pred_pose, _, _ = self.forward(sampled_pts)

        ## Use angle_loss to supervise this.
        # loss = F.mse_loss(pred_pose, params)
        loss = axis_angle_loss(pred_pose.view(-1, 3), params.view(-1, 3))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.optim_step, gamma=self.optim_gamma)
        return [optimizer], [scheduler]

    def pc_normalize(self, input_pc: torch.Tensor):
        """
        :param input_pc: batch_size x N x 3
        the normalization module following
        https://github.com/erikwijmans/Pointnet2_PyTorch/blob/b5ceb6d9ca0467ea34beb81023f96ee82228f626/pointnet2/data/ModelNet40Loader.py#L17
        """
        mean = torch.mean(input_pc, dim=1, keepdim=True)
        input_pc = input_pc - mean
        ## normalize to unit sphere
        m = torch.max(torch.norm(input_pc, dim=2, keepdim=True), dim=1, keepdim=True)[0]
        input_pc = input_pc / m
        return input_pc



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
