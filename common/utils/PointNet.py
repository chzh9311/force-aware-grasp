import torch.nn as nn
import torch.nn.functional as F
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg


class ObjectPNEncoder(nn.Module):
    def __init__(self, feature_channels=3, fc_dims=None):
        super(ObjectPNEncoder, self).__init__()
        if fc_dims is None:
            fc_dims = [512, 512]
        in_channel = 3 + feature_channels
        self.feature_channels = feature_channels
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, fc_dims[0])
        self.bn1 = nn.BatchNorm1d(fc_dims[0])
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.feature_channels:
            feat = xyz[:, 3:, :] # The d-dimensional point feature.
            xyz = xyz[:, :3, :] # Point coordinate
        else:
            feat = None
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return x


