## Where the physics-aware encoder and decoder are defined.
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from prev_sota.contactgen.base_net import Pointnet, LetentEncoder


class PointFeatEnc(nn.Module):
    ''' PointNet-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, in_dim=3):
        hidden_dim = 64
        latent_dim = 64
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.conv4 = torch.nn.Conv1d(hidden_dim, hidden_dim*2, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim*2)
        self.bn5 = nn.BatchNorm1d(latent_dim)
        self.fc1 = nn.Linear(hidden_dim*2 + 3, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        self.actvn = nn.ReLU()

    def forward(self, x, global_vec):
        ## global_vec: b x 3
        x = x.permute(0, 2, 1) # b x c x 2048
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.permute(0, 2, 1)
        x = F.max_pool2d(x, kernel_size=(x.shape[1], 1))
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, global_vec], 1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)

        return x


class PointFeatDec(nn.Module):
    ''' PointNet-based decoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, latent_dim=3, encode_dim=64, out_dim=3):
        super().__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(latent_dim + 3, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim*2)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim*2)
        self.conv1 = torch.nn.Conv1d(latent_dim*2 + encode_dim, hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim*2, 1)
        self.conv4 = torch.nn.Conv1d(hidden_dim*2, out_dim, 1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim*2)

    def forward(self, x, global_vec, pt_cond):
        ## global_vec: b x 3; x: b x n; pt_cond: b x n_pt x n_feat
        # x = x.permute(0, 2, 1) # b x c x 2048
        x = F.relu(self.bn1(self.fc1(torch.cat([x, global_vec], 1))))
        x = F.relu(self.bn2(self.fc2(x)))
        x = x.unsqueeze(1).repeat(1, pt_cond.shape[1], 1) # unpooling
        x = torch.cat([x, pt_cond], -1).permute(0, 2, 1)
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = F.relu(self.bn5(self.conv3(x)))
        x = self.conv4(x)
        x = x.permute(0, 2, 1)

        return x



class PhysAwareEncoder(nn.Module):
    def __init__(self, cfg):
        super(PhysAwareEncoder, self).__init__()
        self.cfg = cfg
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature

        self.num_parts = 16

        encode_dim = self.hc

        self.contact_encoder = PointFeatEnc(in_dim=encode_dim + 1)
        self.part_encoder = PointFeatEnc(in_dim=encode_dim + self.latentD + self.hc)
        self.pressure_encoder = PointFeatEnc(in_dim=encode_dim + self.hc + 16)

        self.contact_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.part_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.pressure_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)

    def forward(self, embed_class, obj_cond, contacts_object, partition_object, pressure_object, gravity_direction):
        contact_latent = self.contact_encoder(torch.cat([obj_cond, contacts_object], -1), gravity_direction)
        contact_mu, contact_std = self.contact_latent(contact_latent)
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))
        z_s_contact = z_contact.rsample()

        partition_feat = embed_class(partition_object.argmax(dim=-1))
        part_latent = self.part_encoder(
            torch.cat([obj_cond, z_s_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1), partition_feat], -1), gravity_direction)
        part_mu, part_std = self.part_latent(part_latent)
        z_part = torch.distributions.normal.Normal(part_mu, torch.exp(part_std))

        ## Include gravity direction to the encoding & decoding
        pressure_latent = self.pressure_encoder(torch.cat([obj_cond, partition_feat, pressure_object], -1), gravity_direction)
        pressure_mu, pressure_std = self.pressure_latent(pressure_latent)
        z_pressure = torch.distributions.normal.Normal(pressure_mu, torch.exp(pressure_std))
        z_s_part = z_part.rsample()
        z_s_pressure = z_pressure.rsample()

        return z_contact, z_part, z_pressure, z_s_contact, z_s_part, z_s_pressure


class PhysAwareDecoder(nn.Module):
    def __init__(self, cfg):
        super(PhysAwareDecoder, self).__init__()
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature
        self.num_parts = 16
        encode_dim = self.hc

        self.contact_decoder = PointFeatDec(latent_dim=self.latentD, encode_dim=self.hc, out_dim=1)
        self.part_decoder = PointFeatDec(latent_dim=self.latentD + self.latentD, encode_dim=self.hc,
                                     out_dim=self.num_parts)
        self.pressure_decoder = PointFeatDec(latent_dim=3*self.latentD, encode_dim=self.hc*2, out_dim=16)

    def forward(self, embed_class, z_contact, z_part, z_pressure, obj_cond, gt_partition_object=None, gravity_direction=None):

        # z_contact = z_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object = self.contact_decoder(z_contact, gravity_direction, obj_cond)
        contacts_object = torch.sigmoid(contacts_object)

        # z_part = z_part.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        partition_object = self.part_decoder(torch.cat([z_part, z_contact], -1), gravity_direction, obj_cond)
        # z_pressure = z_pressure.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)

        if gt_partition_object is not None:
            partition_feat = embed_class(gt_partition_object.argmax(dim=-1))
        else:
            partition_object_ = F.one_hot(partition_object.detach().argmax(dim=-1), num_classes=16)
            partition_feat = embed_class(partition_object_.argmax(dim=-1))
        pressure_object = self.pressure_decoder(torch.cat([z_pressure, z_part, z_contact], -1), gravity_direction,
                                                torch.cat([partition_feat, obj_cond], dim=-1))
        # pressure_object = normalize_vector(pressure_object)
        return contacts_object, partition_object, pressure_object


class ObjPhysFeatureNet(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=3):
        super(ObjPhysFeatureNet, self).__init__()
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
        # l3_points = torch.cat((l3_points, obj_mass.view(-1, 16, 1)), dim=1)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # B x 256 x 128
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # B x 128 x 512
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points) # B x 64 x N
        feat = F.relu(self.bn1(self.conv1(l0_points))) # B x 64 x N
        return feat.permute(0, 2, 1)


# def obj_phys_feat_net(cfg):
#     obj_pointnet = Pointnet2(in_dim=cfg.model.obj_feature, hidden_dim=cfg.model.pointnet_hc, out_dim=cfg.model.pointnet_hc)
#     return obj_pointnet