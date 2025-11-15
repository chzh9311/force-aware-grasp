import torch
from torch import nn
from torch.nn import functional as F
from prev_sota.contactgen.base_net import Pointnet2, Pointnet, LetentEncoder


class TriEncoder(nn.Module):
    def __init__(self, cfg):
        super(TriEncoder, self).__init__()
        self.cfg = cfg
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature

        self.num_parts = 16

        encode_dim = self.hc

        self.contact_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.hc)
        self.part_encoder = Pointnet(in_dim=encode_dim + self.latentD + self.hc, hidden_dim=self.hc, out_dim=self.hc)
        self.pressure_encoder = Pointnet(in_dim=encode_dim + self.hc + self.num_parts, hidden_dim=self.hc, out_dim=self.hc)

        self.contact_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.part_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.pressure_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)

    def forward(self, embed_class, obj_cond, contacts_object, partition_object, pressure_object, global_vec):
        _, contact_latent = self.contact_encoder(torch.cat([obj_cond, contacts_object], -1))
        contact_mu, contact_std = self.contact_latent(contact_latent)
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))
        z_s_contact = z_contact.rsample()

        partition_feat = embed_class(partition_object.argmax(dim=-1))
        _, part_latent = self.part_encoder(
            torch.cat([obj_cond, z_s_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1), partition_feat], -1))
        part_mu, part_std = self.part_latent(part_latent)
        z_part = torch.distributions.normal.Normal(part_mu, torch.exp(part_std))
        _, pressure_latent = self.pressure_encoder(torch.cat([obj_cond, partition_feat, pressure_object], -1))
        pressure_mu, pressure_std = self.pressure_latent(pressure_latent)
        z_pressure = torch.distributions.normal.Normal(pressure_mu, torch.exp(pressure_std))
        z_s_part = z_part.rsample()
        z_s_pressure = z_pressure.rsample()

        return z_contact, z_part, z_pressure, z_s_contact, z_s_part, z_s_pressure


class TriDecoder(nn.Module):
    def __init__(self, cfg):
        super(TriDecoder, self).__init__()
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature
        self.num_parts = 16
        encode_dim = self.hc

        self.contact_decoder = Pointnet(in_dim=encode_dim + self.latentD, hidden_dim=self.hc, out_dim=1)
        self.part_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc,
                                     out_dim=self.num_parts)
        self.pressure_decoder = Pointnet(in_dim=self.hc + encode_dim + self.latentD, hidden_dim=self.hc, out_dim=self.num_parts)

    def forward(self, embed_class, z_contact, z_part, z_pressure, obj_cond, gt_partition_object=None, global_vec=None):

        z_contact = z_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object, _ = self.contact_decoder(torch.cat([z_contact, obj_cond], -1))
        contacts_object = torch.sigmoid(contacts_object)

        z_part = z_part.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        partition_object, _ = self.part_decoder(torch.cat([z_part, obj_cond, z_contact], -1))
        z_pressure = z_pressure.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)

        if gt_partition_object is not None:
            partition_feat = embed_class(gt_partition_object.argmax(dim=-1))
        else:
            partition_object_ = F.one_hot(partition_object.detach().argmax(dim=-1), num_classes=self.num_parts)
            partition_feat = embed_class(partition_object_.argmax(dim=-1))
        pressure_object, _ = self.pressure_decoder(torch.cat([z_pressure, obj_cond, partition_feat], -1))
        return contacts_object, partition_object, pressure_object

def base_obj_net(cfg):
    obj_pointnet = Pointnet2(in_dim=cfg.model.obj_feature, hidden_dim=cfg.model.pointnet_hc, out_dim=cfg.model.pointnet_hc)
    return obj_pointnet