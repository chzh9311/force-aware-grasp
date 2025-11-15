import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_net import LetentEncoder, Pointnet, Pointnet2

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


class ContactGenModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ContactGenModel, self).__init__()
        self.cfg = cfg
        self.n_neurons = cfg.n_neurons
        self.latentD = cfg.latentD
        self.hc = cfg.pointnet_hc
        self.object_feature = cfg.obj_feature

        self.num_parts = 16
        self.embed_class = nn.Embedding(self.num_parts, self.hc)

        encode_dim = self.hc
        self.obj_pointnet = Pointnet2(in_dim=self.object_feature, hidden_dim=self.hc, out_dim=self.hc)

        self.contact_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.hc)
        self.part_encoder = Pointnet(in_dim=encode_dim + self.latentD + self.hc, hidden_dim=self.hc, out_dim=self.hc)
        self.uv_encoder = Pointnet(in_dim=encode_dim + self.hc + 3, hidden_dim=self.hc, out_dim=self.hc)

        self.contact_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.part_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.uv_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)

        self.contact_decoder = Pointnet(in_dim=encode_dim + self.latentD, hidden_dim=self.hc, out_dim=1)
        self.part_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc,
                                     out_dim=self.num_parts)
        self.uv_decoder = Pointnet(in_dim=self.hc + encode_dim + self.latentD, hidden_dim=self.hc, out_dim=3)

    def encode(self, obj_cond, contacts_object, partition_object, uv_object):
        _, contact_latent = self.contact_encoder(torch.cat([obj_cond, contacts_object], -1))
        contact_mu, contact_std = self.contact_latent(contact_latent)
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))
        z_s_contact = z_contact.rsample()

        partition_feat = self.embed_class(partition_object.argmax(dim=-1))
        _, part_latent = self.part_encoder(
            torch.cat([obj_cond, z_s_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1), partition_feat], -1))
        part_mu, part_std = self.part_latent(part_latent)
        z_part = torch.distributions.normal.Normal(part_mu, torch.exp(part_std))
        _, uv_latent = self.uv_encoder(torch.cat([obj_cond, partition_feat, uv_object], -1))
        uv_mu, uv_std = self.uv_latent(uv_latent)
        z_uv = torch.distributions.normal.Normal(uv_mu, torch.exp(uv_std))
        z_s_part = z_part.rsample()
        z_s_uv = z_uv.rsample()

        return z_contact, z_part, z_uv, z_s_contact, z_s_part, z_s_uv

    def decode(self, z_contact, z_part, z_uv, obj_cond, gt_partition_object=None):

        z_contact = z_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object, _ = self.contact_decoder(torch.cat([z_contact, obj_cond], -1))
        contacts_object = torch.sigmoid(contacts_object)

        z_part = z_part.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        partition_object, _ = self.part_decoder(torch.cat([z_part, obj_cond, z_contact], -1))
        z_uv = z_uv.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)

        if gt_partition_object is not None:
            partition_feat = self.embed_class(gt_partition_object.argmax(dim=-1))
        else:
            partition_object_ = F.one_hot(partition_object.detach().argmax(dim=-1), num_classes=16)
            partition_feat = self.embed_class(partition_object_.argmax(dim=-1))
        uv_object, _ = self.uv_decoder(torch.cat([z_uv, obj_cond, partition_feat], -1))
        uv_object = normalize_vector(uv_object)
        return contacts_object, partition_object, uv_object

    def forward(self, verts_object, feat_object, contacts_object, partition_object, uv_object, **kwargs):
        obj_cond = self.obj_pointnet(torch.cat([verts_object, feat_object], -1))
        z_contact, z_part, z_uv, z_s_contact, z_s_part, z_s_uv = self.encode(obj_cond, contacts_object,
                                                                             partition_object, uv_object)
        results = {'mean_contact': z_contact.mean, 'std_contact': z_contact.scale,
                   'mean_part': z_part.mean, 'std_part': z_part.scale,
                   'mean_uv': z_uv.mean, 'std_uv': z_uv.scale}
        contacts_pred, partition_pred, uv_pred = self.decode(z_s_contact, z_s_part, z_s_uv, obj_cond, partition_object)

        results.update({'contacts_object': contacts_pred,
                        'partition_object': partition_pred,
                        'uv_object': uv_pred})
        return results

    def sample(self, verts_object, feat_object):
        bs = verts_object.shape[0]
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            obj_cond = self.obj_pointnet(torch.cat([verts_object, feat_object], -1))
            z_gen_contact = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_contact = torch.tensor(z_gen_contact, dtype=dtype).to(device)
            z_gen_part = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_part = torch.tensor(z_gen_part, dtype=dtype).to(device)
            z_gen_uv = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_uv = torch.tensor(z_gen_uv, dtype=dtype).to(device)
            return self.decode(z_gen_contact, z_gen_part, z_gen_uv, obj_cond)


class ContactGenEncoder(nn.Module):
    def __init__(self, cfg):
        super(ContactGenEncoder, self).__init__()
        self.cfg = cfg
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature

        self.num_parts = 16

        encode_dim = self.hc

        self.contact_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.hc)
        self.part_encoder = Pointnet(in_dim=encode_dim + self.latentD + self.hc, hidden_dim=self.hc, out_dim=self.hc)
        self.uv_encoder = Pointnet(in_dim=encode_dim + self.hc + 3, hidden_dim=self.hc, out_dim=self.hc)

        self.contact_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.part_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.uv_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)

        #
        # self.contact_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.hc)
        # self.part_encoder = Pointnet(in_dim=encode_dim + self.latentD + self.hc, hidden_dim=self.hc, out_dim=self.hc)
        # self.pressure_encoder = Pointnet(in_dim=encode_dim + self.hc + self.num_parts, hidden_dim=self.hc, out_dim=self.hc)
        #
        # self.contact_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        # self.part_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        # self.pressure_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)

    def forward(self, embed_class, obj_cond, contacts_object, partition_object, uv_object):
        _, contact_latent = self.contact_encoder(torch.cat([obj_cond, contacts_object], -1))
        contact_mu, contact_std = self.contact_latent(contact_latent)
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))
        z_s_contact = z_contact.rsample()

        partition_feat = embed_class(partition_object.argmax(dim=-1))
        _, part_latent = self.part_encoder(
            torch.cat([obj_cond, z_s_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1), partition_feat], -1))
        part_mu, part_std = self.part_latent(part_latent)
        z_part = torch.distributions.normal.Normal(part_mu, torch.exp(part_std))
        _, uv_latent = self.uv_encoder(torch.cat([obj_cond, partition_feat, uv_object], -1))
        uv_mu, uv_std = self.uv_latent(uv_latent)
        z_uv = torch.distributions.normal.Normal(uv_mu, torch.exp(uv_std))
        z_s_part = z_part.rsample()
        z_s_uv = z_uv.rsample()

        return z_contact, z_part, z_uv, z_s_contact, z_s_part, z_s_uv


class ContactGenDecoder(nn.Module):
    def __init__(self, cfg):
        super(ContactGenDecoder, self).__init__()
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature
        self.num_parts = 16
        encode_dim = self.hc

        self.contact_decoder = Pointnet(in_dim=encode_dim + self.latentD, hidden_dim=self.hc, out_dim=1)
        self.part_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc,
                                     out_dim=self.num_parts)
        self.uv_decoder = Pointnet(in_dim=self.hc + encode_dim + self.latentD, hidden_dim=self.hc, out_dim=3)

        # self.contact_decoder = Pointnet(in_dim=encode_dim + self.latentD, hidden_dim=self.hc, out_dim=1)
        # self.part_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc,
        #                              out_dim=self.num_parts)
        # self.pressure_decoder = Pointnet(in_dim=self.hc + encode_dim + self.latentD, hidden_dim=self.hc, out_dim=self.num_parts)

    def forward(self, embed_class, z_contact, z_part, z_uv, obj_cond, gt_partition_object=None):

        z_contact = z_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object, _ = self.contact_decoder(torch.cat([z_contact, obj_cond], -1))
        contacts_object = torch.sigmoid(contacts_object)

        z_part = z_part.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        partition_object, _ = self.part_decoder(torch.cat([z_part, obj_cond, z_contact], -1))
        z_uv = z_uv.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)

        if gt_partition_object is not None:
            partition_feat = embed_class(gt_partition_object.argmax(dim=-1))
        else:
            partition_object_ = F.one_hot(partition_object.detach().argmax(dim=-1), num_classes=16)
            partition_feat = embed_class(partition_object_.argmax(dim=-1))
        uv_object, _ = self.uv_decoder(torch.cat([z_uv, obj_cond, partition_feat], -1))
        uv_object = normalize_vector(uv_object)
        return contacts_object, partition_object, uv_object

def contactgen_obj_net(cfg):
    obj_pointnet = Pointnet2(in_dim=cfg.model.obj_feature, hidden_dim=cfg.model.pointnet_hc, out_dim=cfg.model.pointnet_hc)
    return obj_pointnet
