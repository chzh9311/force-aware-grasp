## This file implements the evaluation metrics for hand pose and shape
## Including FID, KID, Diversity, Precision/Recall, PenVolume

import torch
import torch.nn.functional as F
from common.evaluations.backbone import TwoHandFeaturePNEncoder, PointNetCLSMSG
from common.utils.vis import o3dmesh
from common.utils.manolayer import ManoLayer
import open3d as o3d

class FHIDMetrics:
    def __init__(self, model_name):
        if model_name == 'two_hand':
            self.model = TwoHandFeaturePNEncoder(num_class=99)
        elif model_name == 'single_hand':
            self.model = PointNetCLSMSG(num_class=45+9)

        self.n_hand_sample_pts = 1024
        self.n_samples = 10000
        self.mano_layer = ManoLayer('data/mano_v1_2')

    def feat_statistics(self, hand_params: (torch.Tensor, None), is_right=True):
        """
        The mu and sigma of the Gaussian that fits the distribution of latent code.
        """
        batch_size = 128
        rand_ids = torch.randperm(self.n_samples)
        feats = []
        rand_h_params = hand_params[rand_ids[:self.n_samples]]
        for i in range(0, self.n_samples, batch_size):
            v, j, f = self.mano_layer.rel_mano_forward(rand_h_params[i:min(i+batch_size, self.n_samples)],
                                                       is_right=False)
            for j in range(batch_size):
                hand_mesh = o3dmesh(v[j].detach().cpu().numpy(), f)
                # pts = o3d.geometry.sample_points_uniformly(hand_mesh)
                pts = hand_mesh.sample(self.n_hand_sample_pts)
            feat = self.model(hand_params)
            feats.append(feat)
        feats = torch.cat(feats, dim=0)
        mu = torch.mean(feats, dim=0)
        sigma = torch.cov(feats)
        return mu, sigma

    def calculate(self, gt, pred):
        gt_mu, gt_cov = self.feat_statistics(gt)
        pred_mu, pred_cov = self.feat_statistics(pred)
        fid = torch.sum(torch.square(pred_mu - gt_mu)) + torch.trace(gt_cov + pred_cov - 2 * torch.sqrt(gt_cov @ pred_cov))
        return fid