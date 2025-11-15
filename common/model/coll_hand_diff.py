import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import trimesh
from trimesh.sample import sample_surface
from tqdm import tqdm

import open3d as o3d

from common.utils.vis import o3dmesh
# This following line is necessary for visualizing Open3D.Geometry objects in tensorboard.
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from common.utils.manolayer import ManoLayer
from common.dataset_utils.arctic_dataset import get_samples_and_contacts
from common.utils.utils import mano_dict2array
from common.model.physics_constraint_solver import PhysicsConstraintSolver
from common.dataset_utils.arctic_objects import ObjectTensors
from common.evaluations.hand_pose import FHIDMetrics


class LitCondDiff(L.LightningModule):
    def __init__(self, model, scheduler, cfg, dataset_obj):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

        self.target_side = cfg.data.target_side
        self.n_obj_sampled_pts = cfg.data.object_sample
        self.ref_side = 'left' if self.target_side == 'right' else 'right'
        self.obj_sample = 2048

        self.n_timesteps = cfg.test.test_num_diffusion_timesteps
        self.n_samples = cfg.test.n_samples
        self.dataset_obj = dataset_obj
        self.dropout = cfg.model.dropout
        self.cond_w = cfg.model.w

        self.mano_layers = ManoLayer(cfg.data.mano_path)
        self.object_tensors = ObjectTensors()
        self.physics_constraint_solver = PhysicsConstraintSolver(cfg.pcs_cfg, cfg.data.mano_path)
        self.contact_dist = cfg.pcs_cfg.contact_dist
        self.contact_th = cfg.pcs_cfg.contact_th

        self.vis_compare = cfg.test.vis_compare

        self.fid_eval = FHIDMetrics('two_hand')

        ## only for saving hyperparams
        self.save_hyperparameters(dict(cfg))

    def training_step(self, batch, batch_idx):
        left_h, right_h = batch['left_hand'], batch['right_hand']
        if self.target_side == 'left':
            refer_h, target_h = right_h, left_h
        else:
            refer_h, target_h = left_h, right_h

        ref_cat_params = mano_dict2array(refer_h, rot_type='6d')
        target_cat_params = mano_dict2array(target_h, rot_type='6d')

        # Synthesize the target hand
        cond_target = ref_cat_params
        noise = torch.randn_like(target_cat_params).to(self.device)
        steps = torch.randint(self.scheduler.num_timesteps, (noise.shape[0],)).to(self.device)
        noisy_input = self.scheduler.add_noise(target_cat_params, noise, steps)
        param_res = self.model(noisy_input, steps.float(), cond_target, batch['object']['sampled_pts'])

        # Turn them into vertices
        # targetV, targetJ = self.mano_layers.rel_mano_forward(target_param_syn, matrix_to_axis_angle(obj['rot']),
        #                                      obj['trans'], normalize=True, is_right=True)

        # TODO: Implement loss, including params + joints + vertices.
        # loss = F.mse_loss(target_param_syn, target_cat_params)

        # The original loss.
        loss = F.mse_loss(noise, param_res)
        self.log('train_loss', loss, prog_bar=True)
        epoch_dict = {'loss': loss, 'log': {'loss': loss}}
        return epoch_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def forward_sample(self, batch, out_mid=False, denormalize=True):
        left_h, right_h = batch['left_hand'], batch['right_hand']
        if self.target_side == 'left':
            refer_h, target_h = right_h, left_h
        else:
            refer_h, target_h = left_h, right_h
        ref_cat_params = mano_dict2array(refer_h, rot_type='6d')
        target_cat_params = mano_dict2array(target_h, rot_type='6d')

        bs = ref_cat_params.shape[0]
        size = ref_cat_params.shape[1:]
        x_i = torch.randn(bs * self.n_samples, *size).to(self.device)
        ref_cat_param_n_sample = ref_cat_params.unsqueeze(1).repeat(1, self.n_samples, 1).view(-1, *size)
        sampled_pts_n_sample = batch['object']['sampled_pts'].unsqueeze(1).repeat(1, self.n_samples, 1, 1).view(
            bs*self.n_samples, -1, 3).to(self.device)

        x_is = []
        for t in range(self.n_timesteps, 0, -1):
            t_i = torch.tensor([t / self.n_timesteps]).to(self.device)
            t_is = t_i.repeat(bs * self.n_samples)
            z = torch.randn(bs, self.n_samples, *size).view(-1, *size).to(self.device)
            eps = self.model(x_i, t_is, ref_cat_param_n_sample, sampled_pts_n_sample)
            x_i = self.scheduler.denoise_step(x_i, eps, z, t)
            if out_mid:
                x_is.append(x_i)

        if denormalize:
            ref_cat_params = self.dataset_obj.denormalize_vector(ref_cat_params, self.ref_side)
            target_cat_params = self.dataset_obj.denormalize_vector(target_cat_params, self.target_side)
        if out_mid:
            if denormalize:
                x_is = [self.dataset_obj.denormalize_vector(x_i, self.target_side).view(bs, self.n_samples, -1) for x_i in x_is]
            else:
                x_is = [x_i.view(bs, self.n_samples, -1) for x_i in x_is]
            return x_is, ref_cat_params, target_cat_params
        else:
            if denormalize:
                x_i = self.dataset_obj.denormalize_vector(x_i, self.target_side).view(bs, self.n_samples, -1)
            else:
                x_i = x_i.view(bs, self.n_samples, -1)
            return x_i, ref_cat_params, target_cat_params

    def validation_step(self, batch, batch_idx):
        x_i_, ref_cat_params_, target_cat_params_ = self.forward_sample(batch, out_mid=False, denormalize=False)

        # The distance of the denormalized nearest hypothesis among all
        loss = torch.mean(torch.min(torch.norm(x_i_ - target_cat_params_.unsqueeze(1), dim=-1), dim=1)[0])
        x_i = self.dataset_obj.denormalize_vector(x_i_.view(-1, x_i_.shape[-1]), self.ref_side)
        target_cat_params = self.dataset_obj.denormalize_vector(target_cat_params_, self.target_side)
        self.log('val_loss', loss, sync_dist=True)
        # Visualize only one sample
        if batch_idx % 100 == 0:
            self.visualize_static(x_i=x_i, ref_param=self.dataset_obj.denormalize_vector(ref_cat_params_, self.ref_side),
                                  target_param=target_cat_params, obj=batch['object'],
                                  obj_names=batch['meta']['object_name'], step=batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.mano_layers.to(self.device)
        self.object_tensors.to(self.device)
        with torch.no_grad():
            x_is, x_i_s, ref_cat_params, target_cat_params = self.forward_sample(
                batch, out_mid=True, denormalize=True, compare=self.vis_compare)
            s_idx = torch.randint(self.n_samples, (1,)).item()
            ## Pick a random index for x
            x_is = [x_i[:, s_idx] for x_i in x_is]
            if x_i_s is not None:
                x_i_s = [x_i_[:, s_idx] for x_i_ in x_i_s]
            # lc, rc = batch['object']['lh_contacts'], batch['object']['rh_contacts']
            # obj_pts = batch['object']['sampled_pts']
            self.visualize_diff_process(x_is, ref_cat_params, target_cat_params, batch, x_is_cmp=x_i_s)

    def param_vector2dict(self, vec):
        """
        vec: bs * len
        """
        d = {'pose': vec[:, :45], 'shape': vec[:, 45:55], 'rot_6d': vec[:, 55:61], 'trans': vec[:, 61:]}
        return d

    def param_dict2vector(self, dct):
        """
        dict: pose, shape, rot, trans
        """
        return torch.cat((dct['pose'], dct['shape'], dct['rot_6d'], dct['trans']), dim=-1)

    def visualize_static(self, x_i, ref_param, target_param, obj, obj_names, step):
        """
        ref_param: target_param: denormalized hand param vectors
        obj: object param dicts
        """
        refV, refJ, refF = self.mano_layers.rel_mano_forward(ref_param, self.target_side == 'left', obj['rot_aa'], obj['trans'])
        tarV, tarJ, tarF = self.mano_layers.rel_mano_forward(target_param, self.target_side == 'right', obj['rot_aa'], obj['trans'])
        predV, predJ, predF = self.mano_layers.rel_mano_forward(x_i, self.target_side == 'right', obj['rot_aa'], obj['trans'])
        # if self.target_side == 'left':
        #     refF, tarF = self.mano_right_f, self.mano_left_f
        # else:
        #     refF, tarF = self.mano_left_f, self.mano_right_f

        # thing2dev(self.object_tensors.obj_tensors, self.device)
        ## TODO: Why this could only be done on cpu?
        obj_out = self.object_tensors(obj['arti'].to('cpu'), obj['rot_aa'].to('cpu'), obj['trans'].to('cpu'), obj_names)
        v, f = obj_out['v'], obj_out['f']
        writer = self.logger.experiment
        writer.add_3d('reference hand', to_dict_batch([o3dmesh(refV[0].detach().cpu().numpy(), refF, [0.8, 0.8, 0.8])]), step=step)
        writer.add_3d('object', to_dict_batch([o3dmesh(v[0].numpy(), f[0].numpy(), [0.9, 0.9, 0.9])]), step=step)
        writer.add_3d('target hand GT', to_dict_batch([o3dmesh(tarV[0].detach().cpu().numpy(), tarF, [0.8, 0.4, 0.4])]), step=step)
        writer.add_3d('target hand Pred', to_dict_batch([o3dmesh(predV[0].detach().cpu().numpy(), predF, [0.9, 0.2, 0.2])]), step=step)

    def visualize_diff_process(self, x_is, ref_cat_params, target_cat_params, batch,
                               x_is_cmp=None, lc=None, rc=None, obj_pts=None):
        obj_out = self.object_tensors(batch['object']['arti'], batch['object']['rot_aa'],
                                      batch['object']['trans'], batch['meta']['object_name'])
        vo, fo = obj_out['v'], obj_out['f']
        tarV, tarJ, tarF = self.mano_layers.rel_mano_forward(target_cat_params, self.target_side == 'right',
                                                             batch['object']['rot_aa'], batch['object']['trans'])
        refV, refJ, refF = self.mano_layers.rel_mano_forward(ref_cat_params, self.target_side == 'left',
                                                             batch['object']['rot_aa'], batch['object']['trans'])
        pred_list = []
        compare_list = []
        for k in range(len(x_is)):
            predV, predJ, _ = self.mano_layers.rel_mano_forward(x_is[k], self.target_side == 'right',
                                                                batch['object']['rot_aa'], batch['object']['trans'])
            if x_is_cmp is not None:
                cmpV, cmpJ, cmpF = self.mano_layers.rel_mano_forward(x_is_cmp[k], self.target_side == 'right',
                                                                     batch['object']['rot_aa'],
                                                                     batch['object']['trans'])
            pred_list.append(predV)
            compare_list.append(cmpV)
        predVs = torch.stack(pred_list, dim=1).detach().cpu().numpy()
        if x_is_cmp is not None:
            cmpVs = torch.stack(compare_list, dim=1).detach().cpu().numpy()
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Pred Vis', width=1600, height=1000)
        bs = refV.shape[0]
        for i in range(bs):
            tar_mesh = o3dmesh(tarV[i].detach().cpu().numpy(), tarF, [0.3, 0.4, 0.9])
            ref_mesh = o3dmesh(refV[i].detach().cpu().numpy(), refF, [0.7, 0.7, 0.7])
            obj_mesh = o3dmesh(vo[i].detach().cpu().numpy(), fo[i].detach().cpu().numpy(), [0.8, 0.8, 0.8])
            vis.clear_geometries()
            ## the point cloud that measures contact:
            if obj_pts is not None:
                pc_colors = torch.stack((lc, torch.zeros_like(lc), rc), dim=-1).detach().cpu().numpy()
                object_pc = o3d.geometry.PointCloud()
                object_pc.points = o3d.utility.Vector3dVector(obj_pts[i].detach().cpu().numpy())
                object_pc.colors = o3d.utility.Vector3dVector(pc_colors[i])
                vis.add_geometry(object_pc)
            for m in [tar_mesh, ref_mesh, obj_mesh]:
                vis.add_geometry(m)
            pred_mesh = o3dmesh(predVs[i, 0], tarF, [0.9, 0.5, 0.3])
            vis.add_geometry(pred_mesh)
            if x_is_cmp is not None:
                cmp_mesh = o3dmesh(cmpVs[i, 0], tarF, [0.5, 0.8, 0.8])
                vis.add_geometry(cmp_mesh)
            for k in range(0, self.n_timesteps, 1):
                pred_mesh.vertices = o3d.utility.Vector3dVector(predVs[i, k])
                vis.update_geometry(pred_mesh)
                if x_is_cmp is not None:
                    cmp_mesh.vertices = o3d.utility.Vector3dVector(cmpVs[i, k])
                    vis.update_geometry(cmp_mesh)
                vis.poll_events()
                vis.update_renderer()
        vis.destroy_window()


class LitGeometryCondDiff(LitCondDiff):
    """
    MDM style diffusion model. Different mainly in inference by predicting x0 directly.
    """
    def __init__(self, model, scheduler, cfg, dataset_obj):
        super(LitGeometryCondDiff, self).__init__(model, scheduler, cfg, dataset_obj)

    def training_step(self, batch, batch_idx):
        self.mano_layers.to(self.device)
        left_h, right_h = batch['left_hand'], batch['right_hand']
        if self.target_side == 'left':
            refer_h, target_h = right_h, left_h
        else:
            refer_h, target_h = left_h, right_h

        ref_cat_params = mano_dict2array(refer_h, rot_type='6d')
        target_cat_params = mano_dict2array(target_h, rot_type='6d')

        # Synthesize the target hand
        cond_hand = ref_cat_params
        noise = torch.randn_like(target_cat_params).to(self.device)
        steps = torch.randint(self.scheduler.num_timesteps, (noise.shape[0],)).to(self.device)
        noisy_input = self.scheduler.add_noise(target_cat_params, noise, steps)
        cond_object = batch['object']['sampled_pts']
        # Randomly drop conditions to enable unconditioned learning.
        # Since object is the base of generation, we do not drop object conditions
        drop_cond = (torch.rand(ref_cat_params.shape[0], device=self.device) < self.dropout).unsqueeze(-1)
        cond_hand = torch.zeros_like(cond_hand).to(self.device) * drop_cond + cond_hand * torch.logical_not(drop_cond)
        target_param_syn = self.model(noisy_input, steps.float(), cond_hand, cond_object)

        param_loss = F.mse_loss(target_param_syn, target_cat_params)
        syn_V, syn_J, syn_f = self.mano_layers.rel_mano_forward(target_param_syn, self.target_side == 'right',
                                         batch['object']['rot_aa'], batch['object']['trans'])
        tar_V, tar_J, syn_f = self.mano_layers.rel_mano_forward(target_cat_params, self.target_side == 'right',
                                         batch['object']['rot_aa'], batch['object']['trans'])
        vert_loss = self.euclidean_dist_loss(syn_V, tar_V) / 10
        joint_loss = self.euclidean_dist_loss(syn_J, tar_J) / 10
        loss = param_loss + vert_loss + joint_loss

        loss_dict = {'Total loss': loss, "param_loss":  param_loss, "vert_loss": vert_loss, "joint_loss": joint_loss}
        self.logger.experiment.add_scalars('Training loss', loss_dict, global_step=self.global_step)
        return loss

    def forward_sample(self, batch, out_mid=False, denormalize=True, compare=False):
        self.object_tensors.to(self.device)
        self.mano_layers.to(self.device)
        left_h, right_h = batch['left_hand'], batch['right_hand']
        if self.target_side == 'left':
            refer_h, target_h = right_h, left_h
        else:
            refer_h, target_h = left_h, right_h
        ref_cat_params = mano_dict2array(refer_h, rot_type='6d')
        target_cat_params = mano_dict2array(target_h, rot_type='6d')

        bs = ref_cat_params.shape[0]
        size = ref_cat_params.shape[1:]
        x_i = torch.randn(bs * self.n_samples, *size).to(self.device)
        # ref_cat_param_n_sample = ref_cat_params.unsqueeze(1).repeat(1, self.n_samples, 1).view(-1, *size)
        ref_cat_param_n_sample = ref_cat_params.repeat(self.n_samples, 1)

        ## Object vertices
        mesh_data = get_samples_and_contacts(batch['object'] | batch['meta'],
            {'lh_vs': batch['left_hand']['vertices'].cpu().numpy(),
             'rh_vs': batch['right_hand']['vertices'].cpu().numpy()},
            self.object_tensors,
            self.mano_layers,
            self.obj_sample, self.contact_dist, self.contact_th
        )
        for k, v in mesh_data.items():
            mesh_data[k] = torch.from_numpy(np.stack(v)).to(self.device).float()
        repeated_mesh_data = {}
        for k, v in mesh_data.items():
            size = mesh_data[k].shape
            if len(size) == 3:
                repeated_mesh_data[k] = mesh_data[k].unsqueeze(1).repeat(1, self.n_samples, 1, 1).reshape(bs*self.n_samples, *size[1:])
            else:
                repeated_mesh_data[k] = mesh_data[k].unsqueeze(1).repeat(1, self.n_samples, 1).reshape(bs*self.n_samples, *size[1:])
        # sampled_pts_n_sample = mesh_data['sampled_pts'].unsqueeze(1).repeat(1, self.n_samples, 1, 1).view(
        #                        bs*self.n_samples, -1, 3).to(self.device)

        ## Hand vertices
        ref_vs, _, ref_f = self.mano_layers.rel_mano_forward(ref_cat_params, self.target_side == 'left',
                                                             batch['object']['rot_aa'], batch['object']['trans'])
        # tar_vs, _, tar_f = self.mano_layers.rel_mano_forward(target_cat_params, self.target_side == 'right',
        #                                                      batch['object']['rot_aa'], batch['object']['trans'])

        x_is = []
        x_i_s = []
        x_i_ = x_i.clone()
        hand_cond = torch.zeros_like(ref_cat_param_n_sample).to(self.device)
        size = ref_cat_params.shape[1]
        for t in tqdm(range(self.n_timesteps, 0, -1)):
            t_i = torch.tensor([t / self.n_timesteps]).to(self.device)
            t_is = t_i.repeat(bs * self.n_samples)
            z = torch.randn(bs, self.n_samples, size).view(-1, size).to(self.device)
            ## first predict x0, then add noise to t-1
            x0_cond = self.model(x_i, t_is, ref_cat_param_n_sample, repeated_mesh_data['sampled_pts'])
            if compare:
                x0_ = self.model(x_i_, t_is, ref_cat_param_n_sample, repeated_mesh_data['sampled_pts'])
            # x0_uncond = self.model(x_i, t_is, hand_cond, sampled_pts_n_sample)
            # x0 = (1+self.cond_w) * x0_cond - self.cond_w * x0_uncond
            ## Process with physics guidance
            with torch.enable_grad():
                x0 = x0_cond.requires_grad_()
                if self.physics_constraint_solver is not None:
                    # V0, _, f = self.mano_layers.rel_mano_forward(x0, self.target_side == 'right')
                    if self.target_side == 'left':
                        self.physics_constraint_solver.lp_gradient(x0, ref_cat_params.repeat(self.n_samples, 1),
                                                                   batch['object']['rot_aa'].repeat(self.n_samples, 1),
                                                                   batch['object']['trans'].repeat(self.n_samples, 1),
                                                                   repeated_mesh_data['sampled_pts'],
                                                                   repeated_mesh_data['sampled_pts_normals'],
                                                                   repeated_mesh_data['coms'], self.device)
                    else:
                        self.physics_constraint_solver.lp_gradient(ref_cat_params.repeat(self.n_samples, 1), x0,
                                                                   batch['object']['rot_aa'].repeat(self.n_samples, 1),
                                                                   batch['object']['trans'].repeat(self.n_samples, 1),
                                                                   repeated_mesh_data['sampled_pts'],
                                                                   repeated_mesh_data['sampled_pts_normals'],
                                                                   repeated_mesh_data['coms'], self.device)
                    x0 = x0 + x0.grad
            if t > 1:
                x_i = self.scheduler.add_noise(x0, z, t-1)
                if compare:
                    x_i_ = self.scheduler.add_noise(x0_, z, t-1)
            else:
                x_i = x0
                if compare:
                    x_i_ = x0_
            if out_mid:
                x_is.append(x_i)
                if compare:
                    x_i_s.append(x_i_)

        ## Avoid errors
        if not compare:
            x_i_s = x_i_ = None
        if denormalize:
            ref_cat_params = self.dataset_obj.denormalize_vector(ref_cat_params, self.ref_side)
            target_cat_params = self.dataset_obj.denormalize_vector(target_cat_params, self.target_side)
        if out_mid:
            if denormalize:
                x_is = [self.dataset_obj.denormalize_vector(x_i, self.target_side).view(bs, self.n_samples, -1)
                        for x_i in x_is]
                if compare:
                    x_i_s = [self.dataset_obj.denormalize_vector(x_i_, self.target_side).view(bs, self.n_samples, -1)
                             for x_i_ in x_i_s]
            else:
                x_is = [x_i.view(bs, self.n_samples, -1) for x_i in x_is]
                if compare:
                    x_i_s = [x_i_.view(bs, self.n_samples, -1) for x_i_ in x_i_s]
            return x_is, x_i_s, ref_cat_params, target_cat_params
        else:
            if denormalize:
                x_i = self.dataset_obj.denormalize_vector(x_i, self.target_side).view(bs, self.n_samples, -1)
                if compare:
                    x_i_ = self.dataset_obj.denormalize_vector(x_i_, self.target_side).view(bs, self.n_samples, -1)
            else:
                x_i = x_i.view(bs, self.n_samples, -1)
                if compare:
                    x_i_ = x_i_.view(bs, self.n_samples, -1)
            return x_i, x_i_, ref_cat_params, target_cat_params

    def validation_step(self, batch, batch_idx):
        self.mano_layers.to(self.device)
        x_i_, x_i__, ref_cat_params_, target_cat_params_ = self.forward_sample(batch, out_mid=False, denormalize=False)

        # The distance of the denormalized nearest hypothesis among all
        param_loss = torch.mean(torch.min(torch.mean((x_i_ - target_cat_params_.unsqueeze(1))**2, dim=-1), dim=1)[0])
        x_i = self.dataset_obj.denormalize_vector(x_i_.view(-1, x_i_.shape[-1]), self.target_side)
        target_cat_params = self.dataset_obj.denormalize_vector(target_cat_params_, self.target_side)
        syn_V, syn_J, syn_f = self.mano_layers.rel_mano_forward(x_i, self.target_side == 'right',
                                                batch['object']['rot_aa'].view(-1, 1, 3).repeat(1, self.n_samples, 1).view(-1, 3),
                                                batch['object']['trans'].view(-1, 1, 3).repeat(1, self.n_samples, 1).view(-1, 3))
        syn_V = syn_V.view(-1, self.n_samples, *syn_V.shape[-2:])
        syn_J = syn_J.view(-1, self.n_samples, *syn_J.shape[-2:])
        tar_V, tar_J, syn_f = self.mano_layers.rel_mano_forward(target_cat_params, self.target_side == 'right',
                                                batch['object']['rot_aa'], batch['object']['trans'])
        vert_loss = self.mh_euclidean_dist_loss(syn_V, tar_V) / 10
        joint_loss = self.mh_euclidean_dist_loss(syn_J, tar_J) / 10
        loss = param_loss + vert_loss + joint_loss
        loss_dict = {'Total loss': loss, "param_loss":  param_loss, "vert_loss": vert_loss, "joint_loss": joint_loss}
        self.logger.experiment.add_scalars('Validation loss', loss_dict, global_step=self.global_step)
        # Visualize only one sample
        self.visualize_static(x_i=x_i.view(-1, self.n_samples, x_i.shape[-1])[:, 0],
                              ref_param=self.dataset_obj.denormalize_vector(ref_cat_params_, self.ref_side),
                              target_param=target_cat_params, obj=batch['object'],
                              obj_names=batch['meta']['object_name'], step=self.global_step)

    def test_step(self, batch, batch_idx):
        self.mano_layers.to(self.device)
        x_i_, x_i__, ref_cat_params_, target_cat_params_ = self.forward_sample(batch, out_mid=False, denormalize=True)
        predV, predJ, tarF = self.mano_layers.rel_mano_forward(x_i_, self.target_side == 'right',
                                                               batch['object']['rot_aa'], batch['object']['trans'])
        gtV, gtJ, tarF = self.mano_layers.rel_mano_forward(x_i_, self.target_side == 'right',
                                                               batch['object']['rot_aa'], batch['object']['trans'])
        bs = x_i_.shape[0]
        for i in range(bs):
            predMesh = o3dmesh(predV.detach().cpu().numpy(), tarF)

    def euclidean_dist_loss(self, vec1, vec2):
        """
        vec1, vec2: Tensors of ... x N
        """
        dists = torch.norm(vec1 - vec2, dim=-1)
        return torch.mean(dists)

    def mh_euclidean_dist_loss(self, vec1, vec2):
        """
        vec1: bs x n_samples x ... x N;
        vec2: bs x ... x N;
        returns the average of the minimum distances among all samples in vec1 with vec2
        """
        dists = torch.norm(vec1 - vec2.unsqueeze(1), dim=-1)
        return torch.mean(torch.min(dists, dim=1)[0])