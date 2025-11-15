import datetime
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from numpy.ma.core import masked_print_option
from torch import optim
import lightning as L
import trimesh
from collections import defaultdict
from copy import deepcopy, copy
import wandb
from multiprocessing.pool import Pool
import time
import pytorch3d

from prev_sota.contactgen.manopth.manolayer import ManoLayer as testML
from common.utils.manolayer import ManoLayer
from common.model.hand_object import HandObject
from common.model.pose_optimizer import phy_optimize_pose, phy_optimize_sdf_hand
from common.model.losses import calc_stable_loss, calc_aggregated_stable_loss, part_cluster_loss
from prev_sota.contactgen.hand_sdf import ArtiHand
from common.baselines.contact_optimizer import optimize_pose
# from prev_sota.contactgen.base_net import optimize_pose as ctg_optimize_pose
# from prev_sota.contactgen.base_net import compute_uv
from common.evaluations.eval_fns import parallel_calculate_metrics, pressure_value_error, calc_diversity


class PyramidCVAE(L.LightningModule):
    def __init__(self, cfg, encoder, decoder, obj_feat_net, debug=False):
        super(PyramidCVAE, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.label_cfg = cfg.label
        self.n_neurons = cfg.model.n_neurons
        self.latentD = cfg.model.latentD
        self.hc = cfg.model.pointnet_hc
        self.object_feature = cfg.model.obj_feature

        self.num_parts = 16
        self.embed_class = nn.Embedding(self.num_parts, self.hc)
        self.encoder = encoder
        self.decoder = decoder
        self.obj_feat_net = obj_feat_net

        ## Other utils
        ml = ManoLayer('data/mano_v1_2', use_pca=True, ncomps=26, flat_hand_mean=False)
        self.hand_faces = ml.mano_f['right']
        self.part_ids = ml.part_id_right
        self.testml = [testML(mano_root='data/mano_v1_2/models', use_pca=True, ncomps=26, side='right', flat_hand_mean=False)]
        self.mano_layer = ml
        self.inv_obj_rot = True
        self.lr = cfg.train.lr
        self.scheduler_step = cfg.train.scheduler_step
        self.lr_gamma = cfg.train.lr_gamma
        self.max_kl_coef = cfg.model.kl_coef
        self.weight_rec = 1.0
        self.weight_stable = 2.0
        self.weight_part_cluster = 5.0
        self.contact_th = cfg.data.contact_th

        self.hand_model = ArtiHand(self.cfg.hand_model_params, pose_size=self.cfg.pose_size)
        checkpoint = torch.load("prev_sota/contactgen/ckpts/hand_model.pt")
        self.hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.hand_model.eval()
        self.hand_model.to(self.device)
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'obj_feat_net'])
        self.validation_step_outputs = []
        self.pressure_quantise_splits = np.load(os.path.join(cfg.data.force_label_dir, 'separations.npy'))
        splits = torch.as_tensor(self.pressure_quantise_splits).to(self.device).float()
        mid_pts = torch.zeros(splits.shape[0] + 1, device=self.device).float()
        mid_pts[:-1] = splits
        mid_pts[-1] = 2 * splits[-1] - splits[-2]
        mid_pts[1:] += splits
        mid_pts[0] = 0
        self.mid_pts = mid_pts / 2

        self.test_gt = False
        self.debug = debug
        ## For simulation
        # self.object_templates = kwargs['object_templates']
        # self.object_hulls = kwargs['object_hulls']

    def forward(self, verts_object, feat_object, contacts_object, partition_object, pressure_object, **kwargs):
        obj_cond = self.obj_feat_net(torch.cat([verts_object, feat_object], -1))
        z_contact, z_part, z_pressure, z_s_contact, z_s_part, z_s_pressure = self.encoder(self.embed_class, obj_cond, contacts_object,
                                                                                         partition_object, pressure_object, kwargs['gravity_direction'])
        results = {'mean_contact': z_contact.mean, 'std_contact': z_contact.scale,
                   'mean_part': z_part.mean, 'std_part': z_part.scale,
                   'mean_pressure': z_pressure.mean, 'std_pressure': z_pressure.scale}
        contacts_pred, partition_pred, pressure_pred = self.decoder(self.embed_class, z_s_contact, z_s_part, z_s_pressure, obj_cond, partition_object, kwargs['gravity_direction'])

        results.update({'contacts_object': contacts_pred,
                        'partition_object': partition_pred,
                        'pressure_object': pressure_pred})
        return results

    def sample(self, verts_object, feat_object, gravity_direction):
        bs = verts_object.shape[0]
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            obj_cond = self.obj_feat_net(torch.cat([verts_object, feat_object], -1))
            z_gen_contact = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_contact = torch.tensor(z_gen_contact, dtype=dtype).to(device)
            z_gen_part = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_part = torch.tensor(z_gen_part, dtype=dtype).to(device)
            z_gen_pressure = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_pressure = torch.tensor(z_gen_pressure, dtype=dtype).to(device)
            return self.decoder(self.embed_class, z_gen_contact, z_gen_part, z_gen_pressure, obj_cond)

    def training_step(self, batch, batch_idx):
        return self.train_val_process(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.train_val_process(batch, batch_idx, 'val')

    def train_val_process(self, batch, batch_idx, proc_name='train'):
        num_total_iter = self.trainer.num_training_batches if proc_name == 'train' else self.trainer.num_val_batches
        if type(num_total_iter) is list:
            num_total_iter = num_total_iter[0]

        self.weight_kl = self.kl_coeff(step=self.global_step,
                                       total_step=num_total_iter,
                                       constant_step=0,
                                       min_kl_coeff=1e-7,
                                       max_kl_coeff=self.max_kl_coef)

        ho_gt = HandObject(self.device, self.hand_faces, self.mano_layer.part_id_right, self.pressure_quantise_splits, contact_th=self.contact_th)
        ho_gt.load_from_batch(batch, inv_obj_rot=self.inv_obj_rot)
        verts_obj, obj_features, contacts, parts, pressure, grav_dire = ho_gt.get_phy_reps(
            self.model_cfg.obj_feature_keys, random_rotate=True if proc_name=='train' else False)
        results = self.forward(verts_obj, obj_features, contacts, parts, pressure, gravity_direction=grav_dire)
        gt = {'verts_object': verts_obj, 'normals_object':ho_gt.obj_normals, 'gravity_direction': grav_dire, 'contacts_object': contacts, 'partition_object': parts, 'pressure_object': pressure}
        disps = torch.norm(batch['simuDisp'][:, :3], dim=-1)
        disp_weight = 0.05 / (disps + 1e-6)
        disp_weight[disp_weight > 1] = 1
        total_loss, loss_dict = self.loss_net(gt, results, disp_weight)
        log_loss_dict = {}
        for k, v in loss_dict.items():
            log_loss_dict[proc_name+'/'+k] = v
        if proc_name == 'train':
            self.logger.experiment.log(log_loss_dict)
        else:
            self.validation_step_outputs.append(loss_dict)
        if batch_idx % self.cfg[proc_name].vis_every_n_batch == 0:
            if proc_name == 'train':
                dataset = self.trainer.datamodule.train_set
            elif proc_name == 'val':
                dataset = self.trainer.datamodule.train_set
            obj_templates = []
            for obj_name in batch['objName']:
                obj_templates.append(
                    trimesh.Trimesh(dataset.obj_info[obj_name]['verts'], dataset.obj_info[obj_name]['faces']))
            ho_gt.load_from_batch(batch, obj_templates=obj_templates, inv_obj_rot=self.inv_obj_rot)
            gt_img = ho_gt.vis_img(0, 300, 1200)
            ho_pred = copy(ho_gt)
            ho_pred.contact_map, ho_pred.part_map, ho_pred.onehot_pressure = results['contacts_object'].squeeze(
                -1), results['partition_object'], results['pressure_object']
            pred_img = ho_pred.vis_img(0, 300, 1200)
            img = wandb.Image(np.concatenate((gt_img, pred_img), axis=1), caption="Left: GT; Right: Pred")
            self.logger.experiment.log({f"{proc_name} Sample": img})
        return total_loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        all_metrics = defaultdict(list)
        for out in self.validation_step_outputs:
            for k, v in out.items():
                all_metrics[k].append(v)

        mean_metrics = {}
        for k, v in all_metrics.items():
            mean_metrics[k] = sum(v) / len(v)

        self.logger.experiment.log(mean_metrics)
        return mean_metrics

    def on_test_epoch_start(self):
        self.pool = Pool(min(self.cfg.test.batch_size, 32))
        t = datetime.datetime.now()
        self.dump_dir = os.path.join('logs/grasp_results/')
        os.makedirs(self.dump_dir, exist_ok=True)
        self.tmp_dump_file = os.path.join(self.dump_dir, t.strftime('%Y%m%d-%H%M%S') + '.pkl')
        self.sample_joints = []

        ## Testing metrics
        self.runtime = 0
        if not self.debug:
            wandb.define_metric("PyBullet SimuDisp", summary='mean')
            wandb.define_metric("MuJoCo SimuDisp", summary='mean')
            wandb.define_metric("PyBullet Stable Rate", summary='mean')
            wandb.define_metric("MuJoCo Stable Rate", summary='mean')
            wandb.define_metric('Intersection Volume', summary='mean')
            wandb.define_metric('Contact Ratio', summary='mean')
            wandb.define_metric('Pressure Acc', summary='mean')
            wandb.define_metric('Pressure Err', summary='mean')
            wandb.define_metric('Entropy', summary='mean')
            wandb.define_metric('Cluster Size', summary='mean')
            wandb.define_metric('Canonical Entropy', summary='mean')
            wandb.define_metric('Canonical Cluster Size', summary='mean')
            wandb.define_metric('Force value error', summary='mean')
            wandb.define_metric('Clustered force value error', summary='mean')

        if os.path.exists(self.tmp_dump_file):
            with open(self.tmp_dump_file, 'rb') as f:
                self.sample_result = pickle.load(f)
        else:
            self.sample_result = []

        self.result_dict = {'PyBullet SimuDisp': [], 'MuJoCo SimuDisp': [], 'Penetration': []}

    def test_step(self, batch, batch_idx):
        batch_size = self.cfg.test.batch_size
        obj_templates, obj_hull_templates = [], []
        dataset = self.trainer.datamodule.test_set
        for obj_name in batch['objName']:
            obj_templates.append(
                trimesh.Trimesh(dataset.obj_info[obj_name]['verts'], dataset.obj_info[obj_name]['faces']))
            obj_hull_templates.append(dataset.obj_hulls[obj_name])

        self.mano_layer.to(self.device)
        ho_gt = HandObject(self.device, self.hand_faces, self.mano_layer.part_id_right, self.pressure_quantise_splits, contact_th=self.contact_th)

        obj_names = batch['objName']
        ho_gt.load_from_batch_obj_only(batch, obj_templates=obj_templates, obj_hulls=obj_hull_templates, inv_obj_rot=self.inv_obj_rot)
        # use_obj_features = torch.cat([obj_features[k] for k in self.model_cfg.obj_feature_keys], dim=-1)
        verts_obj, obj_features, gt_contacts, gt_parts, gt_pressure, grav_dire = ho_gt.get_phy_reps(
            self.model_cfg.obj_feature_keys)
        batch_start = time.time()
        sample_result = self.sample(verts_obj, obj_features, grav_dire)
        pred_contacts, pred_parts, pred_pressure = sample_result
        # pred_contacts, pred_parts = sample_result
        # pred_contacts, pred_parts, pred_pressure = gt_contacts, gt_parts, gt_pressure
        # res = {'pred_contacts': pred_contacts, 'pred_parts': pred_parts, 'pred_pressure': pred_pressure}

        contacts_object = pred_contacts.squeeze(-1)
        partition_object = pred_parts.argmax(dim=-1)
        pressure_object = torch.sum(F.softmax(pred_pressure * 50, dim=-1) * self.mid_pts.to(self.device).view(1, 1, -1), dim=-1)
        ## Ablation: avg predictor
        # pressure_object[:] = 0.5295791

        with torch.enable_grad():
            # global_pose, mano_pose, mano_shape, mano_trans, vis_params = phy_optimize_pose(ho_gt.obj_verts, ho_gt.obj_normals, self.mano_layer,
            #                                                                    contacts_object, partition_object, pressure_object, glob_iter=200,
            #                                                                                n_iter=500, obj_name=batch['objName'], contact_th=self.contact_th)
            self.testml[0].to(self.device)
            # global_pose, mano_pose, mano_shape, mano_trans, history, vis_params = phy_optimize_sdf_hand(
            global_pose, mano_pose, mano_shape, mano_trans, part_pres, new_part_pres = phy_optimize_sdf_hand(
                                self.hand_model, self.testml[0], self.mano_layer, verts_obj, ho_gt.obj_normals,
                                contacts_object, partition_object, pressure_object, pose_iter=1000, ret_history=False)

            self.runtime += time.time() - batch_start
            print(self.runtime / (batch_idx + 1))
            # global_pose1, mano_pose1, mano_shape1, mano_trans1 = phy_optimize_sdf_hand(self.hand_model, self.testml[0], self.mano_layer, verts_obj, ho_gt.obj_normals,
            #                                                                            contacts_object, partition_object, pressure_object, pose_iter=1000, ret_history=False, w_kp=0)
            # w_contact=args.w_contact, w_pene=args.w_pene,
            # w_uv=args.w_uv)
        res = {'pred_contacts': contacts_object.unsqueeze(-1), 'pred_parts': ho_gt.part_map, 'pred_pressure': ho_gt.pressure_map}
        hand_params = {'rot_aa': global_pose, 'pose': mano_pose, 'shape': mano_shape, 'trans': mano_trans}
        handV, handJ, handF = self.mano_layer.mesh_data_np(hand_params, is_right=True)
        contact_mask = contacts_object > self.contact_th

        ho_pred = copy(ho_gt)
        res.update({'hand_verts': handV, 'hand_joints': handJ,
                    'obj_verts': np.stack([om.vertices for om in ho_gt.obj_models], axis=0),
                    'obj_faces': np.stack([om.faces for om in ho_gt.obj_models], axis=0)})
        self.sample_result.append(res)
        # ho_pred.hand_models = [trimesh.Trimesh(handV[i], handF) for i in range(handV.shape[0])]
        # if self.model_cfg.name == 'contactgen':
        #     handrot = torch.cat([global_pose, mano_pose], dim=-1)
        #     handV, handJ, hand_frames = self.mano_layer(handrot, th_betas=mano_shape, th_trans=mano_trans)
        #     handV = handV.detach().cpu().numpy()
        #     handJ = handJ.detach().cpu().numpy()
        # else:

        hand_models = [trimesh.Trimesh(handV[i], self.hand_faces) for i in range(handV.shape[0])]
        ho_pred.hand_models = hand_models
        ho_pred.hand_joints = handJ
        ho_pred.update_contact_from_hand_models()

        # ho_baseline = copy(ho_pred)
        # hand_params1 = {'rot_aa': global_pose1, 'pose': mano_pose1, 'shape': mano_shape1, 'trans': mano_trans1}
        # handV1, handJ1, handF1 = self.mano_layer.mesh_data_np(hand_params1, is_right=True)
        # ho_baseline.hand_models = [trimesh.Trimesh(handV1[i], self.hand_faces) for i in range(handV1.shape[0])]
        # ho_baseline.hand_joints = handJ1

        ## Calculate force errors: in a way similar to heatmap
        # pe = pressure_error(gt_pressure, pred_pressure, verts_obj)

        # for i in range(handV.shape[0]):
        param_list = [{'label_cfg': deepcopy(self.label_cfg), 'dataset_name': 'grab',
                       'frame_name': f"{obj_names[i]}_{i}", 'hand_model': hand_models[i],
                       'obj_name': obj_names[i], 'hand_joints': handJ[i],
                       'obj_model': ho_gt.obj_models[i], 'obj_hulls': ho_gt.obj_hulls[i],
                       'idx': batch_idx * batch_size + i, 'part_id': self.part_ids} for i in range(len(ho_pred.hand_models))]
        # self.obj_hulls += ho_gt.obj_hulls
        assert handJ.shape[1] == 21
        self.sample_joints.append(handJ)

        result_metrics = self.pool.map(parallel_calculate_metrics, param_list)
        int_vol, contact_ratio = np.stack([res['int_vol'] for res in result_metrics]), np.stack([res['contact_ratio'] for res in result_metrics])
        pb_disp = np.stack([res['pb_disp'] for res in result_metrics], axis=0) * 100
        disps = np.stack([np.linalg.norm(res['obj_disp'][:3]) for res in result_metrics], axis=0) * 100
        qposes = np.stack([res['obj_disp'] for res in result_metrics], axis=0)
        label_disps = np.stack([res['label_obj_disp'] for res in result_metrics], axis=0)
        contacts = [res['contacts'] for res in result_metrics]
        #
        metrics = {
            'PyBullet SimuDisp': np.mean(pb_disp), 'PyBullet Stable Rate': np.sum(pb_disp < 2) / disps.shape[0],
            'MuJoCo SimuDisp': np.mean(disps), 'MuJoCo Stable Rate': np.sum(disps < 2) / disps.shape[0],
            'Intersection Volume': np.mean(int_vol), 'Contact Ratio': np.mean(contact_ratio),
        }
        self.result_dict['PyBullet SimuDisp'].append(pb_disp)
        self.result_dict['MuJoCo SimuDisp'].append(disps)
        self.result_dict['Penetration'].append(int_vol)

        ## Calculate the GT pressure map using simulation.
        ho_pred.calculate_pressure(contacts, torch.as_tensor(qposes).float(), torch.as_tensor(label_disps).float())
        ## Get part-level contact forces
        gt_part_pres = torch.zeros_like(part_pres, device=self.device).float()
        for b in range(batch_size):
            for c in contacts[b]:
                part_id = c['part_id']
                part_force = torch.as_tensor(c['frame'].reshape(3, 3).T @ c['force'].reshape(3, 1), device=self.device).squeeze()
                gt_part_pres[b, part_id] += part_force

        ## Force related predictions:
        metrics['Force value error'] = torch.mean(torch.abs(torch.abs(torch.norm(gt_part_pres, dim=-1) - torch.norm(part_pres, dim=-1))))
        # metrics['Force angular error'] = torch.mean(torch.arccos(torch.sum(gt_part_pres * part_pres, dim=-1) / (torch.norm(gt_part_pres, dim=-1)*torch.norm(part_pres, dim=-1) + 1e-8)))
        metrics['Clustered force value error'] = torch.mean(torch.abs(torch.abs(torch.norm(gt_part_pres, dim=-1) - torch.norm(new_part_pres, dim=-1))))
        # metrics['Clustered force angular error'] = torch.mean(torch.arccos(torch.sum(gt_part_pres * new_part_pres, dim=-1) / (torch.norm(gt_part_pres, dim=-1)*torch.norm(new_part_pres, dim=-1) + 1e-8)))

        simu_pressure = ho_pred.quantised_pressure
        # pred_q_pressure = torch.argmax(pred_pressure, dim=-1)
        if self.model_cfg.name != 'external' and contact_mask.any() and not self.test_gt:
            # pressure_acc = torch.sum(simu_pressure[contact_mask] == pred_q_pressure[contact_mask]) / torch.sum(contact_mask)
            metrics.update({'Pressure Err': pressure_value_error(contact_mask, ho_pred.pressure_map.sum(dim=-1), pressure_object)})
            # gt_pressure = ho_pred.pressure_map.sum(dim=-1)[contact_mask]
            # metrics.update({'Pressure Err Avg Predictor': torch.abs(gt_pressure - 0.5295791).mean()})
            # metrics.update({'Quantised Pressure Err': pressure_value_error(contact_mask, self.mid_pts.to(self.device)[simu_pressure], pressure_object)})
        #
        #     ho_pred.onehot_pressure = pred_pressure
        #
        if not self.debug:
            self.logger.log_metrics(metrics)

        # if batch_idx % self.cfg.test.vis_every_n_batch == 0:
        print(metrics)
        if batch_idx % 1 == 1:
            ## Visualize some samples:
            ### Update sample:
            # for idx in range(32):
            #     print(batch['handSide'][idx])
            #     ho_pred.vis_frame(idx)
            # idx = min(5, ho_pred.obj_verts.shape[0]-1)
            for idx in range(batch_size):
                if not self.test_gt and not self.model_cfg.name == 'external' and self.debug:
                    ho_pred.contact_map, ho_pred.part_map, ho_pred.onehot_pressure = pred_contacts.squeeze(
                        -1), pred_parts, pred_pressure
                    contact_mask = ho_pred.contact_map > 0.11
                    pressure_object = torch.sum(
                        F.softmax(pred_pressure * 50, dim=-1) * self.mid_pts.to(self.device).view(1, 1, -1), dim=-1)
                    # for idx in range(len(ho_pred.hand_models)):
                    # ho_pred.vis_frame(idx, show_procedure=False, show_optim_vars=True, history=history, **vis_params)
                        ## gt
                    # ho_pred.aggregate_simu_forces()
                    # ho_pred.vis_frame(idx, draw_multi_objs=False, draw_maps=False)
                    # ho_pred.simu_force_vecs[idx] = []
                    # ho_pred.simu_contact_pts[idx] = []
                    # ho_pred.simu_part_ids[idx] = []
                    # avg_force = 0.5295791
                    # ho_avg = copy(ho_pred)
                    # for i in range(16):
                    #     part_mask = torch.logical_and(contact_mask[idx], partition_object[idx] == i)  # B x N
                    #     if torch.sum(part_mask) > 0:
                    #         pres = torch.sum(pressure_object[idx].unsqueeze(-1) * -ho_pred.obj_normals[idx] * part_mask.unsqueeze(-1),
                    #                          dim=0)  # B x 3
                    #         contact_pt = torch.sum(pressure_object[idx].unsqueeze(-1) * ho_pred.obj_verts[idx] * part_mask.unsqueeze(-1),
                    #                                dim=0) / ( torch.sum(pressure_object[idx] * part_mask) + 1e-8)  # B x 3
                    #         ho_pred.simu_force_vecs[idx].append(pres)
                    #         ho_pred.simu_contact_pts[idx].append(contact_pt)
                    #         ho_pred.simu_part_ids[idx].append(i)
                    #         avg_pres = torch.sum(avg_force * -ho_pred.obj_normals[idx] * part_mask.unsqueeze(-1), dim=0)
                    #         avg_contact_pt = torch.sum(avg_force * ho_pred.obj_verts[idx] * part_mask.unsqueeze(-1),
                    #                                    dim=0) / (torch.sum(avg_force * part_mask) + 1e-8)  # B x 3
                    #         ho_avg.simu_force_vecs[idx].append(avg_pres)
                    #         ho_avg.simu_contact_pts[idx].append(avg_contact_pt)
                    #         ho_avg.simu_part_ids[idx].append(i)
                    # ho_pred.simu_force_vecs[idx] = torch.stack(ho_pred.simu_force_vecs[idx])
                    # ho_avg.simu_force_vecs[idx] = torch.stack(ho_avg.simu_force_vecs[idx])
                    # # ho_baseline.vis_frame(idx, draw_multi_objs=False, draw_maps=False, draw_force_arrows=False)
                    # ho_pred.vis_frame(idx, draw_multi_objs=False, draw_maps=False) #, show_optim_vars=True, **vis_params)
                    # ho_avg.vis_frame(idx, draw_multi_objs=False, draw_maps=False)

                    ho_pred.vis_history(idx, history, **vis_params)
            # gt_img = ho_gt.vis_img(idx, 300, 1200)
            if not self.debug:
                pred_img = ho_pred.vis_img(idx, 300, 1200)
                # vis_img = np.concatenate((gt_img, pred_img), axis=1)
                vis_img = pred_img
                vis_img = wandb.Image(vis_img, caption='Sampled Result')
                self.logger.experiment.log({'Sample Results': vis_img}, step=batch_idx * batch_size + idx)

    def on_test_epoch_end(self):
        self.sample_joints = np.concatenate(self.sample_joints)
        entropy, cluster_size, entropy2, cluster_size2 = calc_diversity(self.sample_joints)
        self.logger.log_metrics({'Entropy': np.mean(entropy), 'Canonical Entropy': np.mean(entropy2),
                                'Cluster Size': np.mean(cluster_size), 'Canonical Cluster Size': np.mean(cluster_size2)})
        if not os.path.exists(self.tmp_dump_file):
            with open(self.tmp_dump_file, 'wb') as f:
                pickle.dump(self.sample_result, f)

        # with open('logs/grasp_results/grab_Obj_hulls.pkl', 'wb') as f:
        #     pickle.dump(self.obj_hulls, f)
        # summary_metrics = defaultdict(list)
        # for m in self.test_metrics:
        #     for k, v in m.items():
        #         summary_metrics[k].append(v)
        # for k, v in summary_metrics.items():
        #     summary_metrics[k] = sum(v) / len(v)
        # self.logger.experiment.log(summary_metrics)
        # for k, v in self.result_dict.items():
        #     self.result_dict[k] = np.concatenate(v, axis=0)
        # res_df = pd.DataFrame(self.result_dict)
        # res_df.to_csv(os.path.join('data', 'disp_raw_data', self.cfg.model.name + '_' + self.cfg.data.dataset + '.csv'))
        self.pool.close()
        self.pool.join()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def loss_net(self, dorig, drec, disp_weight):
        device = dorig['verts_object'].device
        dtype = dorig['verts_object'].dtype
        batch_size = dorig['verts_object'].shape[0]

        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([batch_size, self.model_cfg.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([batch_size, self.model_cfg.latentD]), requires_grad=False).to(device).type(dtype))

        q_z_contact = torch.distributions.normal.Normal(drec['mean_contact'], drec['std_contact'])
        loss_kl_contact = torch.sum(torch.sum(torch.distributions.kl.kl_divergence(q_z_contact, p_z), dim=[1]) * disp_weight) / torch.sum(disp_weight)

        q_z_part = torch.distributions.normal.Normal(drec['mean_part'], drec['std_part'])
        loss_kl_part = torch.sum(torch.sum(torch.distributions.kl.kl_divergence(q_z_part, p_z), dim=[1]) * disp_weight) / torch.sum(disp_weight)

        q_z_pressure = torch.distributions.normal.Normal(drec['mean_pressure'], drec['std_pressure'])
        loss_kl_pressure = torch.sum(torch.sum(torch.distributions.kl.kl_divergence(q_z_pressure, p_z), dim=[1]) * disp_weight) / torch.sum(disp_weight)

        if self.model_cfg.robustkl:
            loss_kl_contact = torch.sqrt(1 + loss_kl_contact ** 2) - 1
            loss_kl_part = torch.sqrt(1 + loss_kl_part ** 2) - 1
            loss_kl_pressure = torch.sqrt(1 + loss_kl_pressure ** 2) - 1

        loss_dict = {'loss_kl_contact': loss_kl_contact,
                     'loss_kl_part': loss_kl_part,
                     'loss_kl_pressure': loss_kl_pressure
                     }

        loss_kl_contact = loss_kl_contact * self.weight_kl
        loss_kl_part = loss_kl_part * self.weight_kl
        loss_kl_pressure = loss_kl_pressure * self.weight_kl

        target_contact = dorig['contacts_object'].to(device).squeeze(dim=-1)
        weight = 1. + 5. * target_contact
        contact_obj_sub = target_contact - drec['contacts_object'].squeeze(dim=-1)
        contact_obj_weighted = contact_obj_sub * weight

        loss_contact_rec = F.l1_loss(contact_obj_weighted,
                                     torch.zeros_like(contact_obj_weighted, device=target_contact.device,
                                                      dtype=target_contact.dtype), reduction='none')
        loss_contact_rec = self.weight_rec * torch.sum(torch.mean(loss_contact_rec, dim=-1) * disp_weight) / torch.sum(disp_weight)

        target_part = dorig['partition_object'].argmax(dim=-1).to(device)
        loss_part_rec = F.nll_loss(input=F.log_softmax(drec['partition_object'], dim=-1).float().permute(0, 2, 1),
                                   target=target_part.long(), reduction='none')
        loss_part_rec = self.weight_rec * 0.5 * torch.sum(torch.mean(weight * loss_part_rec, dim=-1) * disp_weight) / torch.sum(disp_weight)

        weight_pres = 1. + 5. * target_contact
        target_pressure = torch.sum(dorig['pressure_object'].to(device)* self.mid_pts.to(self.device).view(1, 1, -1), dim=-1)
        ## use soft-argmax with beta=50 for such a regression problem.
        remapped_pressure = torch.sum(F.softmax(drec['pressure_object'] * 50, dim=-1) * self.mid_pts.to(self.device).view(1, 1, -1), dim=-1)

        # loss_pressure_rec = F.nll_loss(input=F.log_softmax(drec['pressure_object'], dim=-1).float().permute(0, 2, 1),
        #                                target=target_pressure.long(), reduction='none')
        # pressure_obj_sub = target_pressure - drec['pressure_object'].squeeze(dim=-1)
        loss_pressure_rec = F.l1_loss(remapped_pressure, target_pressure, reduction='none')
        loss_pressure_rec = self.weight_rec * torch.sum(torch.sum(weight_pres * loss_pressure_rec, dim=-1) / weight_pres.sum(dim=-1) * disp_weight) / torch.sum(disp_weight)

        loss_dict.update({'loss_contact_rec': loss_contact_rec,
                          'loss_part_rec': loss_part_rec,
                          'loss_pressure_rec': loss_pressure_rec
                          })

        # masked_pressure = drec['contacts_object'].squeeze(dim=-1) * remapped_pressure
        soft_part_vec = F.softmax(drec['partition_object'], dim=-1)
        soft_part_mask = drec['contacts_object'] * soft_part_vec
        # masked_pressure = (target_contact > self.contact_th) * remapped_pressure
        stable_loss = self.weight_stable * torch.sum(calc_stable_loss(obj_verts=dorig['verts_object'], obj_normal=dorig['normals_object'],
                                                                      masked_pressure=drec['contacts_object'].squeeze(-1) * remapped_pressure,
                                                                      gravity_direction=dorig['gravity_direction']) * disp_weight) / torch.sum(disp_weight)
        # part_cl = self.weight_part_cluster * torch.sum(part_cluster_loss(dorig['verts_object'], soft_part_mask)  * disp_weight) / torch.sum(disp_weight)

        loss_dict['loss_stable'] = stable_loss
        # loss_dict['loss_part_cluster'] = part_cl
        loss_total = loss_kl_contact + loss_kl_part + loss_contact_rec + loss_part_rec + loss_kl_pressure + loss_pressure_rec + stable_loss
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    @staticmethod
    def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
        return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
