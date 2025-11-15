import os.path as osp
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from alive_progress import alive_bar
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from common.dataset_utils.grab_dataset import key2part_id
from common.utils.manolayer import ManoLayer
from common.utils.utils import force_padding_collate_fn
from common.utils.geometry import axisangle2matrix, get_v2v_rot, normalize_vec
from common.model.hand_object import HandObject
from common.model.pose_optimizer import phyScore
from common.simulation.mujoco_hand_object_simulator import kine_tree_w_tips
from common.model.losses import calc_stable_loss


class StatisticsAnalyser:
    def __init__(self, dataset):
        self.manolayer = ManoLayer('data/mano_v1_2', flat_hand_mean=True)
        self.dataset = dataset
        dataloader = DataLoader(self.dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=force_padding_collate_fn)
        self.contact_pressure = defaultdict(list)
        ## objMass: float; partContacts: 16 x 1 in {0, 1}, indicating the part is in contact;
        ## MANOPose: 45; partPressure: 16 x 3, sum of the support force from one hand part.
        self.pressure = []
        self.contacts = []
        # self.hand_joints = []
        self.all_contact_pressure = []
        self.full_pose = []
        self.disps = []
        self.hand_rot = []
        self.obj_name = []
        self.part_contact_force = []
        self.stability_score = []
        self.stability_loss = []
        self.pressure_quantise_splits = np.load(osp.join('logs', 'simulations', 'grab_v1', 'separations1.npy'))
        with alive_bar(len(dataloader)) as bar:
            bar.title('Reading data...')
            for idx, batch in enumerate(dataloader):
                # if idx > 10:
                #     break
                handobject = HandObject(device='cpu', hand_faces=self.manolayer.mano_f['right'], hand_part_ids=self.manolayer.part_id_right, pressure_quantise_splits=self.pressure_quantise_splits, contact_th=0.22)
                handobject.load_from_batch(batch, inv_obj_rot=True)
                # part_pressure = []
                # part_contacts = []
                # pr = handobject.pressure_map.unsqueeze(-1)  # B x N x 16 x 1
                # normals = handobject.obj_normals  # B x N x 3
                # for i in range(16):
                #     pmap = (handobject.part_map @ torch.arange(16, device=handobject.device).view(1, 16, 1).float())
                #     part_mask = torch.logical_and(pmap == i, handobject.contact_map.unsqueeze(-1) > 0.11) # B x N x 1
                #     part_pressure.append(torch.sum(pr[:, :, i] * normals * part_mask, dim=1).detach().cpu().numpy()) # B x 3
                #     part_contacts.append(torch.sum(part_mask, dim=1).detach().cpu().numpy() > 0) # B x 1

                # self.pressure.append(np.stack(part_pressure, axis=1))
                # self.contacts.append(np.stack(part_contacts, axis=1))
                # self.hand_joints.append(handobject.hand_joints.detach().cpu().numpy())
                self.full_pose.append(torch.cat((batch['handRot'], batch['handPose']), dim=1).detach().cpu().numpy())
                disp = torch.norm(batch['simuDisp'][:, :3], dim=-1)
                self.disps.append(disp.detach().cpu().numpy())
                simu_contacts = batch['simuContacts']
                for contact in simu_contacts:
                    contact_force = defaultdict(float)
                    for c in contact:
                        contact_force[key2part_id[c['part_id']]] += c['force'][0]
                    self.part_contact_force.append(contact_force)
                # # stable_mask = disp < 0.05
                mask = handobject.contact_map > 0.22
                contact_pressure = torch.sum(handobject.pressure_map[mask], dim=-1)
                self.obj_name.append(batch['objName'])
                self.all_contact_pressure.append(contact_pressure.detach().cpu().numpy())

                ## For stability loss
                # contact_mask = handobject.contact_map > 0.11
                # part_mask = contact_mask.unsqueeze(-1) * handobject.part_map
                # part_pres = torch.sum((handobject.pressure_map * part_mask).unsqueeze(-1) * -handobject.obj_normals.unsqueeze(2), dim=1)  # B x 16 x 3
                # part_contact_pt = torch.sum((handobject.pressure_map * part_mask).unsqueeze(-1) * handobject.obj_verts.unsqueeze(2), dim=1) / (
                #         torch.sum(handobject.pressure_map * part_mask, dim=1).unsqueeze(-1) + 1e-8)  # B x 3
                # rs = torch.max(torch.norm(handobject.obj_verts, dim=-1), dim=-1)[0]
                # stable_loss = calc_stable_loss(handobject.obj_verts, handobject.obj_normals, handobject.pressure_map.sum(dim=-1),
                #                                torch.as_tensor([[0], [0], [-1]]).repeat(part_pres.shape[0], 1))
                #
                # for b in range(part_pres.shape[0]):
                #     pm = part_mask[b].sum(dim=0) > 0 # 16
                #     # pm = torch.as_tensor(c, device=handobject.device)
                #     tmp_part_pres = part_pres[b, pm].clone().view(-1, 3)
                #     tmp_contact_pt = part_contact_pt[b, pm].clone().view(-1, 3)
                #     # Fs = part_pres[b, all_contact_parts].clone().view(-1, 3).detach().cpu().numpy()
                #     self.stability_score.append(
                #         phyScore(tmp_part_pres.detach().cpu().numpy(), tmp_contact_pt.detach().cpu().numpy(), radius=rs[b]))
                #     self.stability_loss.append(stable_loss[b].detach().cpu().numpy())

                bar()

        # self.pressure = np.concatenate(self.pressure, axis=0)
        # self.contacts = np.concatenate(self.contacts, axis=0)
        # self.hand_joints = np.concatenate(self.hand_joints, axis=0)
        # self.full_pose = np.concatenate(self.full_pose, axis=0).reshape(-1, 16, 3)
        self.disps = np.concatenate(self.disps, axis=0)
        # self.stability_score = np.array(self.stability_score)
        # self.stability_loss = np.array(self.stability_loss)
        self.all_contact_pressure = np.concatenate(self.all_contact_pressure, axis=0)

        self.fingers = list(kine_tree_w_tips.keys())


    def analyse_stability_labels(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        sns.despine(fig, left=True, bottom=True)
        data = pd.DataFrame({'Log Simulation Displacement': self.disps.repeat(2), 'Loss': np.concatenate([np.sqrt(self.stability_score), self.stability_loss], axis=0),
                             'Losses': ['Sqrt Stability Energy', ]*self.disps.shape[0] + ['Simplified Stability Loss']*self.disps.shape[0]})
        sns.lmplot(data=data, x='Log Simulation Displacement', y='Loss', hue='Losses')
        # ax.scatter(self.disps, np.sqrt(self.stability_score))
        # ax.scatter(self.disps, self.stability_loss)
        plt.show()

    def save_force_statistics(self):
        tmp_fp = osp.join('tmp', 'all_labelled_forces.pkl')
        save_dict = {'disps': self.disps, 'part_contact_force': self.part_contact_force, 'part_pressure': self.all_contact_pressure}
        with open(tmp_fp, 'wb') as f:
            pickle.dump(save_dict, f)

    def save_pressure_quant_splits(self):
        n_splits = 16
        rates = np.arange(1, n_splits) / n_splits
        # mid_pts = np.quantile(self.all_contact_pressure, rates)
        log_pressure = np.log(self.all_contact_pressure[self.all_contact_pressure > 0])
        avg = np.mean(log_pressure)
        std = np.std(log_pressure)
        print(avg, std)
        mid_pts = np.exp(np.linspace(avg - 3 * std, avg + 3 * std, n_splits-1))
        print(mid_pts)
        plt.hist(log_pressure, bins=40)
        plt.show()
        print("Saved pressure quantise splits to ", osp.join('logs', 'simulations', 'grab_v2', 'separations.npy'))
        np.save(osp.join('logs', 'simulations', 'grab_v2', 'separations.npy'), mid_pts)

    def analyse_pose_and_pressure(self, normalise_pressure=True):
        """
        Use the root joint coordinate as the global coordinate system, find the relationship between the RPY of the
        two distal joints and the force direction & value.
        We have totally 6 + 9 variables, and we can try PCA to get the relationships.
        """
        self.finger_data = {}
        stable_mask=self.disps < 0.05

        ## flat hand data
        flat_param = {'rot_aa': torch.zeros(1, 3), 'pose': torch.zeros(1, 45), 'shape': torch.zeros(1, 10),
                      'trans': torch.zeros(1, 3)}
        _, flat_joints, _ = self.manolayer.mesh_data_np(flat_param, is_right=True) # 21 joints
        flat_joints = flat_joints.reshape(21, 3)

        self.pressure = self.pressure[stable_mask]
        self.full_pose = self.full_pose[stable_mask]
        self.contacts = self.contacts[stable_mask]

        global_rot = self.full_pose[:, 0]
        globalR = axisangle2matrix(global_rot)

        if normalise_pressure:
            tp = np.sum(np.linalg.norm(self.pressure, axis=-1), axis=-1)
            pressure = self.pressure / (tp.reshape(-1, 1, 1) + 1e-8)
        else:
            pressure = self.pressure
        pca = PCA()
        for k, v in kine_tree_w_tips.items():
            ## Normalize the pose
            contact_cnt = self.contacts[:, v[1:4]].reshape(-1, 3)
            contact_mask = np.sum(contact_cnt, axis=-1) > 0
            pose = self.full_pose[contact_mask][:, v[1:4]]
            ref_x_axis = normalize_vec(flat_joints[v[1:4]] - flat_joints[v[2:5]]) # point from child to parent.
            x_axis = np.zeros((3, 3))
            x_axis[:, 0] = 1
            ## transform the x-axis from (1, 0, 0) to reference
            ## TODO: validate this via visualization.
            refR = get_v2v_rot(x_axis, ref_x_axis)
            norm_pose = refR.transpose(0, 2, 1).reshape(-1, 3, 3, 3) @ pose.reshape(-1, 3, 3, 1)
            norm_pose = norm_pose.reshape(-1, 3, 3)

            root_rot = norm_pose[:, 0]
            rootR = axisangle2matrix(root_rot)
            ## Normalize the force to the root coordinate
            glob_pres= pressure[contact_mask][:, v[1:4]] # B x 3 x 3
            local_pres = glob_pres @ (rootR @ globalR[contact_mask]).transpose(0, 2, 1) # p_l = R_local @ R_global @ p_g
            data = np.concatenate((norm_pose[:, 1:].reshape(-1, 6)/np.pi, local_pres.reshape(-1, 9)), axis=-1)
            pca.fit(data)
            comp = pca.components_
            var = pca.explained_variance_ratio_
            self.finger_data[k] = [comp, var]

        self.pca_analysis()

    def pca_analysis(self):
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 6, width_ratios=[3, 1, 3, 1, 3, 1], height_ratios=[1, 1])
        idx = 0
        for k, dt in self.finger_data.items():
            comp, var = dt
            hm_ax = fig.add_subplot(gs[int(idx // 3), (idx % 3) * 2])
            hm_ax.set_title(k, loc='left')
            hm_ax_divider = make_axes_locatable(hm_ax)
            cax = hm_ax_divider.append_axes("left", size="5%", pad="2%")
            hm = sns.heatmap(np.abs(comp), cbar=True, ax=hm_ax, cbar_ax=cax, cbar_kws={'location': 'left'}, xticklabels=False, yticklabels=False)

            bar_ax = fig.add_subplot(gs[int(idx // 3), (idx % 3) * 2 + 1])
            sns.barplot(var, ax=bar_ax, orient='y')
            idx += 1

        plt.show()