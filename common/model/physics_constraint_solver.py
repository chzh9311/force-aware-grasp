import numpy as np
import torch
import trimesh

from scipy.optimize import linprog
from scipy.sparse.csc import csc_matrix
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from qpsolvers import solve_qp

from common.utils.geometry import get_perpend_vecs_tensor,get_perpend_vecs, calc_contacts, get_seperate_contact_maps
from common.utils.manolayer import ManoLayer
from common.dataset_utils.arctic_objects import ObjectTensors
from common.utils.utils import mano_array2dict


class PhysicsConstraintSolver:
    def __init__(self, pcs_cfg, mano_path):
        self.force_th = pcs_cfg.force_th
        self.mu = pcs_cfg.mu
        self.a_epsilon = pcs_cfg.a_epsilon
        self.contact_dist = pcs_cfg.contact_dist
        self.contact_th = pcs_cfg.contact_th
        self.mano_layers = ManoLayer(mano_path)
        self.object_tensors = ObjectTensors()

    def balance_residual(self, obj_verts: torch.tensor, obj_normals: torch.tensor, obj_com: torch.tensor, pressure: torch.tensor):
        """
        obj_verts: B x N x 3
        obj_normals: B x N x 3
        obj_com: B x 3
        pressure: B x N
        """
        NF = - torch.sum(obj_normals * pressure, dim=1) # B x 3
        B, T = get_perpend_vecs_tensor(obj_normals, obj_normals.device) # B x N x 3
        diagF = torch.diag_embed(pressure, dim1=1, dim2=2) # B x N x N
        BF = B.transpose(-1, -2) @ diagF
        TF = T.transpose(-1, -2) @ diagF

        ## Calculate the torque stability
        ls = obj_verts - obj_com.unsqueeze(-1)
        NFp = torch.cross(ls, NF, axis=-1)
        BFp = torch.cross(ls, BF, axis=-1)
        TFp = torch.cross(ls, TF, axis=-1)

        ## How to make sure [0, 0, 0] is in this manifold?
        ## Using PCA?

        mg = torch.tensor([[0], [0], [-9.8]])

    def lp_gradient_th(self, lh_params: torch.Tensor, rh_params: torch.Tensor, obj_rot_aa: torch.Tensor, obj_trans: torch.Tensor,
                    obj_pts: torch.Tensor, obj_pts_normal: torch.Tensor, coms: torch.Tensor, device: torch.device):
        """
        The linear programming forward pass. Will produce gradients.
        :param lh_params, rh_params: batch_size x 64 -> the hand parameter vectors. The target hand has requires_grad=True.
        :param obj_params: {'rot_aa', 'trans'} the dict to provide reference rotation for the hand.
        :param obj_pts: batch_size x N_samples x 3 -> sampled points on the object.
        :param obj_pts_normal: batch_size x N_samples x 3 -> normals of the sampled points.
        :param coms: batch_size x 3 -> the co-ordinates of the object center of mass.
        :param device: cpu or cuda
        :return: forces proportional to the object mass
        ! This function has to be called with torch.enable_grad()
        """
        self.mano_layers.to(device)
        self.object_tensors.to(device)
        lh_param_dict = mano_array2dict(lh_params, rot_type='6d')
        rh_param_dict = mano_array2dict(rh_params, rot_type='6d')
        lh_vs, _, lh_f = self.mano_layers.rel_mano_forward(lh_param_dict, is_right=False,
                                                           ref_rot=obj_rot_aa, ref_trans=obj_trans)
        rh_vs, _, rh_f = self.mano_layers.rel_mano_forward(rh_param_dict, is_right=True,
                                                           ref_rot=obj_rot_aa, ref_trans=obj_trans)
        # obj_out = self.object_tensors(obj_params['arti'], obj_params['rot_aa'],
        #                               obj_params['trans'], obj_params['object_name'])
        # o_vs, o_fs = obj_out['v'], obj_out['f']
        bs = lh_vs.shape[0]
        lh_vs_ng, rh_vs_ng = lh_vs.detach().cpu().numpy(), rh_vs.detach().cpu().numpy()
        ## These data are used to index the vertices to update.
        contact_data = {'lh_contacts': [], 'rh_contacts': [], 'lh_nn': [], 'rh_nn': []}
        obj_pts_np = obj_pts.detach().cpu().numpy()
        for i in range(bs):
            ## Use the nearest point to approximate.
            # left hand
            lh_mesh = trimesh.Trimesh(vertices=lh_vs_ng[i], faces=self.mano_layers.mano_f['left'])
            # right hand
            rh_mesh = trimesh.Trimesh(vertices=rh_vs_ng[i], faces=self.mano_layers.mano_f['right'])
            lh_contact, lh_nn, rh_contact, rh_nn = calc_contacts(obj_pts_np[i], lh_mesh, rh_mesh, self.contact_dist)
            lh_contact, rh_contact = get_seperate_contact_maps(lh_contact, rh_contact, self.contact_th)
            contact_data['lh_contacts'].append(lh_contact)
            contact_data['rh_contacts'].append(rh_contact)
            contact_data['lh_nn'].append(lh_nn)
            contact_data['rh_nn'].append(rh_nn)
            # sln = self.physics_stable_sln(all_contacts, obj_pts, obj_pts_normal, coms)

        lh_contacts = np.stack(contact_data['lh_contacts'], axis=0) > 0
        rh_contacts = np.stack(contact_data['rh_contacts'], axis=0) > 0
        lh_nns = torch.stack([lh_vs[i, contact_data['lh_nn'][i]] for i in range(bs)], dim=0)
        rh_nns = torch.stack([rh_vs[i, contact_data['rh_nn'][i]] for i in range(bs)], dim=0)
        all_contacts = torch.from_numpy(np.clip(lh_contacts + rh_contacts, 0, 1)).to(device)
        obj_pts_tensor = obj_pts.requires_grad_()

        ## back propagate the gradients to sampled object points
        ## Only points with contact=1 will get a gradient
        contact_grad, pts_grad = self.physics_stable_slns_grad(all_contacts, obj_pts_tensor, obj_pts_normal, coms)
        ## Project the gradients to the object surface: ## contact_grad: bs x N x 3; normals: bs x N x 3
        pts_grad -= torch.sum(pts_grad * obj_pts_normal, dim=-1).unsqueeze(-1) * obj_pts_normal
        ## Assign the gradients to hand vertices, use autograd to calculate the gradients
        tmp_loss = torch.sum(lh_nns[lh_contacts] * pts_grad[lh_contacts]) \
                   + torch.sum(rh_nns[rh_contacts] * pts_grad[rh_contacts])
        tmp_loss.backward()
        return

    def conatct_lp_update(self, contacts, obj_pts, obj_pts_normal, coms, step, device):
        """
        :param contacts: The initial contacts
        """
        with torch.enable_grad():
            contact_grad, _ = self.physics_stable_slns_grad(contacts, obj_pts, obj_pts_normal, coms)
        contacts = torch.clip(contacts + step * contact_grad, 0, 1)
        return contacts

    def physics_stable_sln(self, contacts: np.array, obj_pts: np.array,
                           obj_normals: np.array, coms: np.array):
        """
        :param contacts: (N) the binary contact values of N sampled points;
        :param obj_pts: (N x 3) the point positions of all N sampled points.
        :param obj_normals: (N x 3) the point norms of all N sampled points.
        :param coms: (3) the centre of mass of the object
        (N + B * diag{\beta} + T * diag{\gamma}) F = m(a - g);
        """
        ## Calculate the force stability
        ## The contact force always points to the inside.
        N = -obj_normals * np.expand_dims(contacts, axis=-1)
        B, T = get_perpend_vecs(obj_normals)
        B *= np.expand_dims(contacts, axis=-1)
        T *= np.expand_dims(contacts, axis=-1)

        Fub, Flb = self.calc_ub_lb_np(N, B, T, self.mu)

        ## Calculate the torque stability
        ls = obj_pts - np.expand_dims(coms, axis=-2)
        Np = np.cross(ls, N, axis=-1)
        Bp = np.cross(ls, B, axis=-1)
        Tp = np.cross(ls, T, axis=-1)
        Mub, Mlb = self.calc_ub_lb_np(Np, Bp, Tp, self.mu)

        avg_acc = np.array([0, 0, -9.8, 0, 0, 0])
        ## We want the total force to be minimized. i.e., min sum(F)
        ## Thus become 6 linear programming problems, one in each dimension.
        N = Fub.shape[0]
        c = np.ones(N)
        Aub = np.concatenate((Flb, Mlb, -Fub, -Mub), axis=-1).T  # 12 x N
        bub = np.concatenate((avg_acc - 1, -avg_acc - 1), axis=-1)  # (12,)
        result = linprog(c, A_ub=Aub, b_ub=bub, bounds=(0, None))

        return result

    def diff_stable_loss(self, obj_sdf, obj_pts, obj_normals, coms, in_tolerance, obj_contact_target, hand_contact_target):
        """
        obj_sdf: Batch size x N x 1
        obj_normals: Batch size x N x 3
        obj_normals: Batch size x N x 3
        coms: Batch size x 3
        """
        N = -obj_normals # * contacts.unsqueeze(-1)
        B, T = get_perpend_vecs_tensor(obj_normals, obj_normals.device) # B x N x 3
        # B *= contacts.unsqueeze(-1)
        # T *= contacts.unsqueeze(-1)
        F = torch.zeros_like(obj_sdf, device=obj_sdf.device)
        F[obj_sdf < in_tolerance] = 1 / (in_tolerance + 0.005 - obj_sdf[obj_sdf < in_tolerance])
        F *= (obj_contact_target * hand_contact_target).unsqueeze(-1)
        ## Normalize F
        F = F / torch.norm(F, dim=1, keepdim=True)
        NF = N.permute(0, 2, 1) @ F
        BLamF = (B * F).permute(0, 2, 1)
        TLamF = (T * F).permute(0, 2, 1)
        Fub, Flb = self.calc_ub_lb(NF.squeeze(-1), BLamF, TLamF, self.mu, sumbt=True) # B x 3

        gravity = torch.tensor([[0, 0, -9.8]], device=obj_sdf.device)
        amax = Fub + gravity
        amin = Flb + gravity
        asum = torch.cat((-amax, amin), dim=-1)
        loss1 = torch.mean(torch.max(asum, dim=-1)[0]) + torch.mean(asum[asum > 0])

        ls = obj_pts - coms.unsqueeze(1)
        Np = torch.cross(ls, N, dim=-1)
        NpF = Np.permute(0, 2, 1) @ F
        BLamFp = torch.cross(ls, BLamF.permute(0, 2, 1), dim=-1).permute(0, 2, 1)
        TLamFp = torch.cross(ls, TLamF.permute(0, 2, 1), dim=-1).permute(0, 2, 1)
        Mub, Mlb = self.calc_ub_lb(NpF.squeeze(-1), BLamFp, TLamFp, self.mu, sumbt=True) # B x 3
        alpha_max = Mub
        alpha_min = Mlb
        alpha_sum = torch.cat((-alpha_max, alpha_min), dim=-1)
        loss2 = torch.mean(torch.max(alpha_sum, dim=-1)[0]) + torch.mean(alpha_sum[alpha_sum > 0])

        return loss1 + loss2, F.squeeze(-1)


    def lp_physics_stable_loss(self, contacts: torch.Tensor, obj_pts: torch.Tensor,
                               obj_normals: torch.Tensor, coms: torch.Tensor,
                               obj_contact_target: torch.Tensor, hand_contact_target: torch.Tensor):
        bs, N = obj_pts.shape[:2]
        Aub, bub = self.get_constraint_matrices(contacts, obj_pts, obj_normals, coms, obj_contact_target, hand_contact_target)
        Aub_np = Aub.detach().cpu().numpy()
        c = np.ones(N)
        losses = []
        Fs = []
        for i in range(bs):
            result = linprog(c, A_ub=Aub_np[i], b_ub=bub, bounds=(0, None), method='highs')
            # slns.append(results)
            # The gravity does not affect the gradient.
            F = torch.from_numpy(result.x).unsqueeze(-1).to(Aub.device).float()
            boundries = Aub[i, result.slack < self.force_th]
            FOpts = F < self.force_th
            losses.append(torch.sum(boundries @ F)) # The loss is set so that this becomes larger, and therefore F becomes smaller.
            Fs.append(F.squeeze())
        losses = torch.stack(losses, dim=0)
        Fs = (torch.stack(Fs, dim=0))
        real_Fs = Fs * contacts # * obj_contact_target * hand_contact_target
        return losses, Fs, real_Fs

    def qp_force_estimate(self, target_contacts: torch.Tensor, obj_pts: torch.Tensor,
                               obj_normals: torch.Tensor, coms: torch.Tensor):
        bs, N = obj_pts.shape[:2]
        Aub, bub = self.get_constraint_matrices(target_contacts, obj_pts, obj_normals, coms)
        Aub_np = Aub.detach().cpu().numpy()
        P = np.eye(N)
        P = csc_matrix(P)
        q = np.zeros(N)
        ## multiprocessing
        if bs > 1:
            pool = ThreadPool(min(bs, mp.cpu_count() - 2))
            mat_dict_list = [{'P': P, 'q': q, 'Aub': csc_matrix(Aub_np[i]), 'bub': bub, 'lb': np.zeros(N)} for i in range(bs)]
            F_out = pool.map(self.solve_force_qp, mat_dict_list)
        else:
            F_out = [solve_qp(P, q, Aub_np[0], bub, A=None, lb=np.zeros(N), solver='clarabel')]
        Fs = torch.from_numpy(np.stack(F_out, axis=0)).float().to(target_contacts.device)
        real_Fs = Fs * target_contacts # * obj_contact_target * hand_contact_target
        return real_Fs

    def qp_physics_stable_loss(self, contacts: torch.Tensor, obj_pts: torch.Tensor,
                               obj_normals: torch.Tensor, coms: torch.Tensor,
                               obj_contact_target: torch.Tensor, hand_contact_target: torch.Tensor):
        bs, N = obj_pts.shape[:2]
        Aub, bub = self.get_constraint_matrices(contacts, obj_pts, obj_normals, coms, obj_contact_target, hand_contact_target)
        Aub_np = Aub.detach().cpu().numpy()
        P = np.eye(N)
        P = csc_matrix(P)
        q = np.zeros(N)
        losses = []
        Fs = []
        ## multiprocessing
        if bs > 1:
            pool = ThreadPool(min(bs, mp.cpu_count() - 2))
            mat_dict_list = [{'P': P, 'q': q, 'Aub': csc_matrix(Aub_np[i]), 'bub': bub, 'lb': np.zeros(N)} for i in range(bs)]
            F_out = pool.map(self.solve_force_qp, mat_dict_list)
        else:
            F_out = [solve_qp(P, q, Aub_np[0], bub, A=None, lb=np.zeros(N), solver='clarabel')]
        for i in range(bs):
            F = F_out[i]
            # F = solve_qp(P, q, Aub_np[i], bub, A=None, lb=np.zeros(N), solver='clarabel')
            if F is not None:
                boundries = Aub[i, bub - Aub_np[i] @ F < self.force_th]
                F = torch.from_numpy(F).unsqueeze(-1).to(Aub.device).float()
                losses.append(torch.sum(boundries @ F)) # The loss is set so that this becomes larger, and therefore F becomes smaller.
                Fs.append(F.squeeze())
            else:
                losses.append(torch.tensor(0, device=obj_pts.device))
                Fs.append(torch.zeros(N, device=obj_pts.device))
        losses = torch.stack(losses, dim=0)
        Fs = (torch.stack(Fs, dim=0))
        real_Fs = Fs * contacts # * obj_contact_target * hand_contact_target
        return losses, Fs, real_Fs

    def diff_qp_stable_loss(self, contacts: torch.Tensor, obj_pts: torch.Tensor,
                            obj_normals: torch.Tensor, coms: torch.Tensor,
                            obj_contact_target: torch.Tensor, hand_contact_target: torch.Tensor):
        bs, N = obj_pts.shape[:2]
        Aub, bub = self.get_constraint_matrices(contacts, obj_pts, obj_normals, coms, obj_contact_target, hand_contact_target)
        Aub_np = Aub.detach().cpu().numpy()
        P = np.eye(N)
        P = csc_matrix(P)
        q = np.zeros(N)
        losses = []
        Fs = []
        ## multiprocessing
        losses = torch.stack(losses, dim=0)
        Fs = (torch.stack(Fs, dim=0))
        real_Fs = Fs * contacts # * obj_contact_target * hand_contact_target
        return losses, Fs, real_Fs

    def solve_force_qp(self, mats: dict):
        P, q, Aub, bub, lb = mats['P'], mats['q'], mats['Aub'], mats['bub'], mats['lb']
        F = solve_qp(P, q, Aub, bub, A=None, lb=lb, solver='clarabel')
        return F

    def get_constraint_matrices(self, contacts: torch.Tensor, obj_pts: torch.Tensor,
                               obj_normals: torch.Tensor, coms: torch.Tensor,
                               obj_contact_target: torch.Tensor | None=None, hand_contact_target: torch.Tensor|None=None):

        ## Calculate the force stability
        ## The contact force always points to the inside.
        # contacts = contacts  * obj_contact_target * hand_contact_target
        contacts = contacts
        N = -obj_normals * contacts.unsqueeze(-1)
        B, T = get_perpend_vecs_tensor(obj_normals, obj_normals.device)
        B *= contacts.unsqueeze(-1)
        T *= contacts.unsqueeze(-1)

        Fub, Flb = self.calc_ub_lb(N, B, T, self.mu)

        ## Calculate the torque stability
        ls = obj_pts - coms.unsqueeze(1)
        Np = torch.cross(ls, N, dim=-1)
        Bp = torch.cross(ls, B, dim=-1)
        Tp = torch.cross(ls, T, dim=-1)
        Mub, Mlb = self.calc_ub_lb(Np, Bp, Tp, self.mu)

        avg_acc = torch.Tensor([0, 0, -9.8, 0, 0, 0])
        ## We want the total force to be minimized. i.e., min sum(F)
        ## Thus become 6 linear programming problems, one in each dimension.
        ## Move variables to cpu
        # Fub_np, Flb_np, Mub_np, Mlb_np = (Fub.detach().cpu().numpy(), Flb.detach().cpu().numpy(),
        #                       Mub.detach().cpu().numpy(), Mlb.detach().cpu().numpy())
        Aub = torch.cat((Flb, Mlb, -Fub, -Mub), dim=-1).permute(0, 2, 1) # bs x 12 x N
        bub = np.concatenate((avg_acc - self.a_epsilon, -avg_acc - self.a_epsilon), axis=-1) # (12,)
        return Aub, bub


    def physics_stable_slns_grad(self, contacts: torch.Tensor, obj_pts: torch.Tensor,
                                 obj_normals: torch.Tensor, coms: torch.Tensor) -> list:
        """
        :param contacts: (batch_size x N) the contact values of N sampled points between 0 and 1;
        :param obj_pts: (batch_size x N x 3) the point positions of all N sampled points.
        :param obj_normals: (batch_size x N x 3) the point norms of all N sampled points.
        :param coms: (batch_size x 3) the centre of mass of the object
        :return: the grad of contact points; the
        (N + B * diag{\beta} + T * diag{\gamma}) F = m(a - g);
        """
        contacts.requires_grad_()
        obj_pts.requires_grad_()
        loss = self.physics_stable_loss(contacts, obj_pts, obj_normals, coms)
        loss.backward()
        obj_pts_grad = obj_pts.grad
        contact_grad = contacts.grad
        # Fs = torch.stack(Fs, dim=0)
        obj_pts.requires_grad_(False)
        contacts.requires_grad_(False)

        return contact_grad, obj_pts_grad


    def calc_ub_lb(self, N: torch.Tensor, B: torch.Tensor, T: torch.Tensor, mu: float, sumbt=False):
        """
        Calculate the upper & lower bound of the Newton's Law II equations
        """
        quantB = torch.zeros_like(B, device=B.device)
        quantB[B > 0] = mu
        quantB[B < 0] = -mu

        quantT = torch.zeros_like(T, device=T.device)
        quantT[T > 0] = mu
        quantT[T < 0] = -mu

        BqB = B * quantB
        TqT = T * quantT
        if sumbt:
            BqB = torch.sum(BqB, dim=-1)
            TqT = torch.sum(TqT, dim=-1)

        ub = N + BqB + TqT
        lb = N - BqB - TqT
        return ub, lb

    def calc_ub_lb_np(self, N: np.ndarray, B: np.ndarray, T: np.ndarray, mu: float):
        """
        Calculate the upper & lower bound of the Newton's Law II equations
        """
        quantB = np.zeros_like(B)
        quantB[B > 0] = 1
        quantB[B < 0] = -1

        quantT = np.zeros_like(T)
        quantT[T > 0] = 1
        quantT[T < 0] = -1

        ub = N + B * quantB * mu + T * quantT * mu
        lb = N - B * quantB * mu - T * quantT * mu
        return ub, lb


