import os
import pickle
import numpy as np
import torch
from alive_progress import alive_bar
from torch.utils.data import Dataset
from common.utils.utils import sample_slice_from_gap
from common.dataset_utils.oakink2_toolkit.dataset import OakInk2__Dataset
from pytorch3d.transforms import quaternion_to_matrix

class OakInk2SegDataset(Dataset):

    def __init__(self, root, split):
        with open(os.path.join(root, "asset", "split", split+'.txt'), 'r') as spf:
            process_dirs = spf.read()
        self.obj_w_mesh = os.listdir(os.path.join(root, "object_raw", "align_ds"))
        self.process_range_list = process_dirs.split('\n')
        self.dataset = OakInk2__Dataset(dataset_prefix=root, return_instantiated=True)
        self.obj_meshes = self.load_obj_metadata(os.path.join('data', 'misc', 'oakink2_obj_mesh_data.pkl'))
        self.target_gap = 12
        self.slice_max_len = 160
        self.slice_min_len = 16
        (
            self.interaction_segment_info_list,
            self.interaction_segment_len_list,
            self.interaction_segment_pose_list,
            self.interaction_segment_tsl_list,
            self.interaction_segment_shape_list,
            self.interaction_segment_text_list,
            self.interaction_segment_obj_traj_list,
            self.interaction_segment_frame_id_list,
            self.interaction_object_list,
        ) = self.load_dataset()

    def collect_obj(self, primitive_task_data, src_obj_list, seg_beg, seg_end, task_beg):
        seg_obj_traj_store = {}
        for obj_id in src_obj_list:
            _off_beg, _off_end = seg_beg - task_beg, seg_end - task_beg
            seg_obj_traj_store[obj_id] = primitive_task_data.obj_transf[obj_id][_off_beg:_off_end].astype(np.float32)
        return seg_obj_traj_store

    def load_obj_metadata(self, save_path=None):
        if save_path is None or not os.path.exists(save_path):
            res = {}
            for obj in self.obj_w_mesh:
                ## all object ids
                affordance = self.dataset.load_affordance(obj)
                v = affordance.obj_mesh.vertices
                f = affordance.obj_mesh.faces
                res[obj] = {"vertices":v, "faces":f}

            if save_path is not None:
                with open(save_path, "wb") as of:
                    pickle.dump(res, of)
        else:
            with open(save_path, "rb") as of:
                res = pickle.load(of)

        return res

    def __len__(self):
        return len(self.interaction_segment_info_list)

    def __getitem__(self, idx):
        return (
            self.interaction_segment_pose_list[idx], self.interaction_segment_tsl_list[idx],
            self.interaction_segment_shape_list[idx], self.interaction_segment_obj_traj_list[idx],
            self.interaction_segment_info_list[idx],  self.interaction_segment_text_list[idx] )

    def collect_bh_mano(self, primitive_task_data, seg_beg, seg_end, task_beg):
        outs = {}
        in_range_mask = torch.logical_and(primitive_task_data[f"lh_in_range_mask"], primitive_task_data[f"rh_in_range_mask"])
        for hand_side in ['lh', 'rh']:
            outs[hand_side] = self.collect_mano(primitive_task_data, hand_side, in_range_mask)
        assert torch.sum(in_range_mask) == seg_end - seg_beg

        seg_mano_pose_traj, seg_mano_tsl_traj, seg_mano_shape_traj = [
            np.stack((outs['lh'][i], outs['rh'][i]), axis=1) for i in range(3)]
        return seg_mano_pose_traj, seg_mano_tsl_traj, seg_mano_shape_traj

    def collect_mano(self, primitive_task_data, hand_side, in_range_mask):
        _pose = primitive_task_data[f"{hand_side}_param"]["pose_coeffs"][in_range_mask]
        _tsl = primitive_task_data[f"{hand_side}_param"]["tsl"][in_range_mask]
        _shape = primitive_task_data[f"{hand_side}_param"]["betas"][in_range_mask]
        seg_mano_pose_traj = _pose
        seg_mano_tsl_traj = _tsl
        seg_mano_shape_traj = _shape
        seg_mano_pose_traj = quaternion_to_matrix(seg_mano_pose_traj)
        seg_mano_pose_traj = seg_mano_pose_traj.numpy().astype(np.float32)
        seg_mano_tsl_traj = seg_mano_tsl_traj.numpy().astype(np.float32)
        seg_mano_shape_traj = seg_mano_shape_traj.numpy().astype(np.float32)
        return seg_mano_pose_traj, seg_mano_tsl_traj, seg_mano_shape_traj

    def load_dataset(self):
        interaction_info_list = []
        interaction_len_list = []
        interaction_pose_list = []
        interaction_tsl_list = []
        interaction_shape_list = []
        interaction_text_list = []
        interaction_obj_traj_list = []
        interaction_frame_id_list = []
        object_list = set()
        with alive_bar(len(self.process_range_list)) as bar:
            for process_key in self.process_range_list:
                complex_task_data = self.dataset.load_complex_task(seq_key=process_key)
                primitive_task_data_list = self.dataset.load_primitive_task(complex_task_data=complex_task_data)

                for primitive_identifier, primitive_task_data in zip(complex_task_data.exec_path, primitive_task_data_list):
                    task_beg = primitive_task_data.frame_range[0]

                    if primitive_task_data.hand_involved == 'bh':
                        # Only include primitives w/ both hands.

                        seg_beg = {}
                        seg_end = {}
                        seg_info = (process_key, primitive_identifier)
                        for hand_side in ['lh', 'rh']:
                            text_desc = primitive_task_data.task_desc
                            seg_beg[hand_side], seg_end[hand_side] = primitive_task_data[f"frame_range_{hand_side}"]

                            src_obj_list = primitive_task_data[f"{hand_side}_obj_list"]
                            if len(src_obj_list) == 0:
                                continue

                            # collect obj
                            object_list.update(src_obj_list)
                            # collect mano

                        seg_beg = max(seg_beg['lh'], seg_beg['rh'])
                        seg_end = min(seg_end['lh'], seg_end['rh'])
                        seg_len = seg_end - seg_beg

                        seg_mano_pose_traj, seg_mano_tsl_traj, seg_mano_shape_traj = self.collect_bh_mano(
                            primitive_task_data, seg_beg, seg_end, task_beg)
                        seg_obj_traj_store = self.collect_obj(primitive_task_data, src_obj_list, seg_beg, seg_end, task_beg)

                        seg_mano_pose_traj_sliced, seg_len_sliced = sample_slice_from_gap(
                            seg_mano_pose_traj, self.target_gap, self.slice_max_len, self.slice_min_len)
                        seg_mano_tsl_traj_sliced, _ = sample_slice_from_gap(seg_mano_tsl_traj, self.target_gap,
                                                                            self.slice_max_len, self.slice_min_len)
                        seg_mano_shape_traj_sliced, _ = sample_slice_from_gap(seg_mano_shape_traj, self.target_gap,
                                                                              self.slice_max_len, self.slice_min_len)
                        # Object trajectory
                        seg_obj_traj_store_sliced = {}
                        for obj_id in src_obj_list:
                            seg_obj_traj_store_sliced[obj_id], _ = sample_slice_from_gap(
                                seg_obj_traj_store[obj_id], self.target_gap, self.slice_max_len, self.slice_min_len)
                        seg_obj_traj_list_sliced = []
                        for _offset in range(len(seg_len_sliced)):
                            _item = {}
                            for obj_id in src_obj_list:
                                _item[obj_id] = seg_obj_traj_store_sliced[obj_id][_offset]
                            seg_obj_traj_list_sliced.append(_item)
                        # frame_id
                        seg_frame_id_list = np.array(list(range(seg_beg, seg_end)))
                        seg_frame_id_list_sliced, _ = sample_slice_from_gap(seg_frame_id_list, self.target_gap,
                                                                            self.slice_max_len, self.slice_min_len)
                        _seg_frame_id_list_sliced = []
                        for _len, _fid_list in zip(seg_len_sliced, seg_frame_id_list_sliced):
                            _new_list = _fid_list[:_len].tolist()
                            _seg_frame_id_list_sliced.append(_new_list)
                        seg_frame_id_list_sliced = _seg_frame_id_list_sliced
                        # extend storage list
                        interaction_info_list.extend([seg_info] * len(seg_len_sliced))
                        interaction_len_list.extend(seg_len_sliced)
                        interaction_pose_list.extend(seg_mano_pose_traj_sliced)
                        interaction_tsl_list.extend(seg_mano_tsl_traj_sliced)
                        interaction_shape_list.extend(seg_mano_shape_traj_sliced)
                        interaction_text_list.extend([text_desc] * len(seg_len_sliced))
                        interaction_obj_traj_list.extend(seg_obj_traj_list_sliced)
                        interaction_frame_id_list.extend(seg_frame_id_list_sliced)
                bar()
                break

        interaction_object_list = sorted(object_list)

        # debug
        # print(len(interaction_segment_len_list), interaction_segment_len_list[:10])
        # print(len(interaction_segment_hand_side_list), interaction_segment_len_list[:10])
        # print(len(interaction_segment_obj_traj_list))
        # for el in interaction_segment_tsl_list[:10]:
        #     print(el.shape)

        return (
            interaction_info_list,
            interaction_len_list,
            interaction_pose_list,
            interaction_tsl_list,
            interaction_shape_list,
            interaction_text_list,
            interaction_obj_traj_list,
            interaction_frame_id_list,
            interaction_object_list,
        )
