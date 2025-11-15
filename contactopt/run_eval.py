# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
for p in ['.', '..']:
    sys.path.append(p)
import pickle
from open3d import visualization as o3dv
import random
import argparse
import numpy as np
import time
import contactopt.util as util
import contactopt.geometric_eval as geometric_eval
import pprint
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics
import trimesh
import os

SAVE_OBJ_FOLDER = 'eval/saveobj'


def vis_sample_comp(gt_ho, in_ho, opt_ho, phy_ho, mje_in=None, mje_phy=None, mje_opt=None):
    hand_gt, obj_gt = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in, obj_in = in_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in.translate((0.0, 0.2, 0.0))
    obj_in.translate((0.0, 0.2, 0.0))

    if not args.split == 'honn':
        opt_ho.hand_contact = in_ho.hand_contact
        opt_ho.obj_contact = in_ho.obj_contact
        phy_ho.hand_contact = in_ho.hand_contact
        phy_ho.obj_contact = in_ho.obj_contact

    opt_hand_out, opt_obj_out = opt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    phy_hand_out, phy_obj_out = phy_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    opt_hand_out.translate((0.0, 0.4, 0.0))
    opt_obj_out.translate((0.0, 0.4, 0.0))
    phy_hand_out.translate((0.0, 0.6, 0.0))
    phy_obj_out.translate((0.0, 0.6, 0.0))

    geom_list = [hand_gt, obj_gt, opt_hand_out, opt_obj_out, phy_hand_out, phy_obj_out, hand_in, obj_in]
    geom_list.append(util.text_3d('In', pos=[-0.4, 0.2, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('ContactOpt', pos=[-0.4, 0.4, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('PhyOpt', pos=[-0.4, 0.6, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('GT', pos=[-0.4, 0.0, 0], font_size=40, density=2))

    if mje_in is not None:
        geom_list.append(util.text_3d('MJE in {:.2f}cm opt {:.2f}cm phy {:.2f}cm'.format(mje_in * 100, mje_opt * 100, mje_phy * 100),
                                      pos=[-0.4, -0.2, 0], font_size=40, density=2))

    o3dv.draw_geometries(geom_list)


def vis_sample(gt_ho, in_ho, out_ho, mje_in=None, mje_out=None):
    hand_gt, obj_gt = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in, obj_in = in_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_in.translate((0.0, 0.2, 0.0))
    obj_in.translate((0.0, 0.2, 0.0))

    if not args.split == 'honn':
        out_ho.hand_contact = in_ho.hand_contact
        out_ho.obj_contact = in_ho.obj_contact

    hand_out, obj_out = out_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)
    hand_out.translate((0.0, 0.4, 0.0))
    obj_out.translate((0.0, 0.4, 0.0))

    geom_list = [hand_gt, obj_gt, hand_out, obj_out, hand_in, obj_in]
    geom_list.append(util.text_3d('In', pos=[-0.4, 0.2, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('Refined', pos=[-0.4, 0.4, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('GT', pos=[-0.4, 0.0, 0], font_size=40, density=2))

    if mje_in is not None:
        geom_list.append(util.text_3d('MJE in {:.2f}cm out {:.2f}cm'.format(mje_in * 100, mje_out * 100), pos=[-0.4, -0.2, 0], font_size=40, density=2))

    o3dv.draw_geometries(geom_list)


def calc_mean_dicts(all_dicts, phase=''):
    keys = all_dicts[0].keys()
    mean_dict = dict()
    stds = ['pen_vol']

    for k in keys:
        l = list()
        for d in all_dicts:
            l.append(d[k])
        mean_dict[k] = np.array(l).mean()

        if k in stds:
            mean_dict[k + '_std'] = np.array(l).std()

    return mean_dict


def calc_sample(ho_test, ho_gt, idx, phase='nophase'):
    stats = geometric_eval.geometric_eval(ho_test, ho_gt)

    return stats


def process_sample(sample, idx):
    gt_ho, in_ho = sample['gt_ho'], sample['in_ho']
    if 'opt_ho' in sample:
        opt_ho = sample['opt_ho']
        opt_stats = calc_sample(opt_ho, gt_ho, idx, 'after ContactOpt')
    else:
        opt_stats = None
    if 'phy_ho' in sample:
        phy_ho = sample['phy_ho']
        phy_stats = calc_sample(phy_ho, gt_ho, idx, 'after PhysOpt')
    else:
        phy_stats = None
    in_stats = calc_sample(in_ho, gt_ho, idx, 'before ContactOpt')

    return in_stats, opt_stats, phy_stats


def run_eval(args):
    in_file = 'data/opt_{}.pkl'.format(args.split)
    runs = pickle.load(open(in_file, 'rb'))
    print('Loaded {} len {}'.format(in_file, len(runs)))

    # if args.vis or args.physics:
    #     print('Shuffling!!!')
    #     random.shuffle(runs)

    if args.partial > 0:
        runs = runs[:args.partial]

    do_parallel = not args.vis
    if do_parallel:
        all_data = Parallel(n_jobs=mp.cpu_count() - 2)(delayed(process_sample)(s, idx) for idx, s in enumerate(tqdm(runs)))
        in_all = [item[0] for item in all_data]
        opt_all = [item[1] for item in all_data]
        phy_all = [item[2] for item in all_data]
    else:
        all_data = []   # Do non-parallel
        for idx, s in enumerate(tqdm(runs)):
            all_data.append(process_sample(s, idx))

            if args.vis:
                print('In vs GT\n', pprint.pformat(all_data[-1][0]))
                print('Opt vs GT\n', pprint.pformat(all_data[-1][1]))
                print('Phy vs GT\n', pprint.pformat(all_data[-1][2]))
                if args.split == 'im_pred_trans':
                    vis_sample_comp(s['gt_ho'], s['in_ho'], s['opt_ho'], s['phy_ho'], mje_in=all_data[-1][0]['objalign_hand_joints'],
                                    mje_opt=all_data[-1][1]['objalign_hand_joints'], mje_phy=all_data[-1][2]['objalign_hand_joints'])
                else:
                    vis_sample_comp(s['gt_ho'], s['in_ho'], s['opt_ho'], s['phy_ho'], mje_in=all_data[-1][0]['unalign_hand_joints'],
                                    mje_opt=all_data[-1][1]['unalign_hand_joints'], mje_phy=all_data[-1][2]['unalign_hand_joints'])

        in_all = [item[0] for item in all_data]
        opt_all = [item[1] for item in all_data]
        phy_all = [item[2] for item in all_data]

    mean_in = calc_mean_dicts(in_all, 'In vs GT')
    print("In vs GT\n", pprint.pformat(mean_in))
    if opt_all[0] is not None:
        mean_opt = calc_mean_dicts(opt_all, "Opt vs GT")
        print("Opt vs GT\n", pprint.pformat(mean_opt))
    if phy_all[0] is not None:
        mean_phy = calc_mean_dicts(phy_all, "Phy vs GT")
        print("Phy vs GT\n", pprint.pformat(mean_phy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval on fitted pkl')
    parser.add_argument('--split', default='aug', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--contact_f1', action='store_true')
    parser.add_argument('--pen', action='store_true')
    parser.add_argument('--saveobj', action='store_true')
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n samples')
    args = parser.parse_args()

    start_time = time.time()
    run_eval(args)
    print('Eval time', time.time() - start_time)
