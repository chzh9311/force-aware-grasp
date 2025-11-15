import sys

for p in ['.', '..']:
    sys.path.append(p)
import os.path as osp
import numpy as np
import yaml
from easydict import EasyDict as edict
import lightning as L
from torch.utils.data import DataLoader

from common.utils.utils import update_config
from common.model.force_labelling_trainer import ForceLabellingModel
from common.dataset_utils.grab_dataset import GRABDataset
from common.dataset_utils.ho3d_dataset import HO3DDataset
from common.utils.utils import force_padding_collate_fn

def main():
    with open(osp.join('configs', 'base.yaml')) as thyml:
        base_cfg = edict(yaml.load(thyml, Loader=yaml.SafeLoader))
    with open(osp.join('configs', 'grab.yaml')) as grabyml:
        grab_cfg = yaml.load(grabyml, Loader=yaml.SafeLoader)
    # cfg = edict(update_config(base_cfg, grab_cfg))
    cfg = edict(update_config(base_cfg, grab_cfg))
    # dataset = HO3DDataset(cfg.data.dataset_path, cfg.data.obj_model_path, 'train', False)
    for split in ['train']:
        dataset = GRABDataset(cfg.data, split, load_force=False)
        dataloader = DataLoader(dataset, cfg.train.batch_size, shuffle=False, num_workers=8, collate_fn=force_padding_collate_fn)
        model = ForceLabellingModel(cfg, dataloader, split, True)
        model.run_labelling()
    # model.run_pybullet_labelling()


if __name__ == '__main__':
    main()