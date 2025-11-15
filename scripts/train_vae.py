import os.path as osp
import yaml
from easydict import EasyDict as edict
import torch
import torch.profiler
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import datetime
import sys
import argparse
sys.path.append('..')
sys.path.append('.')

from common.utils.utils import update_config
from common.dataset_utils.grab_dataset import GRABDatasetModule
from common.dataset_utils.ho3d_dataset import HO3DDatasetModule
from common.model.PyramidCVAE import PyramidCVAE
from common.baselines.contactgen import TriDecoder, TriEncoder, base_obj_net
from common.model.PhysAwareEncDec import PhysAwareEncoder, PhysAwareDecoder, ObjPhysFeatureNet

def main(debug=False, run_phase='train', dataset_name='grab', wandb_key='', ckpt=None):
    with open(osp.join('configs', 'base.yaml')) as thyml:
        base_cfg = edict(yaml.load(thyml, Loader=yaml.SafeLoader))
    with open(osp.join('configs', f'{dataset_name}.yaml')) as datayml:
        data_cfg = yaml.load(datayml, Loader=yaml.SafeLoader)
    cfg = update_config(base_cfg, data_cfg)
    if dataset_name == 'grab':
        dm = GRABDatasetModule(cfg)
    elif dataset_name == 'ho3d':
        dm = HO3DDatasetModule(cfg)

    encoder = TriEncoder(cfg)
    decoder = TriDecoder(cfg)
    obj_net = ObjPhysFeatureNet(in_dim=cfg.model.obj_feature, hidden_dim=cfg.model.pointnet_hc,
                                out_dim=cfg.model.pointnet_hc)
    if run_phase == 'train':
        model = PyramidCVAE(cfg, encoder, decoder, obj_net, debug=debug)
    else:
        model = PyramidCVAE.load_from_checkpoint(ckpt, cfg=cfg, encoder=encoder, decoder=decoder,
                                                 obj_feat_net=obj_net, debug=debug)
    torch.manual_seed(0)
    if not debug:
        wandb.login(key=wandb_key)
        t = datetime.datetime.now()
        wandb_logger = WandbLogger(name=cfg.model.name+'-'+t.strftime('%Y%m%d-%H%M%S'),
                                   project='force_aware_grasp',
                                   log_model=True, save_dir='logs/wandb_logs')
    else:
        wandb_logger = None
    # lit_model = LitGeometryCondDiff(th_model, scheduler, th_cfg, val_dataset)
    ## Use only 1 GPU in testing to avoid unexpected bugs.
    if run_phase == 'train':
        trainer = L.Trainer(max_epochs=cfg.train.max_epochs, accelerator="gpu", devices=2, log_every_n_steps=50, inference_mode=False,
                            check_val_every_n_epoch=cfg.train.val_every_n_epochs, logger=wandb_logger,
                            strategy='ddp_find_unused_parameters_true')
        trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt)
    else:
        # Do predictions
        # profiler = PyTorchProfiler(output_filename='profile.txt', profile_memory=True, activities=[torch.profiler.ProfilerActivity.CUDA])
        test_trainer = L.Trainer(max_epochs=cfg.train.max_epochs, accelerator="gpu", devices=1, log_every_n_steps=50, inference_mode=False,
                            logger=wandb_logger, strategy='ddp_find_unused_parameters_true')
        test_trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', type=str, default='grab')
    parser.add_argument('--run_phase', type=str, default='test')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--key', type=str)
    args = parser.parse_args()
    main(debug=args.debug, run_phase=args.run_phase, dataset_name=args.dataset, ckpt=args.ckpt, wandb_key=args.key)