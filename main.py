import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["WANDB_API_KEY"] = ""
import warnings

warnings.filterwarnings("ignore")

import argparse
import glob

# pytorch_lightning
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from model.model_interface import MInterface
from data.data_interface import DInterface
from Pretrain_lightning.utils.utils import load_model_path_by_args
from utils.logger import SetupCallback
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from tools.logger import BackupCodeCallback

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CDConv')
    parser.add_argument('--data-dir', default='/data_new', type=str, metavar='N', help='data root directory')
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=21, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', nargs='+', default=[32], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=16, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int, metavar='N', help='feature channels')
    parser.add_argument('--epoch', default=500, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--lr_scheduler', default="onecycle")
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[300, 400], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt-path', default='', type=str, help='path where to save checkpoint')
    parser.add_argument('--pretrain_model_type', default='esm', type=str) 
    parser.add_argument('--esm_version', default='ESM650M', type=str)
    
    
    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--res_dir', default='/DiffSDS/Property_lightning/results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--dataset', default='ec', choices=['ec', 'fold', 'func', 'go'])
    parser.add_argument('--level', default='bp', choices=['bp', 'mf', 'cc'])
    args = parser.parse_args()
    
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    return args

def load_callbacks(args):
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='valid_fmax',
    #     mode='max',
    #     patience=20,
    #     min_delta=0.001
    # ))
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    callbacks.append(BackupCodeCallback('/xmyu/DiffSDS/Property_lightning',logdir))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    if args.dataset in ['ec','go']:
        monitor_metric = 'valid_fmax'
        filename = 'best-{epoch:02d}-{valid_fmax:.3f}'
        
    
    if args.dataset in ['fold', 'func']:
        monitor_metric = 'valid_acc'
        filename = 'best-{epoch:02d}-{valid_acc:.3f}'
    
    args.monitor_metric = monitor_metric
        
    callbacks.append(plc.ModelCheckpoint(
        monitor= monitor_metric,
        filename=filename,
        save_top_k=10,
        mode='max',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
    )
    
    # early_stop_callback = plc.EarlyStopping(monitor=monitor_metric, min_delta=0.001, patience=20, verbose=False, mode="max")
    # callbacks.append(early_stop_callback)
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks


def get_best_ckpt(checkpoint_dir, pattern=r"best-epoch=(\d+)-valid_fmax=([\d.]+).ckpt"):
    import re

    checkpoint_files = os.listdir(checkpoint_dir)

   
    best_checkpoint = None
    best_valid_fmax = float("-inf")


    for filename in checkpoint_files:
        match = re.match(pattern, filename)
        if match:
            epoch = int(match.group(1))
            valid_fmax = float(match.group(2))
            if valid_fmax > best_valid_fmax:
                best_valid_fmax = valid_fmax
                best_checkpoint = os.path.join(checkpoint_dir, filename)
    return best_checkpoint

# CUDA_VISIBLE_DEVICES=1 python Property_lightning/main.py --ex_name ec_baseline_epoch50 --epoch 50 --use_pretrain 0 --offline 0
# CUDA_VISIBLE_DEVICES=2 python Property_lightning/main.py --ex_name ec_vqproteinformer_epoch50 --epoch 50 --use_pretrain 1 --offline 0 --pretrain_model_type vqproteinformer
# CUDA_VISIBLE_DEVICES=3 python Property_lightning/main.py --ex_name ec_esm_epoch50 --epoch 50 --use_pretrain 1 --offline 0 --pretrain_model_type esm

if __name__ == "__main__":
    # now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    args = parse_args()
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    args.geometric_radii = [2*args.geometric_radius, 3*args.geometric_radius, 4*args.geometric_radius, 5*args.geometric_radius]
    
    data_module = DInterface(**vars(args))

    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    train_dataset = data_module.trainset
    steps_per_epoch = len(train_loader)
    callbacks = load_callbacks(args)
    
    if load_path is None:
        model = MInterface(loss_weights = train_dataset.weights, 
                           num_classes = train_dataset.num_classes,
                           steps_per_epoch = steps_per_epoch,
                           **vars(args))
    else:
        model = MInterface(loss_weights = train_dataset.weights, 
                           num_classes = train_dataset.num_classes,
                           steps_per_epoch = steps_per_epoch,
                           **vars(args))
        args.ckpt_path = load_path
    

    trainer_config = {
        'gpus': -1,  # Use all available GPUs
        'precision': 32,  # Use 32-bit floating point precision
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        # "distributed_backend":None,
        "strategy": 'ddp',
        # 'auto_scale_batch_size': 'binsearch',
        # "strategy": None,
        'accelerator': 'gpu',  
        'callbacks': load_callbacks(args),
        'logger': [
                    plog.WandbLogger(
                    project = 'Property',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = args.ex_name.replace('/', '-',5),
                    entity = "ggggg"),
                   plog.CSVLogger(args.res_dir, name=args.ex_name)],
        'gradient_clip_val':1.0
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_opt)
    

    trainer.fit(model, data_module)
    
    checkpoint_callback = callbacks[1]
    data_module.setup('test')
    
    if args.dataset == 'ec':
        best_checkpoint = get_best_ckpt(checkpoint_callback.dirpath, pattern=r"best-epoch=(\d+)-valid_fmax=([\d.]+).ckpt")
        model = model.load_from_checkpoint(best_checkpoint)
        test_dataloaders = data_module.test_dataloader()
        test_results = trainer.test(model, dataloaders=test_dataloaders)
        model.textlogger.info("best checkpoint: {}".format(best_checkpoint))
        model.textlogger.info(f"test_fmax: {test_results[0]['test_fmax']:.4f}\n")
        
    if args.dataset == 'fold':
        best_checkpoint = get_best_ckpt(checkpoint_callback.dirpath, pattern=r"best-epoch=(\d+)-valid_acc=([\d.]+).ckpt")
        model = model.load_from_checkpoint(best_checkpoint)
        test_dataloaders = data_module.test_dataloader()
        test_fold_results = trainer.test(model, dataloaders=test_dataloaders[0])
        test_family_results = trainer.test(model, dataloaders=test_dataloaders[1])
        test_superfamily_results = trainer.test(model, dataloaders=test_dataloaders[2])
        
        model.textlogger.info("best checkpoint: {}".format(best_checkpoint))
        model.textlogger.info(f"test_fold: {test_fold_results[0]['test_acc']:.4f}\t test_family: {test_family_results[0]['test_acc']:.4f}\t test_super: {test_superfamily_results[0]['test_acc']:.4f}\n")
    
    if args.dataset == 'func':
        best_checkpoint = get_best_ckpt(checkpoint_callback.dirpath, pattern=r"best-epoch=(\d+)-valid_acc=([\d.]+).ckpt")
        model = model.load_from_checkpoint(best_checkpoint)
        test_dataloaders = data_module.test_dataloader()
        test_results = trainer.test(model, dataloaders=test_dataloaders)
        model.textlogger.info("best checkpoint: {}".format(best_checkpoint))
        model.textlogger.info(f"test_acc: {test_results[0]['test_acc']:.4f}\n")
    
    if args.dataset == 'go':
        best_checkpoint = get_best_ckpt(checkpoint_callback.dirpath, pattern=r"best-epoch=(\d+)-valid_fmax=([\d.]+).ckpt")
        model = model.load_from_checkpoint(best_checkpoint)
        test_dataloaders = data_module.test_dataloader()
        test_results = trainer.test(model, dataloaders=test_dataloaders)
        model.textlogger.info("best checkpoint: {}".format(best_checkpoint))
        model.textlogger.info(f"test_fmax_{args.level}: {test_results[0]['test_fmax']:.4f}\n")
    