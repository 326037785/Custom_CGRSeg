#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
CGRSeg Training and Evaluation Script

A user-friendly script for single-GPU training and evaluation of CGRSeg models.
This script provides easy adjustment of hyperparameters for convenient experimentation.

Usage:
    # Training
    python run_cgrseg.py --mode train --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py

    # Training with custom hyperparameters
    python run_cgrseg.py --mode train --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \
        --lr 0.0001 --batch-size 4 --max-iters 80000

    # Evaluation
    python run_cgrseg.py --mode eval --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \
        --checkpoint ./work_dirs/cgrseg-t_ade20k_160k/latest.pth
"""

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, load_checkpoint
from mmcv.utils import Config, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_device, get_root_logger, setup_multi_processes, build_dp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CGRSeg Training and Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Training:
    python run_cgrseg.py --mode train --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py
    
  Training with custom hyperparameters:
    python run_cgrseg.py --mode train --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \\
        --lr 0.0001 --batch-size 4 --max-iters 80000
    
  Evaluation:
    python run_cgrseg.py --mode eval --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \\
        --checkpoint ./work_dirs/cgrseg-t_ade20k_160k/latest.pth
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'eval'],
        help='Mode: "train" for training, "eval" for evaluation'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., local_configs/cgrseg/cgrseg-t_ade20k_160k.py)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: uses config value, typically 0.00012)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size per GPU (default: uses config value, typically 4)'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=None,
        help='Maximum training iterations (default: uses config value, typically 160000)'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='Evaluation interval during training (default: uses config value, typically 4000)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='Directory to save logs and checkpoints'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether to set deterministic options for CUDNN backend'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to evaluate the checkpoint during training'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Checkpoint file to resume training from'
    )
    parser.add_argument(
        '--load-from',
        type=str,
        default=None,
        help='Checkpoint file to load weights from (for fine-tuning)'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for evaluation (required for eval mode)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show prediction results during evaluation'
    )
    parser.add_argument(
        '--show-dir',
        type=str,
        default=None,
        help='Directory to save visualization results'
    )
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--eval-metric',
        type=str,
        default='mIoU',
        help='Evaluation metric (default: mIoU)'
    )
    
    # GPU settings
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'eval' and args.checkpoint is None:
        parser.error('--checkpoint is required for evaluation mode')
    
    return args


def train(args):
    """
    Training function for CGRSeg.
    
    Args:
        args: Parsed command line arguments
    """
    cfg = Config.fromfile(args.config)
    
    # Override hyperparameters if specified
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
        print(f'Learning rate set to: {args.lr}')
    
    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size
        print(f'Batch size per GPU set to: {args.batch_size}')
    
    if args.max_iters is not None:
        cfg.runner.max_iters = args.max_iters
        cfg.checkpoint_config.interval = args.max_iters  # Save final checkpoint
        print(f'Maximum iterations set to: {args.max_iters}')
    
    if args.eval_interval is not None:
        cfg.evaluation.interval = args.eval_interval
        print(f'Evaluation interval set to: {args.eval_interval}')
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    # Resume or load from checkpoint
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        print(f'Resuming from: {args.resume_from}')
    
    if args.load_from is not None:
        cfg.load_from = args.load_from
        print(f'Loading weights from: {args.load_from}')
    
    # Set CUDNN benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # Set GPU
    cfg.gpu_ids = [args.gpu_id]
    
    # Non-distributed training
    distributed = False
    
    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # Initialize logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    # Set multi-process settings
    setup_multi_processes(cfg)
    
    # Initialize meta dict
    meta = dict()
    
    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    
    # Log config
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # Set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    
    # Build model
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()
    
    # Convert SyncBN to BN for non-distributed training
    warnings.warn(
        'SyncBN is only supported with DDP. Converting SyncBN to BN for single-GPU training.'
    )
    model = revert_sync_batchnorm(model)
    
    logger.info(model)
    
    # Build datasets
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE
        )
    
    model.CLASSES = datasets[0].CLASSES
    meta.update(cfg.checkpoint_config.meta)
    
    # Start training
    print('\n' + '=' * 60)
    print('Starting Training...')
    print(f'Work directory: {cfg.work_dir}')
    print(f'GPU: {args.gpu_id}')
    print('=' * 60 + '\n')
    
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta
    )
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'Checkpoints saved to: {cfg.work_dir}')
    print('=' * 60 + '\n')


def evaluate(args):
    """
    Evaluation function for CGRSeg.
    
    Args:
        args: Parsed command line arguments
    """
    cfg = Config.fromfile(args.config)
    
    # Set multi-process settings
    setup_multi_processes(cfg)
    
    # Set CUDNN benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    
    # Set GPU
    cfg.gpu_ids = [args.gpu_id]
    
    # Non-distributed testing
    distributed = False
    
    # Set work directory
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    json_file = osp.join(work_dir, f'eval_single_scale_{timestamp}.json')
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    
    # Build dataloader
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False
    )
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    # Build model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    
    # Clean GPU memory
    torch.cuda.empty_cache()
    
    # Convert SyncBN to BN for non-distributed testing
    cfg.device = get_device()
    if not torch.cuda.is_available():
        from mmseg import digit_version
        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
            'Please use MMCV >= 1.4.4 for CPU training!'
    
    warnings.warn(
        'SyncBN is only supported with DDP. Converting SyncBN to BN for single-GPU testing.'
    )
    model = revert_sync_batchnorm(model)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    
    print('\n' + '=' * 60)
    print('Starting Evaluation...')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'GPU: {args.gpu_id}')
    print('=' * 60 + '\n')
    
    # Run evaluation
    results = single_gpu_test(
        model,
        data_loader,
        args.show,
        args.show_dir,
        False,
        args.opacity,
        pre_eval=True,
        format_only=False,
        format_args={}
    )
    
    # Compute metrics
    eval_kwargs = dict(metric=args.eval_metric)
    metric = dataset.evaluate(results, **eval_kwargs)
    
    # Save results
    metric_dict = dict(config=args.config, metric=metric)
    mmcv.dump(metric_dict, json_file, indent=4)
    
    print('\n' + '=' * 60)
    print('Evaluation Results:')
    for key, value in metric.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.4f}')
        else:
            print(f'  {key}: {value}')
    print(f'\nResults saved to: {json_file}')
    if args.show_dir:
        print(f'Visualizations saved to: {args.show_dir}')
    print('=' * 60 + '\n')
    
    return metric


def main():
    """Main entry point."""
    args = parse_args()
    
    print('\n' + '=' * 60)
    print('CGRSeg - Context-Guided Spatial Feature Reconstruction')
    print('=' * 60)
    print(f'Mode: {args.mode}')
    print(f'Config: {args.config}')
    print('=' * 60 + '\n')
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)


if __name__ == '__main__':
    main()
