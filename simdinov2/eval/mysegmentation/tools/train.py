import argparse
import logging
import os
import os.path as osp
import sys

import torch

# Add the root directory to sys.path to allow importing simdinov2
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../../')))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../../../')))

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.utils import register_all_modules as register_all_modules_mmseg
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.runner.checkpoint import CheckpointLoader

# Override the default local file loader to use weights_only=False
# This is necessary because mmengine checkpoints contain arbitrary objects (Config, HistoryBuffer, numpy scalars)
# which are blocked by PyTorch 2.6+ default security settings.
def load_checkpoint_with_weights_only_false(filename, map_location=None, logger=None):
    return torch.load(filename, map_location=map_location, weights_only=False)

CheckpointLoader.register_scheme(prefixes=['', 'file://'], loader=load_checkpoint_with_weights_only_false, force=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--data-root', help='the root path of dataset to override config')
    parser.add_argument('--checkpoint', help='the checkpoint path to override backbone init_cfg')
    parser.add_argument('--seed', type=int, help='random seed')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    register_all_modules_mmseg(init_default_scope=False)
    register_all_modules_mmdet(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Override data_root if provided
    if args.data_root is not None:
        # Ensure data_root ends with /
        new_data_root = args.data_root if args.data_root.endswith('/') else args.data_root + '/'

        # 1. Update dataloaders
        for dataloader_name in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            if dataloader_name in cfg:
                cfg[dataloader_name].dataset.data_root = new_data_root

        # 2. Update evaluators (specifically for Panoptic tasks where paths are hardcoded)
        # We need to find the old data_root to replace it in strings
        # Typically defined in dataset config, e.g., 'data/ade/ADEChallengeData2016/'
        # We try to infer it from the train_dataloader if possible, or just replace common prefixes

        # Helper to recursively replace string prefixes in a dictionary
        def replace_prefix_recursive(d, old_prefix, new_prefix):
            for k, v in d.items():
                if isinstance(v, str):
                    if v.startswith(old_prefix):
                        d[k] = v.replace(old_prefix, new_prefix)
                elif isinstance(v, dict):
                    replace_prefix_recursive(v, old_prefix, new_prefix)
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            replace_prefix_recursive(item, old_prefix, new_prefix)

        # Common default roots in our configs
        default_roots = ['data/ade/ADEChallengeData2016/', 'data/coco/']

        for evaluator_name in ['val_evaluator', 'test_evaluator']:
            if evaluator_name in cfg:
                evaluator = cfg[evaluator_name]
                # Handle list of evaluators or single evaluator
                evaluators = evaluator if isinstance(evaluator, list) else [evaluator]

                for eval_item in evaluators:
                    # Try to replace known default roots
                    for old_root in default_roots:
                        replace_prefix_recursive(eval_item, old_root, new_data_root)
                        # Also try without trailing slash just in case
                        replace_prefix_recursive(eval_item, old_root.rstrip('/'), new_data_root.rstrip('/'))

    # Override backbone checkpoint if provided
    if args.checkpoint is not None:
        if 'model' in cfg and 'backbone' in cfg.model:
            if cfg.model.backbone.get('init_cfg') is None:
                cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=args.checkpoint)
            else:
                cfg.model.backbone.init_cfg.checkpoint = args.checkpoint

    # Override random seed if provided
    if args.seed is not None:
        if cfg.get('randomness') is None:
            cfg.randomness = dict(seed=args.seed)
        else:
            cfg.randomness.seed = args.seed

    # Automatically set save_best based on evaluator type
    if 'default_hooks' in cfg and 'checkpoint' in cfg.default_hooks:
        # Check if save_best is already set, if not, infer it
        if cfg.default_hooks.checkpoint.get('save_best') is None:
            val_evaluator = cfg.get('val_evaluator')
            if val_evaluator:
                # Handle list of evaluators
                evaluators = val_evaluator if isinstance(val_evaluator, list) else [val_evaluator]

                # Priority: Panoptic (PQ) > Semantic (mIoU)
                has_panoptic = False
                has_semantic = False

                for eval_item in evaluators:
                    eval_type = eval_item.get('type', '')
                    if 'CocoPanopticMetric' in eval_type:
                        has_panoptic = True
                    elif 'IoUMetric' in eval_type:
                        has_semantic = True

                if has_panoptic:
                    cfg.default_hooks.checkpoint.save_best = 'coco_panoptic/PQ'
                    cfg.default_hooks.checkpoint.rule = 'greater'
                    print_log('Auto-configured checkpoint hook: save_best="coco_panoptic/PQ"', logger='current')
                elif has_semantic:
                    cfg.default_hooks.checkpoint.save_best = 'mIoU'
                    cfg.default_hooks.checkpoint.rule = 'greater'
                    print_log('Auto-configured checkpoint hook: save_best="mIoU"', logger='current')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume:
        cfg.resume = True

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = Runner.from_cfg(cfg)

    # start training
    runner.train()

if __name__ == '__main__':
    main()
