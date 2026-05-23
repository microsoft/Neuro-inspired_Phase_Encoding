import argparse
import logging
import os
import os.path as osp
import sys

# Add the root directory to sys.path to allow importing simdinov2
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../../')))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../../../')))

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.utils import register_all_modules as register_all_modules_mmseg
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.runner.checkpoint import CheckpointLoader
import torch

# Override the default local file loader to use weights_only=False
# This is necessary because mmengine checkpoints contain arbitrary objects (Config, HistoryBuffer, numpy scalars)
# which are blocked by PyTorch 2.6+ default security settings.
def load_checkpoint_with_weights_only_false(filename, map_location=None, logger=None):
    return torch.load(filename, map_location=map_location, weights_only=False)

CheckpointLoader.register_scheme(prefixes=['', 'file://'], loader=load_checkpoint_with_weights_only_false, force=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    # register all modules in mmseg into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmseg(init_default_scope=False)

    # register all modules in mmdet into the registries if available
    register_all_modules_mmdet(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)

    # Set default_scope to mmseg if not present
    if cfg.get('default_scope', None) is None:
        cfg.default_scope = 'mmseg'

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

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.tta:
        if 'tta_model' not in cfg or 'tta_pipeline' not in cfg:
            print_log('Cannot find "tta_model" or "tta_pipeline" in config.'
                      ' Please check your config file.',
                      logger='current', level=logging.WARNING)
            return

        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

        # Wrap the original model with SegTTAModel
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()

if __name__ == '__main__':
    main()
