# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import datetime
import json
import logging
import math
import os
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
from numpy import inf
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma, accuracy
from torch import optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))
from simdinov2.data import make_dataset
from simdinov2.data.transforms import make_finetuning_transform
from simdinov2.eval.metrics import MetricType
from simdinov2.eval.setup import get_args_parser as get_setup_args_parser
from simdinov2.eval.setup import setup_and_build_model
from simdinov2.eval.rand_aug import rand_augment_transform
from simdinov2.utils.checkpoint import Checkpointer, PeriodicCheckpointer
logger = logging.getLogger("dino")


class CheckpointableModelEma:

    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        ema_helper = ModelEma(model, *args, **kwargs)
        object.__setattr__(self, "_ema_helper", ema_helper)
        native_state = getattr(ema_helper, "state_dict", None)
        native_load = getattr(ema_helper, "load_state_dict", None)
        object.__setattr__(self, "_native_state_fn", native_state if callable(native_state) else None)
        object.__setattr__(self, "_native_load_fn", native_load if callable(native_load) else None)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_ema_helper"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_ema_helper"), name, value)

    def state_dict(self) -> Dict[str, Any]:
        native_state_fn = object.__getattribute__(self, "_native_state_fn")
        if native_state_fn is not None:
            return native_state_fn()

        ema_helper = object.__getattribute__(self, "_ema_helper")
        state: Dict[str, Any] = {}
        ema_model = getattr(ema_helper, "ema", None)
        if ema_model is not None and hasattr(ema_model, "state_dict"):
            state["ema_state"] = ema_model.state_dict()

        torch_ema = getattr(ema_helper, "_ema", None)
        if torch_ema is not None and hasattr(torch_ema, "get_state"):
            state["torch_ema"] = torch_ema.get_state()

        if hasattr(ema_helper, "num_updates"):
            state["num_updates"] = ema_helper.num_updates

        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        native_load_fn = object.__getattribute__(self, "_native_load_fn")
        if native_load_fn is not None:
            native_load_fn(state)
            return

        ema_helper = object.__getattribute__(self, "_ema_helper")

        if not isinstance(state, dict):
            raise TypeError("EMA state must be a dict when native load_state_dict is unavailable.")

        torch_ema_state = state.get("torch_ema")
        torch_ema = getattr(ema_helper, "_ema", None)
        if torch_ema_state is not None and torch_ema is not None and hasattr(torch_ema, "set_state"):
            torch_ema.set_state(torch_ema_state)

        ema_state = state.get("ema_state")
        ema_model = getattr(ema_helper, "ema", None)
        if ema_state is not None and ema_model is not None and hasattr(ema_model, "load_state_dict"):
            ema_model.load_state_dict(ema_state)
        elif ema_model is not None and hasattr(ema_model, "load_state_dict") and not state:
            ema_model.load_state_dict(state)

        if "num_updates" in state and hasattr(ema_helper, "num_updates"):
            ema_helper.num_updates = state["num_updates"]

        copy_to = getattr(ema_helper, "copy_to", None)
        if callable(copy_to):
            try:
                copy_to()
            except TypeError:
                target_model = getattr(ema_helper, "ema", None)
                if target_model is not None:
                    copy_to(target_model.parameters())


class ModelWithClassifier(nn.Module):

    def __init__(self, feature_model, embed_dim, num_classes=1000, use_multi_stage_feat=False, use_cls=False):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.use_mean_pooling = not use_cls
        self.head = nn.Linear(embed_dim, num_classes)
        self.use_multi_stage_feat = use_multi_stage_feat
        self.use_cls = use_cls
        if self.use_cls:
            logger.info("Using features: x_norm_clstoken")
        else:
            logger.info("Using features: x_mean_pooling")

    def forward(self, images, return_phase: bool = False):
        if self.use_cls:
            outputs = self.feature_model.forward_features(images)
            features = outputs["x_norm_clstoken"]
            logit = self.head(features)
            if return_phase:
                # B, S, H, D//2, 2
                phase = torch.stack(
                    (outputs["kope_patch_phase_cos"], outputs["kope_patch_phase_sin"]), dim=-1
                )
                return logit, phase
            return logit
        if self.use_multi_stage_feat:
            features = self.feature_model.forward_multistage_features(images)["x_mean_pooling"]
        else:
            features = self.feature_model.forward_features(images)["x_mean_pooling"]
        logit = self.head(features)
        return logit

    def forward_with_phase(self, images):
        return self.forward(images, return_phase=True)

    def get_depths(self):
        return self.feature_model.get_depths()

    def no_weight_decay(self):
        no_wd = getattr(self.feature_model, "no_weight_decay", None)
        if callable(no_wd):
            names = list(no_wd())
        else:
            skip_keywords = ("norm", "bn", "gamma")
            token_keywords = ("pos_embed", "cls_token", "mask_token", "register_tokens")
            names = []
            for name, param in self.feature_model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith(".bias") or param.ndim <= 1 or any(k in name for k in skip_keywords):
                    names.append(name)
                elif any(k in name for k in token_keywords):
                    names.append(name)
        names = ['feature_model.' + name for name in sorted(set(names))]
        return names


def get_parameter_groups(
        model,
        weight_decay=1e-5,
        skip_list=(),
        get_num_layer=None,
        get_layer_scale=None,
        phase_coupling_lr_mult: float = 1.0,
    ):
    parameter_group_names = {}
    parameter_group_vars = {}

    def _is_phase_coupling_param(param_name: str) -> bool:
        # Heuristic match for KoPE coupling weights.
        # This keeps the logic robust across KoPE variants without importing model code.
        name_l = param_name.lower()
        patterns = (
            "phase_coupling",
            "kope_coupling",
            "coupling",
        )
        return any(p in name_l for p in patterns)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        name_lr_mult = (
            phase_coupling_lr_mult
            if phase_coupling_lr_mult != 1.0 and _is_phase_coupling_param(name)
            else 1.0
        )
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        # Force separate group for coupling params to avoid polluting other params in the same layer
        if name_lr_mult != 1.0:
            group_name = f"{group_name}_coupling"

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            scale *= name_lr_mult

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
        args, model, get_num_layer=None, get_layer_scale=None,
        filter_bias_and_bn=True, skip_list=None
    ):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(
            model,
            weight_decay,
            skip,
            get_num_layer,
            get_layer_scale,
            phase_coupling_lr_mult=getattr(args, "phase_coupling_lr_mult", 1.0),
        )
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.learning_rate, weight_decay=weight_decay)
    opt_args['eps'] = 1e-8

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


class LayerDecayValueAssigner(object):

    def __init__(self, values, prefix, net_type, actived_block_idx, depths=None):
        assert net_type in ['swin', 'vit', 'resnet', 'hybridk']
        self.values = values
        self.depths = depths
        self.prefix = prefix
        self.net_type = net_type
        if net_type == 'resnet':
            assert isinstance(actived_block_idx, list)
            assert len(actived_block_idx) == 2
            self.block_id_map = []
            for actived_block_idx_i in actived_block_idx:
                self.block_id_map.append({str(x): i for i, x in enumerate(actived_block_idx_i)})
        else:
            self.block_id_map = {str(x): i for i, x in enumerate(actived_block_idx)}

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_num_layer_for_resnet(self, var_name, num_max_layer, depths):
        if var_name == f"{self.prefix}.mask_token":
            return 0
        elif var_name.startswith(f"{self.prefix}.conv1"):
            return 0
        elif var_name.startswith(f"{self.prefix}.norm1"):
            return 0
        elif var_name.startswith(f"{self.prefix}.layer"):
            stage_id = int(var_name.split('.')[1].replace('layer', '')) - 1
            if stage_id == 1:
                block_id = self.block_id_map[0].get(var_name.split('.')[3], -1)
            elif stage_id == 2:
                block_id = self.block_id_map[1].get(var_name.split('.')[3], -1)
            else:
                block_id = int(var_name.split('.')[3])
            layer_id = sum(depths[:stage_id]) + block_id
            if block_id != -1:
                print(f'resnet-{stage_id}-{layer_id}', var_name)
            else:
                return 0 # not activated parameters
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_num_layer_for_swin(self, var_name, num_max_layer, depths):
        if var_name in (
            f"{self.prefix}.mask_token", f"{self.prefix}.pos_embed"
        ):
            return 0
        elif var_name.startswith(f"{self.prefix}.patch_embed"):
            return 0
        elif var_name.startswith(f"{self.prefix}.stages"):
            stage_id = int(var_name.split('.')[2])
            if stage_id == 2:
                if 'blocks' in var_name:
                    block_id = self.block_id_map.get(var_name.split('.')[4], -1)
                    if block_id != -1:
                        self.cur_block_id = block_id
                else:
                    block_id = self.cur_block_id
            else:
                if 'blocks' in var_name:
                    block_id = int(var_name.split('.')[4])
                    self.cur_block_id = block_id
                else:
                    block_id = self.cur_block_id
            layer_id = sum(depths[:stage_id]) + block_id
            if block_id != -1:
                print(f'swin-{layer_id}', var_name)
            else:
                return 0 # not activated parameters
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_num_layer_for_vit(self, var_name, num_max_layer):
        if var_name in (
            f"{self.prefix}.cls_token", f"{self.prefix}.mask_token", f"{self.prefix}.pos_embed"
        ):
            return 0
        elif var_name.startswith(f"{self.prefix}.patch_embed"):
            return 0
        elif var_name.startswith(f"{self.prefix}.blocks"):
            parts = var_name.split('.')
            # Check for nested structure: prefix.blocks.chunk_id.block_id...
            # If parts[3] is a digit, it's likely the block_id in a nested structure
            if len(parts) > 3 and parts[3].isdigit():
                block_id = parts[3]
            else:
                block_id = parts[2]

            layer_id = self.block_id_map.get(block_id, -1)
            return layer_id + 1
        else:
            return num_max_layer - 1

    def get_layer_id(self, var_name):
        if self.net_type == 'swin':
            return self.get_num_layer_for_swin(var_name, len(self.values), self.depths)
        if self.net_type == 'resnet':
            return self.get_num_layer_for_resnet(var_name, len(self.values), self.depths)
        if self.net_type == 'vit':
            return self.get_num_layer_for_vit(var_name, len(self.values))
        if self.net_type == 'hybridk':
            return self.get_num_layer_for_vit(var_name, len(self.values))


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = self.get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def get_grad_norm_(self, parameters, norm_type: float = 2.0) -> torch.Tensor:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm


class MetricLogger2(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_logger(file_path_name):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    DATE_FORMAT = ''
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(file_path_name)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def strong_transforms(
    img_size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="random",
    use_prefetcher=True,
    mean=IMAGENET_DEFAULT_MEAN,  # (0.485, 0.456, 0.406)
    std=IMAGENET_DEFAULT_STD,  # (0.229, 0.224, 0.225)
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
    re_num_splits=0,
    color_aug=False,
    strong_ratio=0.45,
):
    """
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range

    primary_tfl = []
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * strong_ratio),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
    if color_jitter is not None and color_aug:
        # color jitter is enabled when not using AA
        flip_and_color_jitter = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        secondary_tfl += flip_and_color_jitter

    if interpolation == "random":
        interpolation = (Image.BILINEAR, Image.BICUBIC)
    else:
        interpolation = _pil_interp(interpolation)
    final_tfl = [
        transforms.RandomResizedCrop(
            size=img_size, scale=scale, ratio=ratio, interpolation=Image.BICUBIC
        )
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [transforms.ToTensor()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
            )
        )
    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--arch-name",
        type=str,
        default="vit",
        help="Architecture name: swin, vit, or resnet",
    )
    parser.add_argument(
        "--net-type",
        type=str,
        default="base",
        help="Network type for vit or swin, for example: samll, base, large",
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        '--input-size',
        type=int,
        help='images input size')
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for finetuning.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        '--color-jitter',
        type=float,
        help='Color jitter factor (default: 0.4)'
    )
    parser.add_argument(
        '--aa', type=str, default='rand-m9-mstd0.5-inc1',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
    )
    parser.add_argument(
        '--train-interpolation',
        type=str,
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")'
    )
    parser.add_argument(
        '--reprob',
        type=float,
        help='Random erase prob (default: 0.25)'
    )
    parser.add_argument(
        '--opt',
        type=str,
        help='Optimizer (default: "adamw")'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        help='Weight decay (default: 0.05)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--min-lr',
        type=float,
        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        help='epochs to warmup LR, if scheduler supports'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int, default=-1,
        help='num of steps to warmup LR, will overload warmup_epochs if set > 0'
    )
    parser.add_argument(
        '--layer-decay',
        type=float,
        help='layer lr decay rate (default: 0.9)'
   )
    parser.add_argument(
        "--phase-coupling-lr-mult",
        dest="phase_coupling_lr_mult",
        type=float,
        default=1.0,
        help="LR multiplier applied to KoPE coupling parameters during finetuning (implemented via lr_scale).",
    )
    # * Mixup params
    parser.add_argument(
        '--mixup',
        type=float,
        help='mixup alpha, mixup enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix',
        type=float,
        help='cutmix alpha, cutmix enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix-minmax',
        type=float, nargs='+', default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup-prob',
        type=float, default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup-switch-prob',
        type=float, default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup-mode',
        type=str, default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )
    parser.add_argument(
        '--model-ema', action='store_true',
        help='Using model EMA in training',
    )
    parser.add_argument(
        '--model-ema-decay', type=float,
    )
    parser.add_argument(
        '--smoothing',
        type=float,
        help='Label smoothing (default: 0.1)'
    )
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--clip_grad',
        type=float, default=None,
         help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--drop_path',
        type=float, default=0.1,
        help='Drop path rate (default: 0.1)'
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        input_size=224,
        epochs=200,
        warmup_epochs=20,
        batch_size=128,
        num_workers=32,
        pin_mem=True,
        epoch_length=1250,
        color_jitter=0.4,
        reprob=0.25,
        train_interpolation="bicubic",
        save_checkpoint_frequency=20,
        eval_period_iterations=1250,
        opt='adamw',
        weight_decay=0.05,
        momentum=0.9,
        learning_rate=0.0012,
        min_lr=1e-6,
        layer_decay=0.9,
        mixup=0.8,
        cutmix=1.0,
        smoothing=0.1,
        model_ema=True,
        model_ema_decay=0.7,
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
    )
    return parser

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger2(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
        model_ema: Optional[ModelEma] = None,
    ):
    model.train(True)
    metric_logger = MetricLogger2(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
            metric_logger.update(class_acc=class_acc)
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def run_finetnuing(args):

    if not Path(args.pretrained_weights).exists():
        raise Exception(f'Pretrained model not found: {args.pretrained_weights}')

    # training setting
    assert args.arch_name in ['vit', 'swin', 'resnet', 'hybridk']
    '''
    if args.arch_name in ['vit', 'hybridk']:
        batch_size_dict = {'small': 256, 'base': 256, 'large': 256}
        #batch_size_dict = {'small': 128, 'base': 128, 'large': 128}
    elif args.arch_name == 'swin':
        batch_size_dict = {'tiny': 128, 'small': 128, 'base': 64}
    elif args.arch_name == 'resnet':
        batch_size_dict = {'R50': 128, 'R101': 128, 'R152': 64}
    args.batch_size = batch_size_dict[args.net_type]
    '''

    if args.arch_name in ['vit', 'hybridk']:
        training_epochs_dict = {'small': 100, 'base': 100, 'large': 50}
    elif args.arch_name == 'swin':
        training_epochs_dict = {'tiny': 200, 'small': 100, 'base': 50}
    elif args.arch_name == 'resnet':
        training_epochs_dict = {'R50': 100, 'R101': 100, 'R152': 100}
    training_epochs = training_epochs_dict[args.net_type]
    args.epochs = training_epochs

    if args.arch_name in ['vit', 'hybridk']:
        #learning_rate_dict = {'small': 0.002, 'base': 0.0007, 'large': 0.0018}
        learning_rate_dict = {'small': 0.002, 'base': 0.002, 'large': 0.004}
    elif args.arch_name == 'swin':
        learning_rate_dict = {'tiny': 0.0014, 'small': 0.001, 'base': 0.0007}
    elif args.arch_name == 'resnet':
        learning_rate_dict = {'R50': 0.0014, 'R101': 0.001, 'R152': 0.0007}
    learning_rate = learning_rate_dict[args.net_type]
    args.learning_rate = learning_rate

    if args.arch_name in ['vit', 'hybridk']:
        #layer_decay_dict = {'small': 0.55, 'base': 0.4, 'large': 0.6}
        layer_decay_dict = {'small': 0.65, 'base': 0.65, 'large': 0.75}
    elif args.arch_name == 'swin':
        layer_decay_dict = {'tiny': 0.6, 'small': 0.4, 'base': 0.55}
    elif args.arch_name == 'resnet':
        layer_decay_dict = {'R50': 0.6, 'R101': 0.4, 'R152': 0.45}
    layer_decay = layer_decay_dict[args.net_type]
    warmup_epochs = int(training_epochs * 0.1)

    # network setting
    if args.arch_name in ['vit', 'hybridk']:
        drop_path_dict = {'small': 0.1, 'base': 0.1, 'large': 0.2}
    elif args.arch_name == 'swin':
        drop_path_dict = {'tiny': 0.1, 'small': 0.2, 'base': 0.2}
    elif args.arch_name == 'resnet':
        drop_path_dict = {'R50': None, 'R101': None, 'R152': None}
    args.drop_path = drop_path_dict[args.net_type]
    feature_model, _, cfg = setup_and_build_model(args)

    # Sync args with cfg to ensure checkpoint contains correct model config for eval
    # We will prepare a separate args object for saving, which contains the correct arch_name for eval.
    # The runtime args.arch_name will be kept as 'vit' (or whatever the user provided) to ensure compatibility with finetuning logic.
    args_to_save = copy.copy(args)

    if hasattr(cfg, "student"):
        student_cfg = cfg.student

        def _copy_if_present(cfg_obj, key, target_attr):
            if key in cfg_obj:
                val = cfg_obj[key]
                if val is not None:
                    setattr(args, target_attr, val)
                    setattr(args_to_save, target_attr, val)

        # 0. Arch / variant sync: make sure checkpoint args reflect the KoPE arch so eval rebuilds the same model.
        arch_from_cfg = getattr(student_cfg, "arch", None)
        if arch_from_cfg:
            arch_name = str(arch_from_cfg).lower()
            # Examples: vit_kope_small -> arch_name: vit_kope, variant: small
            parts = arch_name.split("_", 2)
            variant = None
            if len(parts) >= 3:
                variant = parts[-1]
                arch_family = "_".join(parts[:2])
            else:
                arch_family = parts[0]

            # Determine the arch name for saving (eval compatibility)
            save_arch_name = args.arch_name
            if arch_family in ("vit_kope", "vit-rope", "vit_rope"):
                save_arch_name = "vit_kope" if "kope" in arch_family else "vit_rope"
            elif arch_family == "vit":
                save_arch_name = "vit"
            elif arch_family == "swin":
                save_arch_name = "swin"
            else:
                save_arch_name = arch_family

            args_to_save.arch_name = save_arch_name

            # Force runtime arch_name to 'vit' if it's a kope/rope variant, to reuse vit logic
            if save_arch_name in ("vit_kope", "vit_rope"):
                args.arch_name = "vit"

            if variant:
                args.model_variant = variant
                args.net_type = variant
                args_to_save.model_variant = variant
                args_to_save.net_type = variant
            elif not hasattr(args, "model_variant") and hasattr(args, "net_type"):
                args.model_variant = args.net_type
                args_to_save.model_variant = args.net_type

        # 1. Basic Model Config (skip None to avoid clobbering defaults)
        basic_keys = [
            "block_chunks", "num_register_tokens", "qkv_bias", "proj_bias", "ffn_bias", "layerscale",
            "patch_size", "qknorm", "max_resolution", "base", "drop_path_uniform", "block_type", "ffn_layer",
            "drop_masks", "gradient_checkpointing",
        ]
        for key in basic_keys:
            _copy_if_present(student_cfg, key, key)

        # 2. KoPE Parameter Mapping (cfg name -> eval.py args name)
        kope_mapping = {
            "update_ext_token_phase": "kope_update_ext_phase",
            "phase_mode": "kope_phase_mode",
            "phase_head_weight": "kope_phase_head_weight",
            "phase_token_weight": "kope_phase_token_weight",
            "coupling_use_phase_bias_rotation": "kope_coupling_use_phase_bias_rotation",
            "coupling_use_phase_rotation": "kope_coupling_use_phase_rotation",
            "coupling_use_vo_proj": "kope_coupling_use_vo_proj",
            "use_bias_for_phase_update": "kope_use_bias_for_phase_update",
            "fixed_layout": "kope_fixed_layout",
            "use_kope_x_attention": "kope_use_x_attention",
            "use_kope_x_attention_v2": "kope_use_x_attention_v2",
            "phase_n": "kope_phase_n",
            "new_phase_init": "kope_new_phase_init",
            "new_phase_init_scale": "kope_new_phase_init_scale",
            "learn_phase_gamma": "kope_learn_phase_gamma",
            "coupling_use_tanh": "kope_coupling_use_tanh",
            "checkpoint_phase_update": "kope_checkpoint_phase_update",
            "checkpoint_ffn_func": "kope_checkpoint_ffn_func",
            "checkpoint_rotation": "kope_checkpoint_rotation",
            "coupling_qknorm": "kope_coupling_qknorm",
            "coupling_qknorm_learn": "kope_coupling_qknorm_learn",
            "coupling_qk_multilayer": "kope_coupling_qk_multilayer",
            "start_kope_idx": "kope_start_idx",
            "no_phase_norm": "kope_no_phase_norm",
            "use_learnable_pos_embed": "learnable_pos_embed",
            # Direct copy (names match)
            "kope_mix": "kope_mix",
            "kope_mix_init_gain": "kope_mix_init_gain",
            "kope_mix_phase_norm": "kope_mix_phase_norm",
            "kope_gamma": "kope_gamma",
            "share_kope_coupling": "share_kope_coupling",
            "kope_vo_rotation": "kope_vo_rotation",
        }

        for cfg_key, args_key in kope_mapping.items():
            _copy_if_present(student_cfg, cfg_key, args_key)

        # 3. Ensure model_variant is set (eval.py uses model_variant)
        if not hasattr(args, 'model_variant') and hasattr(args, 'net_type'):
            args.model_variant = args.net_type
            args_to_save.model_variant = args.net_type
    else:
        # If no student cfg, just copy args
        args_to_save = copy.copy(args)

    #feature_model.get_by_type(args.net_type)
    feature_model.use_mean_pooling = True

    embed_dim = getattr(feature_model, "feat_dim", None)
    if embed_dim is None:
        embed_dim = getattr(feature_model, "embed_dim", None)
    if embed_dim is None:
        raise AttributeError("Feature model is missing embed_dim/feat_dim attribute")

    if hasattr(feature_model, "block_idx"):
        actived_block_idx = feature_model.block_idx
    elif args.arch_name in ['vit', 'hybridk']:
        num_vit_layers = getattr(feature_model, "n_blocks", None)
        if num_vit_layers is None:
            blocks_container = getattr(feature_model, "blocks", None)
            if blocks_container is None:
                raise AttributeError("Unable to infer ViT block count from feature model")
            num_vit_layers = len(blocks_container)
        actived_block_idx = [str(i) for i in range(num_vit_layers)]
    else:
        raise AttributeError("Feature model is missing block_idx attribute")

    if hasattr(feature_model, "get_num_layers"):
        num_layers = feature_model.get_num_layers()
    elif args.arch_name in ['vit', 'hybridk']:
        num_layers = len(actived_block_idx)
    else:
        raise AttributeError("Feature model is missing get_num_layers method")

    train_transform = make_finetuning_transform(is_train=True, args=args)
    train_dataset = make_dataset(dataset_str=args.train_dataset_str, transform=train_transform)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    logger.info("Sampler_train = %s" % str(sampler_train))
    val_transform = make_finetuning_transform(is_train=False, args=args)
    val_dataset = make_dataset(dataset_str=args.val_dataset_str,transform=val_transform)
    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    training_num_classes = len(torch.unique(torch.Tensor(train_dataset.get_targets().astype(int))))
    use_multi_stage_feat = args.arch_name == 'resnet'
    use_cls = args.arch_name  in ['vit', 'hybridk']
    model = ModelWithClassifier(feature_model, embed_dim, training_num_classes, use_multi_stage_feat, use_cls)
    device = torch.device('cuda')
    model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = CheckpointableModelEma(
            model, decay=args.model_ema_decay, device='', resume=''
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_batch_size = args.batch_size * dist.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model_without_ddp = model.module

    logger.info(f"Finetuning network: {args.net_type}")
    logger.info("Model = %s" % str(model_without_ddp))
    logger.info("Number of params: %d " % n_parameters)
    logger.info("LR = %.8f" % learning_rate)
    logger.info("Total batch size = %d" % total_batch_size)
    logger.info("Number of training examples = %d" % len(train_dataset))
    logger.info("Number of training step per epoch = %d" % num_training_steps_per_epoch)


    if layer_decay < 1.0 and args.arch_name == 'resnet':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='resnet', actived_block_idx=actived_block_idx,
            depths=model_without_ddp.get_depths()
        )
    elif layer_decay < 1.0 and args.arch_name == 'swin':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='swin', actived_block_idx=actived_block_idx,
            depths=model_without_ddp.get_depths()
        )
    elif layer_decay < 1.0 and args.arch_name == 'vit':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='vit', actived_block_idx=actived_block_idx
        )
    elif layer_decay < 1.0 and args.arch_name == 'hybridk':
        assigner = LayerDecayValueAssigner(
            list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            prefix = 'feature_model', net_type='hybridk', actived_block_idx=actived_block_idx
        )
    else:
        assigner = None
    skip_weight_decay_list = model_without_ddp.no_weight_decay()
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    # checkpoint setup (directory creation on main process only)
    output_dir_path = None
    if args.output_dir:
        output_dir_path = Path(args.output_dir)
        if dist.get_rank() == 0:
            output_dir_path.mkdir(parents=True, exist_ok=True)

    loss_scaler = NativeScalerWithGradNormCount()

    checkpointables = {
        "optimizer": optimizer,
        "loss_scaler": loss_scaler,
        # "args": args,
    }
    if model_ema is not None:
        checkpointables["model_ema"] = model_ema

    checkpointer = Checkpointer(
        model,
        save_dir=str(output_dir_path) if output_dir_path is not None else "",
        save_to_disk=output_dir_path is not None and dist.get_rank() == 0,
        **checkpointables,
    )
    extra_state = {}
    max_accuracy = 0.0
    start_epoch = 0
    has_prior_checkpoint = (
        not args.no_resume
        and output_dir_path is not None
        and checkpointer.has_checkpoint()
    )
    if has_prior_checkpoint:
        extra_state = checkpointer.resume_or_load(args.pretrained_weights, resume=True)
        start_epoch = extra_state.get("epoch", -1) + 1
        max_accuracy = extra_state.get("max_accuracy", 0.0)
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        logger.info("No existing finetuning checkpoints detected; relying on model builder for pretrained weights.")

    logger.info("Use step level LR scheduler!")
    lr_schedule_values = cosine_scheduler(
        learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=warmup_epochs, warmup_steps=args.warmup_steps,
    )

    periodic_checkpointer = None
    if output_dir_path is not None:
        ckpt_period = args.save_checkpoint_frequency
        if ckpt_period is None or ckpt_period <= 0:
            ckpt_period = training_epochs
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=max(1, ckpt_period),
            max_iter=training_epochs,
            file_prefix="finetune",
        )
    # mix up training
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=training_num_classes)
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    logger.info("Criterion = %s" % str(criterion))

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, training_epochs):
        train_data_loader.sampler.set_epoch(epoch)
        # Training
        train_stats = train_one_epoch(
            model, criterion, train_data_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad, mixup_fn,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=None,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=1,
            model_ema=model_ema,
        )
        # Evaluation
        eval_model = model_ema.ema if args.model_ema else model
        test_stats = evaluate(val_data_loader, eval_model, device)
        logger.info(
            f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}%"
        )
        improved = test_stats["acc1"] > max_accuracy
        if improved:
            max_accuracy = test_stats["acc1"]
            if output_dir_path is not None and dist.get_rank() == 0:
                # Save best checkpoint for downstream evaluation.
                # `supervised/eval/eval.py` will load checkpoint["model"].
                # Since finetuning validates with EMA (when enabled), store EMA weights as "model".
                original_state = None
                use_ema_for_best = args.model_ema and model_ema is not None and getattr(model_ema, "ema", None) is not None
                if use_ema_for_best:
                    original_state = model_without_ddp.state_dict()
                    model_without_ddp.load_state_dict(model_ema.ema.state_dict(), strict=False)
                try:
                    checkpointer.save(
                        "finetune_best",
                        epoch=epoch,
                        iteration=epoch,
                        max_accuracy=max_accuracy,
                        args=args_to_save,
                    )
                finally:
                    if use_ema_for_best and original_state is not None:
                        model_without_ddp.load_state_dict(original_state, strict=False)
        logger.info(f'Epoch[{epoch + 1}/{training_epochs}] Max accuracy: {max_accuracy:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'max_accuracy': max_accuracy}
        if args.output_dir and dist.get_rank() == 0:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if periodic_checkpointer is not None:
            periodic_checkpointer.step(epoch, epoch=epoch, max_accuracy=max_accuracy, args=args_to_save)
        last_epoch = epoch

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if (
        output_dir_path is not None
        and dist.get_rank() == 0
        and last_epoch >= 0
        and periodic_checkpointer is None
    ):
        checkpointer.save(
            "finetune_final",
            epoch=last_epoch,
            iteration=last_epoch,
            max_accuracy=max_accuracy,
            args=args_to_save,
        )


def main(args):
    run_finetnuing(args)
    return 0


if __name__ == "__main__":
    description = "Finetuning evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
