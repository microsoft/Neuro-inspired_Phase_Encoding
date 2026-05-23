# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

#from losses import DistillationLoss
from simdinov2.supervised.deit.losses import DistillationLoss
from simdinov2.supervised.deit.losses import PhaseLoss
#import utils
from simdinov2.supervised.deit import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    use_phase_loss: bool = False, phase_loss_weight: float = 0.1, phase_criterion = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            if use_phase_loss:
                assert phase_criterion is not None
                outputs, phases = model(samples, return_phase=True)
                loss_phase = phase_criterion(phases)
            else:
                outputs = model(samples)
                loss_phase = None
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())
            if use_phase_loss and phase_loss_weight > 0:
                loss = loss + phase_loss_weight * loss_phase

        loss_value = loss.item()
        if use_phase_loss and loss_phase is not None:
            metric_logger.update(loss_phase=loss_phase.item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _resolve_backbone(model: torch.nn.Module):
    """Find backbone for analysis: prefer .backbone, then .feature_model, else model itself."""
    for attr in ("backbone", "feature_model"):
        bb = getattr(model, attr, None)
        if bb is not None:
            return bb
    return model


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    analyze_attention = args is not None and getattr(args, 'analyze_attention', False)
    real_model = model.module if hasattr(model, 'module') else model
    backbone = _resolve_backbone(real_model)

    if analyze_attention and backbone and hasattr(backbone, 'set_analysis_mode'):
        backbone.set_analysis_mode(True)
        print("Attention analysis enabled.")

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        if analyze_attention and backbone and hasattr(backbone, 'get_analysis_results'):
            metrics = backbone.get_analysis_results()
            if metrics:
                metric_logger.update(**metrics)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if analyze_attention and backbone and hasattr(backbone, 'set_analysis_mode'):
        backbone.set_analysis_mode(False)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
