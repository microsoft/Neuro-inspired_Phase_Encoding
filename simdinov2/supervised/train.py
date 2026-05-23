# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Optional, Sequence


def str2bool(value: str) -> bool:
	if isinstance(value, bool):
		return value
	value_lower = value.lower()
	if value_lower in {"true", "1", "yes", "y", "t", "on"}:
		return True
	if value_lower in {"false", "0", "no", "n", "f", "off"}:
		return False
	raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict

from simdinov2.eval.finetuning import ModelWithClassifier
from simdinov2.models import vision_transformer as vit_models
from simdinov2.models import vision_transformer_kope as kope_models
from simdinov2.models import vision_transformer_rope as rope_models

from simdinov2.supervised.deit import augment as deit_augment
from simdinov2.supervised.deit import datasets as deit_datasets
from simdinov2.supervised.deit import engine as deit_engine
from simdinov2.supervised.deit import losses as deit_losses
from simdinov2.supervised.deit import samplers as deit_samplers
from simdinov2.supervised.deit import utils as deit_utils


def get_args_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		"SimDINOv2 supervised ImageNet training (DeiT III recipe)",
		add_help=False,
	)

	# core training hyper-parameters (mirroring DeiT III defaults)
	parser.add_argument("--batch-size", default=64, type=int)
	parser.add_argument("--epochs", default=300, type=int)
	parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
	parser.add_argument("--unscale-lr", action="store_true")

	# model / backbone selection
	parser.add_argument(
		"--arch-name",
		default="vit",
		choices=["vit", "vit_rope", "vit_kope"],
		help="SimDINOv2 backbone family to train",
	)
	parser.add_argument(
		"--model-variant",
		default="base",
		type=str,
		help="Backbone size identifier (e.g. small/base/large/basev2, etc.)",
	)
	parser.add_argument(
		"--patch-size",
		default=16,
		type=int,
		help="Patch size for ViT-style backbones",
	)
	parser.add_argument(
		"--input-size",
		default=224,
		type=int,
		help="Input image resolution",
	)
	parser.add_argument(
		"--drop",
		type=float,
		default=0.0,
		metavar="PCT",
		help="Feed-forward dropout rate",
	)
	parser.add_argument(
		"--attn-drop",
		type=float,
		default=0.0,
		metavar="PCT",
		help="Attention dropout rate",
	)
	parser.add_argument(
		"--drop-path",
		type=float,
		default=0.1,
		metavar="PCT",
		help="Stochastic depth rate",
	)
	parser.add_argument(
		"--block-type",
		type=str,
		default="base",
		help="Transformer block type for ViT/RoPE/KoPE backbones",
	)
	parser.add_argument(
		"--num-register-tokens",
		type=int,
		default=0,
		help="Number of DINO register tokens",
	)
	parser.add_argument(
		"--block-chunks",
		type=int,
		default=0,
		help="Chunk count for block lists (FSDP compatibility)",
	)
	parser.add_argument(
		"--layerscale",
		type=float,
		default=None,
		help="LayerScale init value",
	)
	parser.add_argument(
		"--ffn-layer",
		type=str,
		default=None,
		help="Override FFN implementation (e.g. swiglu_fused). Defaults to backbone preset.",
	)
	parser.add_argument(
		"--drop-path-uniform",
		action="store_true",
		help="Use a single stochastic depth rate across transformer blocks",
	)
	parser.add_argument(
		"--no-drop-path-uniform",
		action="store_false",
		dest="drop_path_uniform",
	)
	parser.set_defaults(drop_path_uniform=False)
	parser.add_argument(
		"--no-qkv-bias",
		action="store_false",
		dest="qkv_bias",
		help="Disable qkv bias in attention projection",
	)
	parser.add_argument(
		"--no-ffn-bias",
		action="store_false",
		dest="ffn_bias",
		help="Disable bias in FFN layers",
	)
	parser.add_argument(
		"--no-proj-bias",
		action="store_false",
		dest="proj_bias",
		help="Disable bias in patch/linear projections",
	)
	parser.set_defaults(qkv_bias=True, ffn_bias=True, proj_bias=True)
	parser.add_argument(
		"--qknorm",
		action="store_true",
		help="Enable QK-Norm in attention blocks when supported",
	)
	parser.add_argument(
		"--max-resolution",
		type=int,
		default=14,
		help="Maximum rope resolution divisor",
	)
	parser.add_argument(
		"--base",
		type=int,
		default=20,
		help="Base for rotary embeddings",
	)
	parser.add_argument(
		"--gradient-checkpointing",
		action="store_true",
		help="Enable gradient checkpointing inside backbone blocks",
	)
	parser.add_argument(
		"--torch-compile",
		action="store_true",
		help="Run the classifier through torch.compile before training",
	)
	parser.add_argument(
		"--compile-mode",
		type=str,
		default="default",
		help="torch.compile mode (default/ reduce-overhead/ max-autotune)",
	)
	parser.add_argument(
		"--drop-masks",
		action="store_true",
		help="Enable mask token support (MAE-style) in ViT/KoPE backbones",
	)
	parser.add_argument(
		"--learnable-pos-embed",
		action="store_true",
		help="Add learnable positional embeddings (RoPE/KoPE only)",
	)
	parser.add_argument(
		"--no-learnable-pos-embed",
		action="store_false",
		dest="learnable_pos_embed",
	)
	parser.set_defaults(learnable_pos_embed=False)
	parser.add_argument(
		"--kope-gamma",
		type=float,
		default=0.005,
		help="KoPE phase update step size",
	)
	parser.add_argument(
		"--kope-update-ext-phase",
		action="store_true",
		help="Allow KoPE to update external token phases",
	)
	parser.add_argument(
		"--kope-learn-phase-gamma",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Learn KoPE phase gamma instead of keeping it fixed",
	)
	parser.add_argument(
		"--share-kope-coupling",
		action="store_true",
		help="Share KoPE coupling weights across transformer blocks",
	)
	parser.add_argument(
		"--kope-coupling-use-vo-proj",
		type=str2bool,
		default=True,
		metavar="{True,False}",
		help="Use the value/output projection inside the KoPE coupling",
	)
	parser.add_argument(
		"--kope-coupling-use-tanh",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Replace KoPE coupling softmax with tanh-based gating",
	)
	parser.add_argument(
		"--kope-vo-rotation",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Rotate KoPE attention values with phase and demodulate outputs",
	)
	parser.add_argument(
		"--kope-mix",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Enable learnable KoPE phase mixing before rotary rotations",
	)
	parser.add_argument(
		"--kope-mix-init-gain",
		type=float,
		default=0.4,
		help="Initialization gain applied to KoPE phase mixing weights",
	)
	parser.add_argument(
		"--kope-mix-phase-norm",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Normalize mixed KoPE phases onto unit circle (with optional checkpointing overhead)",
	)
	parser.add_argument(
		"--kope-checkpoint-phase-update",
		type=str2bool,
		default=True,
		metavar="{True,False}",
		help="Use checkpointing around KoPE phase update to save activations",
	)
	parser.add_argument(
		"--kope-checkpoint-ffn-func",
		type=str2bool,
		default=True,
		metavar="{True,False}",
		help="Checkpoint KoPE block FFN residual to reduce memory",
	)
	parser.add_argument(
		"--kope-checkpoint-rotation",
		type=str2bool,
		default=True,
		metavar="{True,False}",
		help="Checkpoint KoPE rotation to save memory",
	)
	parser.add_argument(
		"--kope-fixed-layout",
		type=str2bool,
		default=None,
		metavar="{True,False}",
		help="Force KoPE coupling outputs into interleaved (cos,sin) layout",
	)
	parser.add_argument(
		"--kope-coupling-qknorm",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Apply RMSNorm to KoPE coupling q/k projections",
	)
	parser.add_argument(
		"--kope-coupling-qknorm-learn",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Make KoPE coupling qk scale learnable when qknorm is enabled",
	)
	parser.add_argument(
		"--kope-coupling-qk-multilayer",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Use a small MLP instead of linear layers for KoPE coupling q/k projections",
	)
	parser.add_argument(
		"--kope-no-phase-norm",
		type=str2bool,
		default=False,
		metavar="{True,False}",
		help="Disable KoPE phase normalization (use Identity)",
	)

	parser.add_argument("--model-ema", action="store_true")
	parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
	parser.set_defaults(model_ema=True)
	parser.add_argument("--model-ema-decay", type=float, default=0.99996)
	parser.add_argument("--model-ema-force-cpu", action="store_true", default=False)

	# optimizer options (matching DeiT III)
	parser.add_argument("--opt", default="adamw", type=str)
	parser.add_argument("--opt-eps", default=1e-8, type=float)
	parser.add_argument("--opt-betas", default=None, type=float, nargs="+")
	parser.add_argument("--clip-grad", type=float, default=None, metavar="NORM")
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--weight-decay", type=float, default=0.05)

	# scheduler options
	parser.add_argument("--sched", default="cosine", type=str)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--lr-noise", type=float, nargs="+", default=None)
	parser.add_argument("--lr-noise-pct", type=float, default=0.67)
	parser.add_argument("--lr-noise-std", type=float, default=1.0)
	parser.add_argument("--warmup-lr", type=float, default=1e-6)
	parser.add_argument("--min-lr", type=float, default=1e-5)
	parser.add_argument("--decay-epochs", type=float, default=30)
	parser.add_argument("--warmup-epochs", type=int, default=5)
	parser.add_argument("--cooldown-epochs", type=int, default=10)
	parser.add_argument("--patience-epochs", type=int, default=10)
	parser.add_argument("--decay-rate", type=float, default=0.1)

	# augmentation knobs (DeiT III defaults)
	parser.add_argument("--color-jitter", type=float, default=0.3)
	parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
	parser.add_argument("--smoothing", type=float, default=0.1)
	parser.add_argument("--train-interpolation", type=str, default="bicubic")
	parser.add_argument("--repeated-aug", action="store_true")
	parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
	parser.set_defaults(repeated_aug=True)
	parser.add_argument("--train-mode", action="store_true")
	parser.add_argument("--no-train-mode", action="store_false", dest="train_mode")
	parser.set_defaults(train_mode=True)
	parser.add_argument("--ThreeAugment", action="store_true")
	parser.add_argument("--src", action="store_true")

	parser.add_argument("--reprob", type=float, default=0.25)
	parser.add_argument("--remode", type=str, default="pixel")
	parser.add_argument("--recount", type=int, default=1)
	parser.add_argument("--resplit", action="store_true", default=False)

	# mixup / cutmix
	parser.add_argument("--mixup", type=float, default=0.8)
	parser.add_argument("--cutmix", type=float, default=1.0)
	parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
	parser.add_argument("--mixup-prob", type=float, default=1.0)
	parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
	parser.add_argument("--mixup-mode", type=str, default="batch")

	# distillation controls
	parser.add_argument("--teacher-model", default="regnety_160", type=str)
	parser.add_argument("--teacher-path", type=str, default="")
	parser.add_argument(
		"--distillation-type",
		default="none",
		choices=["none", "soft", "hard"],
		type=str,
	)
	parser.add_argument("--distillation-alpha", default=0.5, type=float)
	parser.add_argument("--distillation-tau", default=1.0, type=float)

	# additional loss toggles
	parser.add_argument("--bce-loss", action="store_true")
	parser.add_argument("--cosub", action="store_true")
	parser.add_argument("--attn-only", action="store_true")
	parser.add_argument("--finetune", default="", type=str)
	parser.add_argument(
		"--use-phase-loss",
		action="store_true",
		help="Enable KoPE phase clustering loss during training",
	)
	parser.add_argument(
		"--phase-loss-weight",
		type=float,
		default=0.1,
		help="Weight applied to the KoPE phase clustering loss when enabled",
	)
	parser.add_argument(
		"--phase-attractor-n",
		type=int,
		default=32,
		help="Number of equally spaced attractors on the unit circle for KoPE phase loss",
	)
	parser.add_argument(
		"--phase-local-sync-deg",
		type=float,
		default=15.0,
		help="Local synchronization degree parameter for phase loss"
	)
	parser.add_argument(
		"--phase-consensus-sharpness",
		type=float,
		default=10.0,
		help="Consensus sharpness parameter for phase loss"
	)
	parser.add_argument(
		"--phase-loss-start-epoch",
		type=int,
		default=0,
		help="Epoch index to begin applying KoPE phase loss (earlier epochs keep weight at 0)",
	)
	parser.add_argument(
		"--analyze-attention",
		action="store_true",
		help="Enable attention analysis (entropy, gini, phase sync) during evaluation",
	)

	# dataset parameters
	parser.add_argument("--data-path", type=str, default="/datasets01/imagenet_full_size/061417/")
	parser.add_argument(
		"--data-set",
		default="IMNET",
		choices=["CIFAR", "IMNET", "INAT", "INAT19"],
	)
	parser.add_argument(
		"--imagenet-train-subset",
		type=str,
		default="full",
		# choices=["full", "1percent", "10percent"],
		help="Use full ImageNet (default) or the SimCLR 1%/10% training subsets",
	)
	parser.add_argument(
		"--inat-category",
		default="name",
		choices=[
			"kingdom",
			"phylum",
			"class",
			"order",
			"supercategory",
			"family",
			"genus",
			"name",
		],
	)

	parser.add_argument("--output-dir", default="")
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--resume", default="", help="Resume from checkpoint")
	parser.add_argument("--eval", action="store_true", help="Evaluation only")
	parser.add_argument("--eval-crop-ratio", default=0.875, type=float)
	parser.add_argument(
		"--eval-interval",
		type=int,
		default=1,
		help="Run validation every N epochs (default 1 = every epoch)",
	)
	parser.add_argument("--dist-eval", action="store_true", default=False)
	parser.add_argument("--num_workers", default=8, type=int)
	parser.add_argument("--pin-mem", action="store_true")
	parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
	parser.set_defaults(pin_mem=True)
	parser.add_argument(
		"--persistent-workers",
		action="store_true",
		help="Keep dataloader workers alive across epochs to avoid startup stalls",
	)
	parser.add_argument(
		"--no-auto-resume",
		action="store_false",
		dest="auto_resume",
		help="Disable automatic checkpoint discovery inside --output-dir",
	)
	parser.set_defaults(auto_resume=True)

	# distributed setup
	parser.add_argument("--distributed", action="store_true", default=False)
	parser.add_argument("--world_size", default=1, type=int)
	parser.add_argument("--dist_url", default="env://")

	return parser


def _build_simdinov2_backbone(args: argparse.Namespace) -> torch.nn.Module:
	arch = args.arch_name
	variant = args.model_variant
	patch_size = args.patch_size
	img_size = args.input_size
	common_transformer_kwargs = dict(
		layerscale=args.layerscale,
		qkv_bias=args.qkv_bias,
		ffn_bias=args.ffn_bias,
		proj_bias=args.proj_bias,
		drop_path_uniform=args.drop_path_uniform,
		qknorm=args.qknorm,
		max_resolution=args.max_resolution,
		base=args.base,
	)
	if args.ffn_layer:
		common_transformer_kwargs["ffn_layer"] = args.ffn_layer

	if arch == "vit":
		fn_name = f"vit_{variant}"
		if not hasattr(vit_models, fn_name):
			raise ValueError(f"Unsupported ViT variant '{variant}' for SimDINOv2")
		builder = getattr(vit_models, fn_name)
		backbone = builder(
			patch_size=patch_size,
			img_size=img_size,
			drop_path_rate=args.drop_path,
			ffn_drop=args.drop,
			attn_drop=args.attn_drop,
			block=args.block_type,
			num_register_tokens=args.num_register_tokens,
			block_chunks=args.block_chunks,
			gradient_checkpointing=args.gradient_checkpointing,
			drop_masks=args.drop_masks,
			**common_transformer_kwargs,
		)
	elif arch == "vit_rope":
		fn_name = f"vit_rope_{variant}"
		if not hasattr(rope_models, fn_name):
			raise ValueError(f"Unsupported ViT-RoPE variant '{variant}' for SimDINOv2")
		builder = getattr(rope_models, fn_name)
		backbone = builder(
			patch_size=patch_size,
			img_size=img_size,
			drop_path_rate=args.drop_path,
			ffn_drop=args.drop,
			attn_drop=args.attn_drop,
			block=args.block_type,
			num_register_tokens=args.num_register_tokens,
			block_chunks=args.block_chunks,
			gradient_checkpointing=args.gradient_checkpointing,
			use_learnable_pos_embed=args.learnable_pos_embed,
			**common_transformer_kwargs,
		)
	elif arch == "vit_kope":
		fn_name = f"vit_kope_{variant}"
		if not hasattr(kope_models, fn_name):
			raise ValueError(f"Unsupported ViT-KoPE variant '{variant}' for SimDINOv2")
		builder = getattr(kope_models, fn_name)
		print("Train Coupling use vo proj:", args.kope_coupling_use_vo_proj)
		backbone = builder(
			patch_size=patch_size,
			img_size=img_size,
			drop_path_rate=args.drop_path,
			ffn_drop=args.drop,
			attn_drop=args.attn_drop,
			block_type=args.block_type,
			num_register_tokens=args.num_register_tokens,
			block_chunks=args.block_chunks,
			gradient_checkpointing=args.gradient_checkpointing,
			drop_masks=args.drop_masks,
			# KoPE params
			kope_gamma=args.kope_gamma,
			learn_phase_gamma=args.kope_learn_phase_gamma,
			update_ext_token_phase=args.kope_update_ext_phase,
			share_kope_coupling=args.share_kope_coupling,
			use_learnable_pos_embed=args.learnable_pos_embed,
			# Coupling
			coupling_use_vo_proj=args.kope_coupling_use_vo_proj,
			coupling_use_tanh=args.kope_coupling_use_tanh,
			kope_vo_rotation=args.kope_vo_rotation,
			fixed_layout=args.kope_fixed_layout,
			coupling_qknorm=args.kope_coupling_qknorm,
			coupling_qknorm_learn=args.kope_coupling_qknorm_learn,
			coupling_qk_multilayer=args.kope_coupling_qk_multilayer,
			# KoPE mix
			kope_mix=args.kope_mix,
			kope_mix_init_gain=args.kope_mix_init_gain,
			kope_mix_phase_norm=args.kope_mix_phase_norm,
			# Checkpointing
			checkpoint_phase_update=args.kope_checkpoint_phase_update,
			checkpoint_ffn_func=args.kope_checkpoint_ffn_func,
			checkpoint_rotation=args.kope_checkpoint_rotation,
			# Misc
			start_kope_idx=0,
			no_phase_norm=args.kope_no_phase_norm,
			**common_transformer_kwargs,
		)
	else:
		raise ValueError(f"Unsupported arch '{arch}'")

	return backbone


def _build_classifier(args: argparse.Namespace, num_classes: int) -> ModelWithClassifier:
	backbone = _build_simdinov2_backbone(args)
	embed_dim = getattr(backbone, "embed_dim", None)
	if embed_dim is None:
		embed_dim = getattr(backbone, "num_features", None)
	if embed_dim is None:
		raise AttributeError("Backbone missing 'embed_dim'/'num_features' attribute")
	model = ModelWithClassifier(backbone, embed_dim, num_classes=num_classes, use_multi_stage_feat=False, use_cls=True)
	return model


def _auto_discover_checkpoint(output_dir: Path) -> Optional[Path]:
	preferred_names = ["checkpoint.pth", "checkpoint-last.pth"]
	for name in preferred_names:
		candidate = output_dir / name
		if candidate.is_file():
			return candidate
	candidates = [p for p in output_dir.glob("checkpoint*.pth") if p.is_file()]
	if not candidates:
		return None
	candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return candidates[0]


def train(args: argparse.Namespace) -> None:
	deit_utils.init_distributed_mode(args)
	device = torch.device(args.device)

	seed = args.seed + deit_utils.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)
	cudnn.benchmark = True

	dataset_train, args.nb_classes = deit_datasets.build_dataset(is_train=True, args=args)
	dataset_val, _ = deit_datasets.build_dataset(is_train=False, args=args)

	if args.distributed:
		num_tasks = deit_utils.get_world_size()
		global_rank = deit_utils.get_rank()
		if args.repeated_aug:
			sampler_train = deit_samplers.RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
		else:
			sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
		if args.dist_eval:
			if len(dataset_val) % num_tasks != 0 and deit_utils.is_main_process():
				print(
					"Warning: Enabling distributed evaluation with a validation set not divisible by the number of tasks."
				)
			sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
		else:
			sampler_val = torch.utils.data.SequentialSampler(dataset_val)
	else:
		sampler_train = torch.utils.data.RandomSampler(dataset_train)
		sampler_val = torch.utils.data.SequentialSampler(dataset_val)

	data_loader_train = torch.utils.data.DataLoader(
		dataset_train,
		sampler=sampler_train,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True,
		persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
	)
	if args.ThreeAugment:
		data_loader_train.dataset.transform = deit_augment.new_data_aug_generator(args)

	data_loader_val = torch.utils.data.DataLoader(
		dataset_val,
		sampler=sampler_val,
		batch_size=int(1.5 * args.batch_size),
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False,
		persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
	)

	mixup_fn: Optional[Mixup]
	mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
	if mixup_active:
		mixup_fn = Mixup(
			mixup_alpha=args.mixup,
			cutmix_alpha=args.cutmix,
			cutmix_minmax=args.cutmix_minmax,
			prob=args.mixup_prob,
			switch_prob=args.mixup_switch_prob,
			mode=args.mixup_mode,
			label_smoothing=args.smoothing,
			num_classes=args.nb_classes,
		)
	else:
		mixup_fn = None

	model = _build_classifier(args, args.nb_classes)
	model.to(device)

	if args.finetune:
		if args.finetune.startswith('https'):
			checkpoint = torch.hub.load_state_dict_from_url(
				args.finetune, map_location='cpu', check_hash=True)
		else:
			checkpoint = torch.load(args.finetune, weights_only=False, map_location='cpu')

		print("Load ckpt from %s" % args.finetune)
		checkpoint_model = None
		for model_key in ['model_ema', 'model']:
			if model_key in checkpoint and checkpoint[model_key] is not None:
				print(f"Load state_dict by model_key = {model_key}")
				checkpoint_model = checkpoint[model_key]
				break
		if checkpoint_model is None:
			checkpoint_model = checkpoint

		state_dict = model.state_dict()

		# Detect if checkpoint is backbone-only or full model
		# If keys don't start with 'feature_model.', assume it's a backbone checkpoint
		# and prepend 'feature_model.' to keys.
		new_checkpoint_model = {}
		for k, v in checkpoint_model.items():
			# Clean common prefixes potentially left by DDP or compile
			for prefix in ["module.", "_orig_mod."]:
				if k.startswith(prefix):
					k = k[len(prefix):]
				# Double check if multiple prefixes accumulated (e.g. module._orig_mod.)
				if k.startswith(prefix):
					k = k[len(prefix):]

			if not k.startswith('feature_model.') and not k.startswith('head.'):
				new_key = 'feature_model.' + k
			else:
				new_key = k
			new_checkpoint_model[new_key] = v
		checkpoint_model = new_checkpoint_model

		# Handle head mismatch
		for k in ['head.weight', 'head.bias']:
			if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
				print(f"Removing key {k} from pretrained checkpoint")
				del checkpoint_model[k]

		# interpolate position embedding
		if 'feature_model.pos_embed' in checkpoint_model:
			pos_embed_checkpoint = checkpoint_model['feature_model.pos_embed']
			embedding_size = pos_embed_checkpoint.shape[-1]
			num_patches = model.feature_model.patch_embed.num_patches
			num_extra_tokens = model.feature_model.pos_embed.shape[-2] - num_patches
			# height (== width) for the checkpoint position embedding
			orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
			# height (== width) for the new position embedding
			new_size = int(num_patches ** 0.5)
			# class_token and dist_token are kept unchanged
			if orig_size != new_size:
				print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
				extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
				# only the position tokens are interpolated
				pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
				pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
				pos_tokens = torch.nn.functional.interpolate(
					pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
				pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
				new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
				checkpoint_model['feature_model.pos_embed'] = new_pos_embed

		msg = model.load_state_dict(checkpoint_model, strict=False)
		print(msg)

	if args.torch_compile:
		if not hasattr(torch, "compile"):
			raise RuntimeError("torch.compile is unavailable in this PyTorch build")
		compile_kwargs = {}
		if args.compile_mode:
			compile_kwargs["mode"] = args.compile_mode
		model = torch.compile(model, **compile_kwargs)

	if args.attn_only:
		for name, param in model.named_parameters():
			requires_grad = ".attn." in name or name.startswith("head")
			param.requires_grad = requires_grad
	model_ema: Optional[ModelEma] = None
	if args.model_ema:
		model_ema = ModelEma(model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else "")

	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(
			model,
			device_ids=[args.gpu] if hasattr(args, "gpu") else None,
			find_unused_parameters=True,
		)
		model_without_ddp = model.module
	else:
		model_without_ddp = model

	n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
	if deit_utils.is_main_process():
		print(f"Number of trainable parameters: {n_parameters}")
		print(model_without_ddp)
	if not args.unscale_lr:
		scale = args.batch_size * deit_utils.get_world_size() / 512.0
		args.lr = args.lr * scale

	optimizer = create_optimizer(args, model_without_ddp)
	loss_scaler = NativeScaler()
	lr_scheduler, _ = create_scheduler(args, optimizer)

	criterion_base: torch.nn.Module
	if mixup_active:
		criterion_base = SoftTargetCrossEntropy()
	elif args.smoothing > 0.0:
		criterion_base = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
	else:
		criterion_base = torch.nn.CrossEntropyLoss()
	if args.bce_loss:
		criterion_base = torch.nn.BCEWithLogitsLoss()

	use_phase_loss = args.use_phase_loss if hasattr(args, "use_phase_loss") else False
	if use_phase_loss:
		phase_loss_weight = args.phase_loss_weight
		#phase_criterion = deit_losses.PhaseLoss(args.phase_attractor_n)
		#phase_criterion = deit_losses.PhaseLoss(args.phase_local_sync_deg)
		phase_criterion = deit_losses.PhaseLoss(args.phase_consensus_sharpness)
	else:
		phase_loss_weight = 0.0
		phase_criterion = None
	phase_loss_start_epoch = getattr(args, "phase_loss_start_epoch", 0)

	teacher_model = None
	if args.distillation_type != "none":
		if not args.teacher_path:
			raise ValueError("--teacher-path must be provided when using distillation")
		print(f"Creating teacher model: {args.teacher_model}")
		teacher_model = create_model(
			args.teacher_model,
			pretrained=False,
			num_classes=args.nb_classes,
			global_pool="avg",
		)
		if args.teacher_path.startswith("https"):
			checkpoint = torch.hub.load_state_dict_from_url(
				args.teacher_path, map_location="cpu", check_hash=True
			)
		else:
			checkpoint = torch.load(args.teacher_path, weights_only=False, map_location="cpu")
		teacher_model.load_state_dict(checkpoint["model"])
		teacher_model.to(device)
		teacher_model.eval()

	criterion = deit_losses.DistillationLoss(
		criterion_base,
		teacher_model,
		args.distillation_type,
		args.distillation_alpha,
		args.distillation_tau,
	)

	output_dir = Path(args.output_dir) if args.output_dir else None
	if output_dir and deit_utils.is_main_process():
		output_dir.mkdir(parents=True, exist_ok=True)
	if args.auto_resume and not args.resume and output_dir:
		auto_resume_path = _auto_discover_checkpoint(output_dir)
		if auto_resume_path is not None:
			args.resume = str(auto_resume_path)
			if deit_utils.is_main_process():
				print(f"Auto-resuming from checkpoint: {args.resume}")

	max_accuracy = 0.0
	if args.resume:
		checkpoint = torch.load(args.resume, weights_only=False, map_location="cpu")
		model_without_ddp.load_state_dict(checkpoint["model"])
		if not args.eval and all(k in checkpoint for k in ["optimizer", "lr_scheduler", "epoch"]):
			optimizer.load_state_dict(checkpoint["optimizer"])
			lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
			args.start_epoch = checkpoint["epoch"] + 1
			if args.model_ema and model_ema is not None and "model_ema" in checkpoint:
				deit_utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
			if "scaler" in checkpoint:
				loss_scaler.load_state_dict(checkpoint["scaler"])
		if "best_acc1" in checkpoint:
			max_accuracy = float(checkpoint["best_acc1"])

	# lr_scheduler.step(args.start_epoch)

	if args.eval:
		stats = deit_engine.evaluate(data_loader_val, model, device, args=args)
		print(
			f"Accuracy of the network on the {len(dataset_val)} validation images: {stats['acc1']:.1f}%"
		)
		return

	print(f"Start training for {args.epochs} epochs")
	start_time = time.time()
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			data_loader_train.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

		lr_scheduler.step(epoch)

		train_stats = deit_engine.train_one_epoch(
			model,
			criterion,
			data_loader_train,
			optimizer,
			device,
			epoch,
			loss_scaler,
			args.clip_grad,
			model_ema,
			mixup_fn,
			use_phase_loss=use_phase_loss,
			phase_loss_weight=phase_loss_weight if epoch >= phase_loss_start_epoch else 0.0,
			phase_criterion=phase_criterion,
			set_training_mode=args.train_mode,
			args=args,
		)

		# lr_scheduler.step(epoch)

		if output_dir:
			checkpoint = {
				"model": model_without_ddp.state_dict(),
				"optimizer": optimizer.state_dict(),
				"lr_scheduler": lr_scheduler.state_dict(),
				"epoch": epoch,
				"model_ema": get_state_dict(model_ema),
				"scaler": loss_scaler.state_dict(),
				"args": args,
				"best_acc1": max_accuracy,
			}
			deit_utils.save_on_master(checkpoint, output_dir / "checkpoint.pth")

		run_eval = ((epoch + 1) % max(1, args.eval_interval) == 0)
		test_stats = None
		if run_eval:
			test_stats = deit_engine.evaluate(data_loader_val, model, device, args=args)
			acc1 = test_stats["acc1"]
			print(f"Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.1f}%")
			if max_accuracy < acc1:
				max_accuracy = acc1
				if output_dir:
					best_ckpt = {
						"model": model_without_ddp.state_dict(),
						"optimizer": optimizer.state_dict(),
						"lr_scheduler": lr_scheduler.state_dict(),
						"epoch": epoch,
						"model_ema": get_state_dict(model_ema),
						"scaler": loss_scaler.state_dict(),
						"args": args,
						"best_acc1": max_accuracy,
					}
					deit_utils.save_on_master(best_ckpt, output_dir / "best_checkpoint.pth")
		else:
			print(f"Skipping evaluation at epoch {epoch} (eval_interval={args.eval_interval})")

		if run_eval:
			print(f"Max accuracy: {max_accuracy:.2f}%")

		log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}
		if run_eval and test_stats is not None:
			log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

		if output_dir and deit_utils.is_main_process():
			with (output_dir / "log.txt").open("a") as f:
				f.write(json.dumps(log_stats) + "\n")

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print(f"Training time {total_time_str}")


def main(argv: Optional[Sequence[str]] = None) -> None:
	parser = argparse.ArgumentParser(parents=[get_args_parser()])
	args = parser.parse_args(argv)
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	train(args)


if __name__ == "__main__":
	main()
