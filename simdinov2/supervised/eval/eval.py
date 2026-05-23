# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluation script for supervised SimDINOv2 checkpoints on ImageNet splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from simdinov2.eval.finetuning import ModelWithClassifier
from simdinov2.models import vision_transformer as vit_models
from simdinov2.models import vision_transformer_kope as kope_models
from simdinov2.models import vision_transformer_rope as rope_models

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_IMAGENET_V2_DIRNAMES: Dict[str, str] = {
    "matched-frequency": "imagenetv2-matched-frequency-format-val",
    "threshold-0.7": "imagenetv2-threshold0.7-format-val",
    "top-images": "imagenetv2-top-images-format-val",
}


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def compute_topk(output: Tensor, target: Tensor, topk: Sequence[int] = (1, 5)) -> List[Tensor]:
    with torch.no_grad():
        max_k = max(topk)
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.unsqueeze(0))
        res: List[Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Tensor, int, str]:
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


class RealLabelEvaluator:
    def __init__(self, filenames: Sequence[str], real_json_path: Optional[str] = None) -> None:
        try:
            from timm.data import RealLabelsImagenet
        except ImportError as err:  # pragma: no cover
            raise RuntimeError("Real labels evaluation requires timm to be installed") from err
        self._backend = RealLabelsImagenet(filenames=filenames, real_json=real_json_path)

    def update(self, logits: Tensor) -> None:
        self._backend.add_result(logits.detach().cpu())

    def compute(self) -> Tuple[float, float]:
        accuracy = self._backend.get_accuracy()
        top1 = float(accuracy.get(1, 0.0))
        top5 = float(accuracy.get(5, 0.0))
        return top1, top5


def _clean_state_dict_prefixes(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    prefixes = ("module.", "_orig_mod.")
    cleaned = state_dict.__class__()
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _resolve_imagenet_val_dir(path_str: str) -> str:
    base_path = Path(path_str)
    if (base_path / "val").is_dir():
        return str(base_path / "val")
    return str(base_path)


def _resolve_imagenet_v2_location(path_str: str, variant: str) -> str:
    subset_dir = _IMAGENET_V2_DIRNAMES[variant]
    base_path = Path(path_str)
    if base_path.name == subset_dir:
        return str(base_path.parent)
    return str(base_path)


def _resolve_checkpoint_path(checkpoint: Path, explicit_name: Optional[str]) -> Path:
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint path '{checkpoint}' is neither a file nor a directory")
    search_order = [explicit_name] if explicit_name else [
        "model.pth",            # release format
        "best_checkpoint.pth",  # training format (best val acc)
        "checkpoint.pth",       # training format (last)
        "checkpoint-last.pth",
    ]
    for name in search_order:
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate checkpoint inside '{checkpoint}'. Looked for: {', '.join(search_order)}"
    )


def _apply_default_backbone_args(args: argparse.Namespace) -> None:
    defaults: Dict[str, Any] = {
        "arch_name": "vit",
        "model_variant": "base",
        "patch_size": 16,
        "input_size": 224,
        "layerscale": 0.1, #None,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_uniform": False,
        "qknorm": False,
        "max_resolution": 14,
        "base": 20,
        "ffn_layer": None,
        "drop_path": 0.0,
        "drop": 0.0,
        "attn_drop": 0.0,
        "block_type": "nested",
        "num_register_tokens": 4, #0,
        "block_chunks": 4, #1,
        "gradient_checkpointing": False,
        "drop_masks": False,
        "kope_gamma": 0.05,
        "kope_update_ext_phase": True,
        "share_kope_coupling": True,
        "learnable_pos_embed": True,
        "kope_coupling_use_vo_proj": False,
        "kope_coupling_use_tanh": False,
        "kope_vo_rotation": True,
        "kope_fixed_layout": None,
        "kope_coupling_qknorm": True, #False,
        "kope_coupling_qknorm_learn": True, #False,
        "kope_coupling_qk_multilayer": False,
        "kope_mix": True,
        "kope_mix_init_gain": 0.1,
        "kope_mix_phase_norm": True, #False,
        "kope_checkpoint_phase_update": True,
        "kope_checkpoint_ffn_func": True,
        "kope_checkpoint_rotation": True,
        "kope_no_phase_norm": True, #False,
        "kope_learn_phase_gamma": True, #False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


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
            learn_phase_gamma=args.kope_learn_phase_gamma,
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
    model = ModelWithClassifier(
        backbone,
        embed_dim,
        num_classes=num_classes,
        use_multi_stage_feat=False,
        use_cls=True,
    )
    return model


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    use_ema: bool,
    device: torch.device,
) -> Tuple[torch.nn.Module, argparse.Namespace]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------
    # Detect checkpoint format. The release format produced by
    # ``tools/convert_supervised_checkpoint.py`` is a slim dict with
    # ``{backbone, head, meta}`` (and no optimizer / EMA / argparse.Namespace).
    # The legacy training checkpoint has ``{model, model_ema?, args, ...}``.
    # ------------------------------------------------------------------
    is_release_format = (
        "backbone" in checkpoint and "head" in checkpoint and "meta" in checkpoint
    )

    if is_release_format:
        meta = checkpoint["meta"]
        train_args = argparse.Namespace(**meta)
        if use_ema and meta.get("source_model_field") != "model_ema":
            print(
                "Warning: --use-ema was passed but the release checkpoint already "
                "contains a single backbone "
                f"(source_model_field={meta.get('source_model_field')!r}). "
                "The flag is ignored for release-format checkpoints."
            )
    else:
        train_args = checkpoint.get("args")
        if train_args is None:
            raise KeyError("Checkpoint is missing stored training arguments under 'args'")
        if not isinstance(train_args, argparse.Namespace):
            train_args = argparse.Namespace(**getattr(train_args, "__dict__", dict(train_args)))

    if not hasattr(train_args, "nb_classes"):
        train_args.nb_classes = 1000
    _apply_default_backbone_args(train_args)

    model = _build_classifier(train_args, train_args.nb_classes)

    if is_release_format:
        # Re-attach the ``feature_model.`` / ``head.`` prefixes expected by the
        # wrapping ``ModelWithClassifier`` module.
        state_dict: Dict[str, Tensor] = {}
        for k, v in checkpoint["backbone"].items():
            state_dict[f"feature_model.{k}"] = v
        for k, v in checkpoint["head"].items():
            state_dict[f"head.{k}"] = v
    else:
        state_key = "model_ema" if use_ema else "model"
        state_dict = checkpoint.get(state_key)
        if use_ema and (state_dict is None or not state_dict):
            state_key = "model"
            state_dict = checkpoint.get(state_key)
        if state_dict is None:
            available = ", ".join(sorted(checkpoint.keys()))
            raise KeyError(
                f"Checkpoint does not contain '{state_key}' weights. Available keys: {available}"
            )
        state_dict = _clean_state_dict_prefixes(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing {len(missing)} keys when loading weights: {missing}")
    if unexpected:
        print(f"Warning: unexpected {len(unexpected)} keys when loading weights: {unexpected}")

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, train_args


def _build_eval_transform(image_size: int, crop_pct: float) -> transforms.Compose:
    transform_list = []
    if image_size > 32:
        resize_size = int(round(image_size / crop_pct))
        transform_list.append(
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC)
        )
        transform_list.append(transforms.CenterCrop(image_size))
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return transforms.Compose(transform_list)


def _create_loader(
    data_path: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    pin_mem: bool,
) -> DataLoader:
    resolved_path = _resolve_imagenet_val_dir(data_path)
    dataset = ImageFolderWithPaths(resolved_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    return loader


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    real_evaluator: Optional[RealLabelEvaluator] = None,
    description: str = "",
) -> Dict[str, float]:
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    model.eval()
    device_type = device.type
    total_batches = len(loader)
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if not isinstance(batch, (list, tuple)):
                raise TypeError(f"Unexpected batch type: {type(batch)!r}")
            if len(batch) == 3:
                images, targets, _ = batch
            elif len(batch) == 2:
                images, targets = batch
            else:
                raise ValueError("Unexpected batch structure from dataloader")
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            autocast_kwargs = dict(enabled=False)
            if amp_dtype is not None and device_type in {"cuda", "xpu"}:
                autocast_kwargs = dict(enabled=True, device_type=device_type, dtype=amp_dtype)
            with torch.autocast(**autocast_kwargs):  # type: ignore[arg-type]
                outputs = model(images)
            acc1, acc5 = compute_topk(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            top1_meter.update(acc1.item() * (100.0 / batch_size), batch_size)
            top5_meter.update(acc5.item() * (100.0 / batch_size), batch_size)
            if real_evaluator is not None:
                real_evaluator.update(outputs)
            if description:
                progress = (idx + 1) / total_batches
                print(f"[{description}] {idx + 1}/{total_batches} ({progress:.1%})", end="\r")
    elapsed = time.time() - start_time
    if description:
        print(f"[{description}] done in {elapsed:.1f}s" + " " * 20)
    results = {
        "acc1": top1_meter.avg,
        "acc5": top5_meter.avg,
        "time_sec": elapsed,
    }
    if real_evaluator is not None:
        real_top1, real_top5 = real_evaluator.compute()
        results["real_acc1"] = real_top1
        results["real_acc5"] = real_top5
    return results


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("SimDINOv2 supervised evaluation")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint file or directory")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Specific checkpoint filename when --checkpoint is a directory")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available")

    parser.add_argument("--imagenet-val", type=str, default=None, help="Path to ImageNet validation directory")
    parser.add_argument("--imagenet-real", type=str, default=None, help="Optional path to ImageNet ReaL images (defaults to --imagenet-val)")
    parser.add_argument("--imagenet-v2", type=str, default=None, help="Path to ImageNet-V2 dataset root (optional)")
    parser.add_argument(
        "--imagenet-v2-variant",
        type=str,
        default="matched-frequency",
        choices=["matched-frequency", "threshold-0.7", "top-images"],
        help="ImageNet-V2 subset to evaluate",
    )
    parser.add_argument("--real-labels-root", type=str, default=None, help="Directory containing real_labels.json (optional)")
    parser.add_argument("--no-real", action="store_true", help="Disable ImageNet ReaL evaluation")

    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--pin-mem", action="store_true", help="Pin dataloader memory")

    parser.add_argument("--input-size", type=int, default=None, help="Override input resolution")
    parser.add_argument("--crop-pct", type=float, default=None, help="Override evaluation crop percentage (default uses training args or 0.875)")
    parser.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"], help="Autocast precision for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")

    parser.add_argument("--output-json", type=str, default=None, help="Optional path to store metrics as JSON")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    amp_map = {
        "none": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    amp_dtype = amp_map[args.amp]

    checkpoint_path = _resolve_checkpoint_path(Path(args.checkpoint), args.checkpoint_name)
    model, train_args = _load_model_from_checkpoint(checkpoint_path, args.use_ema, device)

    image_size = args.input_size or getattr(train_args, "input_size", 224)
    crop_pct = args.crop_pct or getattr(train_args, "eval_crop_ratio", 0.875)
    transform = _build_eval_transform(image_size, crop_pct)

    results: Dict[str, Dict[str, float]] = {}

    val_path = args.imagenet_val
    if val_path:
        val_dir = _resolve_imagenet_val_dir(val_path)
        val_dataset = ImageFolderWithPaths(val_dir, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        real_evaluator: Optional[RealLabelEvaluator] = None
        if not args.no_real:
            filenames_source = getattr(val_dataset, "samples", getattr(val_dataset, "imgs"))
            filenames = [Path(path).name for path, _ in filenames_source]
            try:
                real_evaluator = RealLabelEvaluator(filenames, args.real_labels_root)
            except RuntimeError as exc:
                print(f"Warning: {exc}. Skipping ImageNet ReaL evaluation.")
                real_evaluator = None
        val_results = _evaluate_loader(
            model,
            val_loader,
            device,
            amp_dtype,
            real_evaluator=real_evaluator,
            description="ImageNet-val",
        )
        results["imagenet_val"] = val_results
        if (not args.no_real) and args.imagenet_real:
            real_dir = _resolve_imagenet_val_dir(args.imagenet_real)
            if Path(real_dir).resolve() == Path(val_dir).resolve():
                real_dir = None
        else:
            real_dir = None
        if real_dir is not None:
            real_dataset = ImageFolderWithPaths(real_dir, transform=transform)
            real_loader = DataLoader(
                real_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
            filenames_source = getattr(real_dataset, "samples", getattr(real_dataset, "imgs"))
            filenames = [Path(path).name for path, _ in filenames_source]
            try:
                real_eval = RealLabelEvaluator(filenames, args.real_labels_root)
            except RuntimeError as exc:
                print(f"Warning: {exc}. Skipping ImageNet ReaL evaluation.")
            else:
                real_results = _evaluate_loader(
                    model,
                    real_loader,
                    device,
                    amp_dtype,
                    real_evaluator=real_eval,
                    description="ImageNet-ReaL",
                )
                results["imagenet_real"] = {
                    "real_acc1": real_results.get("real_acc1", real_results.get("acc1", 0.0)),
                    "real_acc5": real_results.get("real_acc5", real_results.get("acc5", 0.0)),
                    "time_sec": real_results["time_sec"],
                }

    if args.imagenet_v2:
        try:
            from imagenetv2_pytorch import ImageNetV2Dataset
        except ImportError as exc:
            raise RuntimeError("ImageNet-V2 evaluation requires 'imagenetv2-pytorch' package") from exc
        v2_kwargs: Dict[str, Any] = {"transform": transform}
        location: Optional[str] = None
        if args.imagenet_v2:
            location = _resolve_imagenet_v2_location(args.imagenet_v2, args.imagenet_v2_variant)
            #subset_dir_name = _IMAGENET_V2_DIRNAMES[args.imagenet_v2_variant]
            #expected_dir = Path(location) / subset_dir_name
            #if not expected_dir.is_dir():
            #    raise FileNotFoundError(
            #        f"Expected ImageNet-V2 subset directory '{expected_dir}' not found."
            #    )
            v2_kwargs["location"] = location
        try:
            dataset_v2 = ImageNetV2Dataset(args.imagenet_v2_variant, **v2_kwargs)
        except TypeError:
            if location is None:
                dataset_v2 = ImageNetV2Dataset(args.imagenet_v2_variant, transform=transform)
            else:
                dataset_v2 = ImageNetV2Dataset(
                    args.imagenet_v2_variant,
                    transform=transform,
                    root=location,
                )
        loader_v2 = DataLoader(
            dataset_v2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        results["imagenet_v2"] = _evaluate_loader(
            model,
            loader_v2,
            device,
            amp_dtype,
            real_evaluator=None,
            description="ImageNet-V2",
        )

    if not results:
        raise RuntimeError("No evaluation dataset provided. Specify --imagenet-val and/or --imagenet-v2.")

    pretty = json.dumps(results, indent=2)
    print("Evaluation results:\n" + pretty)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
