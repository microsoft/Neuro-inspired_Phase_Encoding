# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convert a supervised training checkpoint to a clean release format.

The supervised trainer (``simdinov2/supervised/train.py``) saves
``checkpoint.pth`` / ``best_checkpoint.pth`` in the DeiT-III style, which
contains the optimizer / lr_scheduler / scaler / epoch / ``argparse.Namespace``
on top of the model weights.

This script reads such a training checkpoint and writes a slim release
checkpoint containing only model weights. The exact layout depends on
``--source``:

  * ``--source model`` (or ``--no-ema``): keep only the non-EMA model::

        {
            "backbone": <state_dict>,
            "head":     {"weight": ..., "bias": ...},
            "meta":     {... see below ...},
        }

  * ``--source ema``: keep only ``model_ema`` (promoted to backbone/head)::

        {
            "backbone": <EMA state_dict>,
            "head":     <EMA head>,
            "meta":     {"source_model_field": "model_ema", ...},
        }

  * ``--source both`` (default): keep both, with EMA suffixed::

        {
            "backbone":     <model state_dict>,
            "head":         {"weight": ..., "bias": ...},
            "backbone_ema": <model_ema state_dict>,     # if present
            "head_ema":     {"weight": ..., "bias": ...}, # if present
            "meta":         {...},
        }

``meta`` always contains the KoPE / training-config fields needed to rebuild
the model (``arch_name``, ``model_variant``, ``patch_size``, ``num_register_tokens``,
``block_type``, ``kope_*``, ...) plus provenance fields (``source``,
``source_model_field``, ``source_epoch``, ``source_best_acc1``).

Optionally also writes a ``.safetensors`` mirror (one tensor per fully-qualified
key, prefixed with ``backbone.`` / ``backbone_ema.`` / ``head.`` / ``head_ema.``)
plus a side-car ``<output>.meta.json`` so that downstream users can reconstruct
the model without ever ``pickle``-loading anything.

Example
-------
::

    # Release the non-EMA "small" model
    python tools/convert_supervised_checkpoint.py \\
        --input  checkpoints/supervised_vitkope_small_patch16_in1k_300e/best_checkpoint.pth \\
        --output release/vit_kope_small_in1k.pth \\
        --source model

    # Release the EMA "base" model
    python tools/convert_supervised_checkpoint.py \\
        --input  checkpoints/supervised_vitkope_base_patch16_in1k_300e/best_checkpoint.pth \\
        --output release/vit_kope_base_in1k.pth \\
        --source ema
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


# ----------------------------------------------------------------------------
# Fields to keep from the training ``argparse.Namespace`` so users can rebuild
# the model. Listed explicitly (allow-list) so we never leak unexpected paths
# (e.g. ``--data-path``) into the public artifact.
# ----------------------------------------------------------------------------
_META_FIELDS = (
    # backbone selection / geometry
    "arch_name",
    "model_variant",
    "patch_size",
    "input_size",
    "block_type",
    "num_register_tokens",
    "block_chunks",
    "layerscale",
    "ffn_layer",
    "qkv_bias",
    "ffn_bias",
    "proj_bias",
    "qknorm",
    "max_resolution",
    "base",
    "learnable_pos_embed",
    "drop_masks",
    # KoPE-specific (ignored if arch is not vit_kope)
    "kope_gamma",
    "kope_update_ext_phase",
    "kope_learn_phase_gamma",
    "share_kope_coupling",
    "kope_coupling_use_vo_proj",
    "kope_coupling_use_tanh",
    "kope_vo_rotation",
    "kope_mix",
    "kope_mix_init_gain",
    "kope_mix_phase_norm",
    "kope_fixed_layout",
    "kope_coupling_qknorm",
    "kope_coupling_qknorm_learn",
    "kope_coupling_qk_multilayer",
    "kope_no_phase_norm",
)


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Return a new state_dict with ``prefix`` removed from any key that has it."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _split_backbone_head(sd: Dict[str, torch.Tensor]):
    """``ModelWithClassifier`` keys look like ``feature_model.*`` (backbone) and
    ``head.*`` (linear classifier). Split and strip prefixes."""
    # ``torch.compile`` wraps the model in an ``OptimizedModule`` whose
    # state_dict prefixes every key with ``_orig_mod.``. DDP adds ``module.``.
    # Strip both so downstream code sees plain ``feature_model.`` / ``head.``.
    sd = _strip_prefix(sd, "_orig_mod.")
    sd = _strip_prefix(sd, "module.")
    backbone: Dict[str, torch.Tensor] = {}
    head: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("feature_model."):
            backbone[k[len("feature_model."):]] = v
        elif k.startswith("head."):
            head[k[len("head."):]] = v
        else:
            # Unknown top-level key. Keep under backbone for forward-compat.
            backbone[k] = v
    return backbone, head


def _extract_meta(args_obj: Any) -> Dict[str, Any]:
    """Build a JSON-friendly meta dict from the training argparse Namespace."""
    if args_obj is None:
        return {}
    meta: Dict[str, Any] = {}
    for field in _META_FIELDS:
        if hasattr(args_obj, field):
            val = getattr(args_obj, field)
            # Coerce to JSON-safe primitives.
            if isinstance(val, (str, int, float, bool, type(None))):
                meta[field] = val
            elif isinstance(val, (list, tuple)):
                meta[field] = list(val)
            else:
                meta[field] = str(val)
    return meta


def convert(
    input_path: Path,
    output_path: Path,
    *,
    source: str = "both",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read a training checkpoint and write a slim release checkpoint.

    Args:
        source: which state_dict(s) to keep.
            - ``"model"`` : keep only the non-EMA model as ``backbone``/``head``.
            - ``"ema"``   : keep only ``model_ema`` and promote it to
              ``backbone``/``head`` (no ``backbone_ema``/``head_ema`` keys).
            - ``"both"``  : keep ``backbone``/``head`` from ``model`` AND
              ``backbone_ema``/``head_ema`` from ``model_ema`` (if present).
    """
    if source not in ("model", "ema", "both"):
        raise ValueError(f"source must be one of model/ema/both, got {source!r}")

    print(f"[load] {input_path}")
    # ``weights_only=False`` is required because the training checkpoint
    # contains an ``argparse.Namespace``. This is safe for YOUR OWN
    # checkpoints; never run it on files you do not trust.
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise KeyError(
            f"Input checkpoint {input_path} has no 'model' key; not a supervised "
            "training checkpoint. Keys present: " + ", ".join(sorted(ckpt.keys()))
        )

    has_ema = "model_ema" in ckpt and ckpt["model_ema"] is not None
    if source == "ema" and not has_ema:
        raise KeyError(
            f"--source ema requested but {input_path} has no model_ema field."
        )

    release: Dict[str, Any] = {}
    if source == "ema":
        backbone, head = _split_backbone_head(ckpt["model_ema"])
        release["backbone"] = backbone
        release["head"] = head
        canonical_source_field = "model_ema"
    else:
        backbone, head = _split_backbone_head(ckpt["model"])
        release["backbone"] = backbone
        release["head"] = head
        canonical_source_field = "model"
        if source == "both" and has_ema:
            backbone_ema, head_ema = _split_backbone_head(ckpt["model_ema"])
            release["backbone_ema"] = backbone_ema
            release["head_ema"] = head_ema

    meta = _extract_meta(ckpt.get("args"))
    if extra_meta:
        meta.update(extra_meta)
    meta["source"] = str(input_path.name)
    meta["source_model_field"] = canonical_source_field
    if "epoch" in ckpt:
        meta["source_epoch"] = int(ckpt["epoch"])
    if "best_acc1" in ckpt:
        meta["source_best_acc1"] = float(ckpt["best_acc1"])
    release["meta"] = meta

    print(f"[save] {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, output_path)

    src_mb = input_path.stat().st_size / (1024 ** 2)
    dst_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"[done] {src_mb:.1f} MB -> {dst_mb:.1f} MB "
          f"(backbone: {len(backbone)} tensors, head: {len(head)} tensors, "
          f"source={canonical_source_field}, "
          f"ema_kept={'yes' if 'backbone_ema' in release else 'no'})")
    return release


def write_safetensors(release: Dict[str, Any], output_path: Path) -> None:
    """Optionally write a flat .safetensors mirror + meta json side-car."""
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is not installed; run `pip install safetensors` "
            "to enable --safetensors export."
        ) from exc

    flat: Dict[str, torch.Tensor] = {}
    for group in ("backbone", "backbone_ema", "head", "head_ema"):
        sd = release.get(group)
        if sd is None:
            continue
        for k, v in sd.items():
            flat[f"{group}.{k}"] = v.contiguous().to(torch.float32) \
                if v.dtype not in (torch.float32, torch.float16, torch.bfloat16,
                                   torch.int8, torch.int16, torch.int32,
                                   torch.int64, torch.uint8, torch.bool) else v.contiguous()

    st_path = output_path.with_suffix(".safetensors")
    print(f"[save] {st_path}")
    # Safetensors metadata must be a flat string->string map.
    str_meta = {k: json.dumps(v) for k, v in release["meta"].items()}
    save_file(flat, str(st_path), metadata=str_meta)

    meta_path = output_path.with_suffix(".meta.json")
    print(f"[save] {meta_path}")
    meta_path.write_text(json.dumps(release["meta"], indent=2), encoding="utf-8")


def _list_meta(items: Iterable[str]) -> str:
    return ", ".join(items) or "(none)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the supervised training checkpoint (.pth).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the slim release checkpoint (.pth).")
    parser.add_argument("--source", choices=("model", "ema", "both"), default="both",
                        help="Which state_dict to keep. 'model' = non-EMA only, "
                             "'ema' = EMA only (promoted to backbone/head), "
                             "'both' = keep model under backbone/head and EMA under "
                             "backbone_ema/head_ema. (default: both)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Backward-compat alias for --source model.")
    parser.add_argument("--safetensors", action="store_true",
                        help="Also write a .safetensors mirror and a .meta.json side-car.")
    args = parser.parse_args()

    source = "model" if args.no_ema else args.source

    release = convert(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        source=source,
    )
    print(f"[meta] {_list_meta(release['meta'].keys())}")

    if args.safetensors:
        write_safetensors(release, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
