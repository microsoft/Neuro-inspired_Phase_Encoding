# OpenCLIP KoPE Integration

This folder contains the KoPE vision-tower integration for [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip). It is intended to be copied into an upstream OpenCLIP checkout rather than used as a standalone package.

## Files

- [`transformer_kope.py`](transformer_kope.py): KoPE-enabled Vision Transformer implementation for OpenCLIP.
- [`model.py`](model.py): patched OpenCLIP model builder with `CLIPVisionCfg` KoPE fields and `vision_cfg.kope_mode` support.
- [`ViT-KoPE-B-16.json`](ViT-KoPE-B-16.json): model config for `ViT-KoPE-B-16`.

## Installation

Set up OpenCLIP following the upstream instructions:

```bash
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
pip install -e ".[training]"
```

Copy the KoPE files into the OpenCLIP source tree:

```bash
cp PATH_TO_THIS_REPO/open_clip/transformer_kope.py src/open_clip/transformer_kope.py
cp PATH_TO_THIS_REPO/open_clip/model.py src/open_clip/model.py
cp PATH_TO_THIS_REPO/open_clip/ViT-KoPE-B-16.json src/open_clip/model_configs/ViT-KoPE-B-16.json
```

## Data

Use the standard OpenCLIP data pipeline. WebDataset shards are recommended for large-scale training; CSV datasets are also supported. See the upstream [OpenCLIP data documentation](https://github.com/mlfoundations/open_clip#data).

## Training

After copying the files above, use the standard upstream OpenCLIP training workflow and select the `ViT-KoPE-B-16` model config. See the upstream [OpenCLIP training documentation](https://github.com/mlfoundations/open_clip#training-clip).

## Evaluation

Use the standard upstream OpenCLIP evaluation / zero-shot workflow with the `ViT-KoPE-B-16` model config. See the upstream [OpenCLIP evaluation documentation](https://github.com/mlfoundations/open_clip#evaluation--zero-shot).

## Acknowledgments

This integration builds on [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).