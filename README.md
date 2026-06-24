# Neuro-inspired Phase Encoding

This repo is the official implementation of ["Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency"](https://arxiv.org/abs/2604.07904) (ICML 2026). It includes code and configurations for the following tasks:

> **Supervised Image Classification** on ImageNet-1K. See [`simdinov2/supervised`](simdinov2/supervised/).

> **Self-Supervised Pre-training** (SimDINOv2 style) on ImageNet-1K. See [`simdinov2`](simdinov2/).

> **Semantic and Panoptic Segmentation**. See [`simdinov2/eval/mysegmentation`](simdinov2/eval/mysegmentation/).

> **Vision–Language Pre-training**. See [`open_clip`](open_clip/).

> **Few-shot Abstract Visual Reasoning (ARC-AGI)**. See [`VARC`](VARC/).

## Introduction

Spatiotemporal neural dynamics and oscillatory synchronization are widely implicated in biological information processing and have been hypothesized to support flexible coordination such as feature binding. By contrast, most deep learning architectures represent and propagate information through activation values alone, neglecting the joint dynamics of rate and phase.

**Kuramoto Oscillatory Phase Encoding (KoPE)** introduces an additional, evolving phase state to Vision Transformers and equips them with a neuro-inspired Kuramoto-style synchronization mechanism. Through synchronization-enhanced structure learning, KoPE improves training, parameter, and data efficiency, and benefits tasks that require structured understanding — including semantic and panoptic segmentation, representation alignment with language, and few-shot abstract visual reasoning (ARC-AGI). Theoretical analysis and empirical verification further show that KoPE accelerates attention concentration, indicating that synchronization can serve as a scalable, neuro-inspired mechanism for advancing state-of-the-art neural network models.

The core KoPE implementation lives in [`simdinov2/models/vision_transformer_kope.py`](simdinov2/models/vision_transformer_kope.py), [`open_clip/transformer_kope.py`](open_clip/transformer_kope.py), and [`VARC/src/ARC_ViT_KoPE_EncDec.py`](VARC/src/ARC_ViT_KoPE_EncDec.py).

### Single-file reference implementation

A minimal, single-file PyTorch reference implementation is also provided at the repository root in [`vit_kope.py`](vit_kope.py).

## Installation

The training and evaluation code targets PyTorch on CUDA and has been tested with Python 3.10, PyTorch 2.7.1, CUDA 12.6, and xFormers 0.0.31. A typical setup is:

```bash
conda create -n kope python=3.10 -y
conda activate kope
pip install -r requirements.txt
```

For ImageNet-V2 evaluation and segmentation / dense-prediction dependencies:

```bash
pip install -r requirements-extras.txt
```

Segmentation additionally requires the OpenMMLab stack:

```bash
mim install --no-deps mmengine
mim install --no-deps "mmcv==2.1.0"
mim install "mmdet>=3.0.0"
mim install "mmsegmentation==1.2.2"
```

For vision-language pre-training, refer to [`open_clip/README.md`](open_clip/README.md) and [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip). For ARC-AGI experiments, refer to [`VARC/README.md`](VARC/README.md) and [`VARC/requirements.txt`](VARC/requirements.txt).

## Data preparation

Data preparation is task-specific:

- **Supervised classification** ([`simdinov2/supervised/`](simdinov2/supervised/)) and **self-supervised pre-training** ([`simdinov2/`](simdinov2/)) use **ImageNet-1K**. Follow the [DINOv2 ImageNet-1K data preparation](https://github.com/facebookresearch/dinov2#imagenet-1k); this repository keeps the same DINOv2 / SimDINOv2 `root` + `extra` convention in [`simdinov2/data/datasets/image_net.py`](simdinov2/data/datasets/image_net.py). In the commands below, `PATH_TO_IMAGENET` is the ImageNet root and `PATH_TO_IMAGENET_EXTRAS` is the DINOv2-style extras directory.
- **Semantic and panoptic segmentation** ([`simdinov2/eval/mysegmentation/`](simdinov2/eval/mysegmentation/)): refer to the [MMSegmentation dataset preparation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) (ADE20K) and the [MMDetection dataset preparation guide](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/dataset_prepare.md) (COCO panoptic).
- **Vision–language pre-training** ([`open_clip/`](open_clip/)): refer to [`open_clip/README.md`](open_clip/README.md) and the upstream [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) data instructions.
- **Few-shot abstract visual reasoning** ([`VARC/`](VARC/)): refer to [`VARC/README.md`](VARC/README.md).

## Pre-trained Models

This release ships six ImageNet-1K checkpoints — three self-supervised (SimDINOv2-style with KoPE) and three supervised (DeiT-III-style with KoPE) — at the small, base, and large scales.

### Self-supervised pre-training (SimDINOv2)


| Model          | Patch | Resolution | Pre-train Epochs | #Params | Download  |
| :------------- | :---: | :--------: | :--------------: | :-----: | :-------- |
| ViT+KoPE-Small |  16  |    224    |       100       |  22 M  | [link](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/releases/download/v1.0.0/simdinov2_vitkope_small_patch16_in1k_100e.zip) |
| ViT+KoPE-Base  |  16  |    224    |       100       |  87 M  | [link](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/releases/download/v1.0.0/simdinov2_vitkope_base_patch16_in1k_100e.zip) |
| ViT+KoPE-Large |  16  |    224    |       100       |  306 M  | [link](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/releases/download/v1.0.0/simdinov2_vitkope_large_patch16_in1k_100e.zip) |

Each SSL release packages two files:

- `config.yaml` — the training/eval configuration (`student.*`, `teacher.*`, optimizer schedule, etc.) needed by the SimDINOv2 evaluation scripts to rebuild the model. The shipped `train.dataset_path` is a placeholder; update it only when using this config for training or resuming.
- `teacher_checkpoint.pth` — `torch.save({"teacher": state_dict}, ...)`, holding the EMA teacher weights.

### Supervised on ImageNet-1K


| Model          | Patch | Resolution | Train Epochs | #Params | Download  |
| :------------- | :---: | :--------: | :----------: | :-----: | :-------- |
| ViT+KoPE-Small |  16  |    224    |     300     |  22 M  | [link](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/releases/download/v1.0.0/supervised_vitkope_small_patch16_in1k_300e.zip) |
| ViT+KoPE-Base  |  16  |    224    |     300     |  87 M  | [link](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/releases/download/v1.0.0/supervised_vitkope_base_patch16_in1k_300e.zip) |
| ViT+KoPE-Large |  16  |    224    |     300     |  306 M  | [link](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/releases/download/v1.0.0/supervised_vitkope_large_patch16_in1k_300e.zip) |

Each supervised release is a single `.pth` file with the structure `{"backbone": state_dict, "head": {"weight", "bias"}, "meta": {...}}`.

## Evaluating the checkpoints

### Supervised classification

`simdinov2/supervised/eval/eval.py` auto-detects the checkpoint format (the `{backbone, head, meta}` dict layout) and reconstructs the classifier from `meta`. Point `--checkpoint` either at the released `.pth` file directly or at a folder that contains one:

```shell
python simdinov2/supervised/eval/eval.py \
  --checkpoint PATH_TO_CHECKPOINT.pth \
  --imagenet-val PATH_TO_IMAGENET/val \
  --imagenet-v2 PATH_TO_IMAGENET_V2 \
  --output-json ./eval_results.json
```

### Self-supervised learning (k-NN / linear probe / fine-tuning)

The released `config.yaml` + `teacher_checkpoint.pth` plug straight into the SimDINOv2 evaluation scripts via the standard `--config-file` and `--pretrained-weights` arguments; `simdinov2.utils.utils.load_pretrained_weights` automatically picks up the `teacher` key and strips the saved `backbone.` wrapper.

For SSL evaluation, pass the ImageNet paths through `--train-dataset` and `--val-dataset` as shown below. These command-line dataset arguments override the placeholder `train.dataset_path` in the released `config.yaml`, so the config file does not need to be edited for k-NN, linear probing, or fine-tuning evaluation.

k-NN classification example:

```shell
torchrun --nproc_per_node=4 simdinov2/eval/knn.py \
  --config-file PATH_TO_CHECKPOINT_FOLDER/config.yaml \
  --pretrained-weights PATH_TO_CHECKPOINT_FOLDER/teacher_checkpoint.pth \
  --output-dir ./eval_knn_base \
  --train-dataset ImageNet:split=TRAIN:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS \
  --val-dataset   ImageNet:split=VAL:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS
```

Linear probing example:

```shell
torchrun --nproc_per_node=4 simdinov2/eval/linear.py \
  --config-file PATH_TO_CHECKPOINT_FOLDER/config.yaml \
  --pretrained-weights PATH_TO_CHECKPOINT_FOLDER/teacher_checkpoint.pth \
  --output-dir ./eval_linear_base \
  --epochs 100 \
  --batch-size 256 \
  --train-dataset ImageNet:split=TRAIN:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS \
  --val-dataset   ImageNet:split=VAL:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS
```

Fine-tuning example:

```shell
torchrun --nproc_per_node=4 simdinov2/eval/finetuning.py \
  --config-file PATH_TO_CHECKPOINT_FOLDER/config.yaml \
  --arch-name vit \
  --net-type base \
  --batch-size 256 \
  --phase-coupling-lr-mult 0.5 \
  --pretrained-weights PATH_TO_CHECKPOINT_FOLDER/teacher_checkpoint.pth \
  --output-dir ./eval_finetune_base \
  --train-dataset ImageNet:split=TRAIN:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS \
  --val-dataset   ImageNet:split=VAL:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS
```

Evaluate the best fine-tuned checkpoint on ImageNet val / ImageNet-V2:

```shell
python simdinov2/supervised/eval/eval.py \
  --checkpoint ./eval_finetune_base/finetune_best.pth \
  --imagenet-val PATH_TO_IMAGENET/val \
  --imagenet-v2 PATH_TO_IMAGENET_V2 \
  --output-json ./eval_finetune_base/eval_finetune_best.json
```

For semantic / panoptic segmentation, pass `teacher_checkpoint.pth` to the segmentation training script's checkpoint argument (see [Semantic and panoptic segmentation](#semantic-and-panoptic-segmentation) below).

## Getting Started

### Supervised classification on ImageNet-1K

```shell
torchrun --standalone --nproc_per_node=4 simdinov2/supervised/train.py \
  --arch-name vit_kope --model-variant base \
  --batch-size 512 --epochs 300 --lr 3e-3 --weight-decay 0.05 --sched cosine --drop-path 0.2 \
  --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --opt fusedlamb --warmup-lr 1e-6 \
  --mixup .8 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss --color-jitter 0.3 --ThreeAugment \
  --kope-gamma 0.05 --kope-learn-phase-gamma False --share-kope-coupling --learnable-pos-embed \
  --kope-update-ext-phase --kope-coupling-qk-multilayer False --kope-vo-rotation True \
  --kope-mix True --kope-mix-init-gain 0.1 --kope-mix-phase-norm True --base 20 \
  --torch-compile --compile-mode default --persistent-workers \
  --output-dir ./results/supervised_vit_kope_base \
  --data-path PATH_TO_IMAGENET
# lr=4e-3 for ViT-S; drop-path=0.05/0.2/0.45 for ViT-S/B/L; refer to DeiT-III repo

python simdinov2/supervised/eval/eval.py \
  --checkpoint ./results/supervised_vit_kope_base \
  --use-ema \
  --imagenet-val PATH_TO_IMAGENET_VAL \
  --imagenet-v2 PATH_TO_IMAGENET_V2 \
  --output-json ./results/supervised_vit_kope_base/eval_ema_results.json
```

### Self-supervised learning (SimDINOv2) on ImageNet-1K

```shell
torchrun --standalone --nproc_per_node=4 simdinov2/train/train.py \
  --config-file simdinov2/configs/simdino_config.yaml \
  --output-dir ./results/simdinov2_vit_kope_base \
    train.dataset_path=ImageNet:split=TRAIN:root=PATH_TO_IMAGENET:extra=PATH_TO_IMAGENET_EXTRAS \
    train.batch_size_per_gpu=256 \
    train.num_workers=8 \
    student.arch=vit_kope_base \
    student.patch_size=16 \
    student.block=nested \
    student.block_type=nested \
    crops.local_crops_size=96 \
    student.drop_path_rate=0.1 \ # 0.1 for ViT-S/B, 0.2 for ViT-L in our paper
    student.kope_gamma=0.05 \
    student.learn_phase_gamma=learn \
    student.coupling_qknorm=true \
    student.coupling_qknorm_learn=true \
    student.share_kope_coupling=true \
    student.update_ext_token_phase=true \
    student.use_learnable_pos_embed=true \
    student.kope_vo_rotation=true \
    student.kope_mix=true \
    student.kope_mix_init_gain=0.1 \
    student.kope_mix_phase_norm=true \
    student.base=20 \
    student.no_phase_norm=true \
    optim.layerwise_decay=0.9 \
    optim.phase_coupling_lr_mult=0.5 \
    optim.base_lr=4e-3 \
    ibot.phase_loss_weight=0.0   # 1.0 for ViT-L
```

### Semantic and panoptic segmentation

Relies on mmsegmentation and mmdetection.

```shell
torchrun --standalone --nproc_per_node=4 simdinov2/eval/mysegmentation/tools/train.py \
  simdinov2/eval/mysegmentation/configs/coco/mask2former_vit_kope_simdino_base_50e_coco.py \
  --checkpoint PATH_TO_CHECKPOINT \
  --work-dir ./results/coco_panoptic_vit_kope_base \
  --data-root PATH_TO_COCO \
  --launcher pytorch \
  --seed 42 \
  --resume
```

### Vision–language pre-training

See [`open_clip/README.md`](open_clip/README.md). Drop [`open_clip/transformer_kope.py`](open_clip/transformer_kope.py) and [`open_clip/ViT-KoPE-B-16.json`](open_clip/ViT-KoPE-B-16.json) into a checkout of [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip), register the new model in `src/open_clip/model.py`, then install from source and train as usual.

### Few-shot abstract visual reasoning (ARC-AGI)

See [`VARC/README.md`](VARC/README.md) and the launch scripts under [`VARC/script/`](VARC/script/), e.g. [`offline_train_VARC_ViTKoPE.sh`](VARC/script/offline_train_VARC_ViTKoPE.sh).

## Citation

If you find this work useful, please cite our ICML 2026 paper:

```bibtex
@inproceedings{xiao2026kope,
  title     = {Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency},
  author    = {Xiao, Mingqing and Wang, Yansen and Han, Dongqi and Shan, Caihua and Li, Dongsheng},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

Code authored within this repository is released under the [MIT License](LICENSE). The [`simdinov2/`](simdinov2/) tree additionally retains its upstream [Apache-2.0 LICENSE](simdinov2/LICENSE).

## Acknowledgments

This codebase builds upon the following open-source projects, and we thank the authors for their excellent work:

- [SimDINO](https://github.com/RobinWu218/SimDINO)
- [DeiT](https://github.com/facebookresearch/deit)
- [open_clip](https://github.com/mlfoundations/open_clip)
- [VARC](https://github.com/lillian039/VARC)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
