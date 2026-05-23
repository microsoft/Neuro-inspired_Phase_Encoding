# Third Party Notices

This repository includes Microsoft-authored code as well as code adapted from or designed to integrate with the third-party projects listed below. The component inventory is also recorded in [`cgmanifest.json`](cgmanifest.json).

## SimDINO / DINOv2-derived code

- Path: [`simdinov2/`](simdinov2/)
- Source: [RobinWu218/SimDINO](https://github.com/RobinWu218/SimDINO)
- Related upstream: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- Notice: Files inherited from DINOv2 / SimDINO retain their upstream copyright and license notices where present. The DINOv2-derived tree includes [`simdinov2/LICENSE`](simdinov2/LICENSE).

## DeiT-derived utilities

- Path: [`simdinov2/supervised/deit/`](simdinov2/supervised/deit/)
- Source: [facebookresearch/deit](https://github.com/facebookresearch/deit)
- Notice: Files retain upstream notices where present.

## OpenMMLab configuration templates

- Path: [`simdinov2/eval/mysegmentation/configs_mmdet/`](simdinov2/eval/mysegmentation/configs_mmdet/) and [`simdinov2/eval/mysegmentation/configs_mmseg/`](simdinov2/eval/mysegmentation/configs_mmseg/)
- Sources: [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) and [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- License: Apache-2.0. This repository keeps only the small configuration subset required by the custom KoPE segmentation configs under [`simdinov2/eval/mysegmentation/configs/`](simdinov2/eval/mysegmentation/configs/).

## OpenCLIP integration

- Path: [`open_clip/`](open_clip/)
- Source: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- Notice: This repository provides KoPE integration files for use with an upstream OpenCLIP checkout. Files adapted from OpenCLIP / OpenAI CLIP retain upstream attribution where present.

## VARC integration

- Path: [`VARC/`](VARC/)
- Source: [lillian039/VARC](https://github.com/lillian039/VARC)
- Notice: The upstream VARC repository does not publish an explicit license file as of this release preparation. This repository is intended to publish only the Microsoft-authored ViT encoder-decoder / KoPE model files and replacement helper files documented in [`VARC/README.md`](VARC/README.md), which can be copied into an upstream VARC checkout by users who obtain VARC under appropriate terms.
