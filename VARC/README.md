# VARC KoPE Integration

This folder documents the KoPE integration for [VARC](https://github.com/lillian039/VARC). It is intended to be used with an upstream VARC checkout rather than as a standalone copy of the VARC repository.

## Files

- [`src/ARC_ViT_KoPE_EncDec.py`](src/ARC_ViT_KoPE_EncDec.py): ViT+KoPE encoder-decoder model definitions.
- [`src/ARC_ViT_KoPE_EncDec_frompretrain.py`](src/ARC_ViT_KoPE_EncDec_frompretrain.py): variant used for initialization from a pretrained checkpoint.
- [`src/ARC_ViT_EncDec.py`](src/ARC_ViT_EncDec.py): ViT encoder-decoder baseline used alongside the KoPE variants.
- [`utils/args.py`](utils/args.py): replacement helper file adding CLI registration for `vit_encdec`, `vit_kope_encdec`, `vit_kope_encdec2`, and KoPE-specific arguments.
- [`utils/load_model.py`](utils/load_model.py): replacement helper file adding model-loading registration for `ARCViTEncDec`, `ARCViTKoPEEncDec`, and `ARCViTKoPE2EncDec`.

## Usage

Set up data, dependencies, offline training, and test-time training following the upstream VARC workflow. Copy the files above into the corresponding `src/` and `utils/` directories of an upstream VARC checkout, replacing the upstream helper files where paths overlap, then run the upstream VARC launch flow with `--architecture "vit_kope_encdec"`.

## Data

Follow the upstream VARC data preparation instructions for ARC-AGI / ARC-AGI-2 and any Re-ARC augmentation used in your experiments.

## License Note

The upstream VARC repository does not publish an explicit license file as of this release preparation. Do not redistribute a full VARC checkout from this repository without separate permission or license clearance.

## Acknowledgments

This integration builds on [VARC](https://github.com/lillian039/VARC).