# Text/UVD LIBERO ablations

This directory contains a standalone wrapper for the four conditioning
ablations. It does not alter the existing `QwenGR00T`, `QwenGR00TCoT`, or
geometry data paths.

## Prepare decoded embodied-CoT sidecars

```bash
python examples/modelExtensions/TextUVD/scripts/prepare_libero_cot_sidecar.py \
  --cot-root /root/data/yxz/datasets/libero_cot \
  --rerender-root /root/data/yxz/datasets/libero_rerender \
  --tokenizer /root/data/yxz/models/deepthinkvla_tokenizer
```

The source tokenizer is used only for this one-time decode. Training stores
UTF-8 text and lets the Qwen3.5 processor tokenize the prompt.

## Train one ablation

Use the corrected standalone runner:

```bash
accelerate launch starVLA/training/train_starvla_text_uvd_runner.py \
  --config_yaml examples/modelExtensions/TextUVD/configs/qwen35_gr00t_libero_base.yaml
```

Replace `base` with `text`, `uvd`, or `both` for the other three YAMLs.

At an episode boundary, inherited delta-action chunks are zero-padded. The
new UVD adapter always returns fixed `K` points; repeated tail points are
marked invalid in `uvd_valid_mask` and are masked before the action expert.
