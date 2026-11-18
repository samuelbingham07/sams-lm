---
license: mit
language: en
datasets:
  - roneneldan/TinyStories
tags:
  - onnx
  - privacy
  - memorization
  - differential-privacy
  - membership-inference
pipeline_tag: text-generation
---

# Sam's LM: ONNX models for the in-browser demo

Four small GPT models (27.5M parameters each) from
[sams-lm](https://github.com/samuelbingham07/sams-lm), a project that builds a language model from scratch,
plants fake secrets ("canaries") in its training data, and measures what leaks and which defenses stop it.
These files power the [in-browser demo](https://huggingface.co/spaces/samuelbingham07/sams-lm-demo), an
extension built beyond the scope of the original course project.

| File | Training run | What it is | Validation loss |
|---|---|---|---|
| `no_defense.onnx` | `canary_M` | Standard training on TinyStories + 160 canaries (0 to 64 copies each) | 1.347 |
| `dedup.onnx` | `canary_dedup_M` | Same data after exact deduplication | 1.348 |
| `dp_eps8.onnx` | `dp_bigbatch_eps8_M` | DP-SGD, ε = 8, δ ≈ 1.1e-7 (per 512-token example), batch 4,096, 200 steps | 3.860 |
| `shadow.onnx` | `shadow_M` | Reference ("shadow") model: same recipe on a disjoint half of the data, no canaries | 1.413 |

**The canaries are fake.** The names and 6-digit numbers were randomly generated for this experiment;
no real personal data is involved.

## Architecture

GPT-2-style decoder: 8 layers, 8 heads, width 512, context 512, learned position embeddings,
pre-LayerNorm, GELU MLP, no biases, tied input/output embeddings. Byte-level BPE tokenizer with 4,096
tokens (`tokenizer.json`; `<|endoftext|>` = 4095). Every story is preceded by `<|endoftext|>`.

## ONNX interface

Exported by `demo/export_onnx.py` (opset 17), with a key/value cache:

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `input_ids` | int64 | [B, T] | new tokens |
| `position_ids` | int64 | [B, T] | their absolute positions |
| `shared_k`, `shared_v` | float32 | [8, 1, 8, Ps, 64] | cache shared by every row (broadcast) |
| `past_k`, `past_v` | float32 | [8, B, 8, Pb, 64] | per-row cache |
| `key_mask` | float32 | [B, Ps + Pb + T] | 1 = attend to this key, 0 = padding |
| → `logits` | float32 | [B, T, 4096] | next-token logits |
| → `new_k`, `new_v` | float32 | [8, B, 8, T, 64] | cache entries for the new tokens |

To run without a cache, pass one zero slot for each cache and mask it out. Weights are stored as float16
and cast to float32 inside the graph, so all arithmetic is float32. Outputs match the PyTorch models
(with the same float16-rounded weights) to within 3e-5.

## Intended use

Teaching and demonstrating training-data memorization, membership inference and their defenses. These
are toy models trained on children's stories, not for production text generation.

## Credits

Data: TinyStories (Eldan & Li, 2023), CDLA-Sharing-1.0. Code: MIT.
