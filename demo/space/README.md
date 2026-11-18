---
title: Sam's LM
emoji: 🔐
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: A GPT built from scratch, attacked for its training data
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Sam's LM: in-browser demo

A 27.5M-parameter GPT built from scratch and trained on TinyStories, with fake secrets hidden in its
training data. This page runs the models in your browser (ONNX Runtime Web) and lets you:

- **Steal a secret:** beam-search attack on planted secrets, against the undefended model, a deduplicated
  model and a DP-SGD (ε = 8) model.
- **Was it trained on this?:** reference-model membership inference on any story.
- **Write a story** with each model.

Code, experiments and write-up: https://github.com/samuelbingham07/sams-lm
Model files: https://huggingface.co/samuelbingham07/sams-lm
