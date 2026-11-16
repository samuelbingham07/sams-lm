# Final Project — Handoff / Catch-Up Doc

Self-contained context for continuing this project on the Windows PC (training machine).
If you're an AI assistant reading this: this file is the source of truth for what we decided and why. Start here.

Author: Samuel Bingham. Course: Ethical Software Systems (CS 181E, Fall 2026). Partner: asked, awaiting answer — Sam will proceed solo if not.

---

## 1. The assignment (what has to be delivered)

Final project for an "Ethical Software Systems" course. Requirements:
- Must relate to **software/computer systems with societal/ethical impact**.
- Must have a **significant programming component** (build/implement something, not a philosophy paper).
- Scope ≈ **twice a normal weekly assignment**. Grader considers scope.
- Can be done with a partner (recommended).

**Deadlines:**
- Proposal: **Wed Nov 18, 2026** — short Gradescope form: (1) topic, (2) group members, (3) one-paragraph description.
- Full project: **Wed Dec 9, 2026, 11:59pm** — submit (1) all code, (2) `report.pdf`, (3) `reflection.txt`.
- Presentation: in class **Dec 7–9**; submit slide deck on Gradescope by Dec 9.

(Year corrected from 2025: the lecture slides are dated Fall 2026.)

**`reflection.txt` must answer:** how long it took; how AI was used; whether AI helped / saved time / affected learning; any other feedback.

---

## 2. What we're building (the decision)

**Build a small GPT-style language model from scratch, then study how it memorizes and leaks its training data — and whether a privacy defense reduces that leakage.**

Two intertwined goals:
- **Portfolio goal:** a genuinely-works, built-from-scratch LLM. Trained on **TinyStories** (dataset of simple children's stories, small vocabulary) so a small model produces *coherent English short stories* — the impressive, demo-able result. Clean repo + README + training curves + sample generations + (stretch) a demo UI.
- **Course / ethics goal:** language models memorize specific training examples and can regurgitate private data ("training data extraction"). We plant unique "canary" secrets in the training data, train the model, then try to extract them back out, measuring memorization. Then we train a version **with a privacy defense** (DP-SGD and/or data deduplication) and show leakage drops — at some cost to model quality.

**Ethical argument:** LLMs don't just learn patterns, they memorize. If training data contains private info, the model can leak it. This is a concrete, current harm and the reason differential privacy in ML matters. We demonstrate the harm on our own model and measure a mitigation.

### Build philosophy: from scratch, thorough, matched to the class (updated 2026-11-15)
The tokenizer (byte-pair encoding), multi-head self-attention, transformer blocks, and training loop are implemented from scratch in this repo (no Hugging Face model or tokenizer libraries), with code written by Claude Code and directed and reviewed by Samuel. Priorities: **thoroughness** (tests, sanity checks, reproducibility) and **matching the class's own code style and scope** where possible. nanoGPT is used only as a reference to check results against. PyTorch is the tensor/autograd engine. AI use is documented in `reflection.txt`.

### What we measure / the key graphs
- Sample generations at several training checkpoints (quality over time).
- Training/validation loss curves.
- **Memorization vs. repetition:** extraction success for canaries that appear 1, 5, 10, 50… times in training data (classic rising curve).
- **Leakage with no defense vs. with defense** (DP-SGD and/or dedup).
- **Utility cost of the defense** (quality/loss with vs. without) — the privacy/utility tradeoff, now on our own LLM.

### Alignment with the CS 181E lectures
Lecture slides (PDFs in Sam's OneDrive `Documents\Class Files`, not committed) relevant to this project:
- **L12–13 ML1/ML2:** Naive Bayes, perceptron, SVM, neural nets; code is plain Python with `pandas` + `sklearn`; models are evaluated with `sklearn.metrics` (accuracy, precision, recall, F1, confusion matrix).
- **L16 ML-Attacks:** frames attacks with confidentiality/integrity/availability (C/I/A). *Inference-stage confidentiality attacks:* **membership inference** (shadow models → attack classifier, "83–92% accuracy"), **training data extraction**, model inversion, model extraction. Defenses: **differentially-private training**, outlier mitigation, gradient masking, explainability.
- **L17–18 ML-DP1/DP2:** where to add DP noise (data / training / model / predictions); **composition → privacy budget**; **DP-SGD = clip each example's gradient (bounds sensitivity) + add Gaussian noise to the aggregated gradient → approximate (ε, δ)-DP**; utility comparison exercise ("how does accuracy compare to the non-private model?"). Notes that DP-SGD lives in tensorflow-privacy, not diffprivlib.
- **L4–5 dp1/dp2:** ε-DP definition, sensitivity, Laplace mechanism, composition, randomized response, RAPPOR.
- L14 ML-Bias and L15 ML-Expl (LIME): not central here.

How the project maps onto them:
- Memorization attacks are **inference-stage confidentiality attacks** (L16 terms), evaluated **as binary classifiers with `sklearn.metrics`**, the class's own evaluation tooling, plus ROC-AUC and TPR at low FPR (standard for MIA).
- **Membership inference:** loss-threshold attack (members = training stories; non-members = validation stories with the train overlap removed). Stretch: the **shadow-model attack from L16** (small shadow LMs on disjoint data halves → sklearn classifier on loss features).
- **Training data extraction:** planted canaries; prompt with the prefix, check whether greedy decoding reproduces the secret; also canary "exposure" (rank of true secret among candidates).
- **DP-SGD:** implement per-example clipping + Gaussian noise ourselves, exactly as in L18 (PyTorch, since tensorflow-privacy doesn't fit); use an accountant only to compute ε at fixed δ (privacy budget over all steps = composition, L17).
- **Dedup:** data-side defense (related to L16 outlier mitigation).
- **Privacy/utility tradeoff:** validation loss and sample quality with vs. without each defense, mirroring the DP Naive Bayes exercise.
- Scope: the class works at intro-ML level. The from-scratch GPT is the portfolio piece; the report should put the weight on the privacy analysis, in the course's vocabulary.

### Partner split (once partner confirmed)
- **Partner A:** build + train the GPT (tokenizer, architecture, training loop, generation).
- **Partner B:** canary design, extraction attack, the defense (DP/dedup), analysis + plots + report.

### Open decisions (not yet locked)
- [ ] Partner: asked, awaiting answer (solo is fine).
- [x] GPU: AMD RX 7800 XT, 16 GB VRAM, ROCm PyTorch working (see HARDWARE.md).
- [ ] Final model size (16 GB VRAM → top of the 10–60M param range is feasible).
- [x] Tokenizer: byte-level BPE, vocab 4096. Context length: 512 planned.
- [x] Keep natural duplicates in baseline training data (so dedup has something to remove).
- [ ] Defense: DP-SGD (Opacus) vs. dedup vs. both. Start with dedup (simpler), add DP-SGD if time.
- [ ] Stretch: Gradio/web demo for portfolio.

---

## 3. PC environment setup (Windows)

Done on the training PC (2026-11-15). Kept as a record and for re-setup.

### 3.1 Document hardware first
See `HARDWARE.md` in this folder. Run the commands there (esp. `nvidia-smi`) and paste outputs. The VRAM number sets our model size.

### 3.2 Install Python
- Install **Python 3.11** (3.12 is fine too) from python.org or the Microsoft Store. Check "Add to PATH".
- Verify: `python --version`

### 3.3 Create a virtual environment (in the project folder)
```powershell
cd <project folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.)

### 3.4 Install PyTorch with ROCm (AMD GPU) — DONE 2026-11-15
The training PC has an **AMD RX 7800 XT (gfx1101), not NVIDIA**, so we use AMD's ROCm build of PyTorch instead of CUDA. Verified working on Windows with Python 3.14:
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "torch[device-gfx1101]==2.12.0+rocm7.14.1" `
  "torchvision[device-gfx1101]==0.27.0+rocm7.14.1" `
  "torchaudio==2.11.0+rocm7.14.1"
```
Or reproduce the whole env: `pip install -r requirements.txt --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/`

Source: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html

Notes:
- ROCm PyTorch still uses the `torch.cuda` API: `device="cuda"`, `torch.cuda.is_available()` → `True`. nanoGPT/tutorial code works unchanged.
- Measured: ~9 TFLOPS fp32, **~36 TFLOPS bf16** → train in bf16 autocast.
- Built-in `scaled_dot_product_attention` warns that flash/mem-efficient kernels are experimental on this GPU. Irrelevant for us since we write attention ourselves, but if we use it for speed: set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
- `torch.compile` on ROCm/Windows is untested — try it, fall back to eager if it breaks.

### 3.5 Other dependencies
```powershell
pip install numpy matplotlib tqdm datasets tiktoken ftfy pytest
```
- `ftfy` → repair mojibake during data cleaning. `pytest` → tests.
- `datasets` → download TinyStories (`roneneldan/TinyStories` on Hugging Face).
- `tiktoken` → reference tokenizer for comparison (we still build our own BPE).
- Later, if doing DP-SGD: `pip install opacus`.
- Later, if doing a demo: `pip install gradio`.

Freeze versions when stable: `pip freeze > requirements.txt`

---

## 4. Repo structure (✓ = exists)
```
Sam's LM/
  PROJECT_HANDOFF.md        ✓ this file
  HARDWARE.md               ✓ machine specs
  requirements.txt          ✓ pinned env (needs AMD extra index, see 3.4)
  tokenizer/bpe4096.json    ✓ trained tokenizer (small, committed)
  data/                     ✓ gitignored: hf_cache/, clean/, tokens/
  src/
    download_data.py        ✓ TinyStories -> data/ (raw text export)
    prepare_data.py         ✓ cleaning -> data/clean/{train,validation}.txt + stats.json
    tokenizer.py            ✓ byte-level BPE, from scratch (train / encode / decode / save / load)
    encode_data.py          ✓ -> data/tokens/{split}.bin (uint16) + {split}_offsets.npy
    model.py                ✓ GPT from scratch: explicit causal attention, pre-LN blocks, weight tying, generate()
    train.py                  training loop, checkpointing, loss logging
    generate.py               sampling / text generation
    canaries.py               insert canary secrets into training data
    extract.py                membership inference + extraction attacks
    defense.py                dedup and DP-SGD training variants
  tests/
    test_tokenizer.py       ✓ 26 tests incl. exact match vs. naive reference BPE
    test_model.py           ✓ 14 tests: causality, explicit attn == SDPA, overfits a batch, per-example grads (DP-SGD)
  experiments/
    benchmark.py            ✓ throughput of candidate model sizes
    plots.py                  report figures
    results/                  loss logs, generations, metrics (committed)
  report/  report.pdf, reflection.txt
  slides/
```
Rebuild data from scratch: `python src/download_data.py; python src/prepare_data.py; python src/tokenizer.py; python src/encode_data.py` (~5 min total).

---

## 5. Data + tokenizer (done 2026-11-15)
**Raw TinyStories:** 2,119,719 train / 21,990 validation stories.

Problems found and how they were handled (`src/prepare_data.py`, numbers in `data/clean/stats.json`):
- **Mojibake** in 129,567 train stories (6%): `â€™` instead of `'` etc. Some closing quotes lost a byte, which ftfy can't fix. Fixed with an explicit map, then ftfy, then ASCII folding. Only 279 chars dropped in the whole train set.
- **230 empty stories:** dropped.
- **Exact duplicates:** 320,241 train stories appear exactly twice (1,799,248 unique). **Kept on purpose:** a natural "seen once vs. twice" memorization comparison, and real work for the dedup defense.
- **Train/validation overlap:** 6,601 of 21,990 validation stories (30%) also appear in train. **Removed from validation** (15,389 remain), so val loss is honest and validation can serve as true non-members for membership inference.

**Tokenizer** (`src/tokenizer.py`): byte-level BPE, GPT-2-style regex pre-split, 256 bytes + 3,839 merges + `<|endoftext|>` = **4,096 tokens**. Trained on all train chunks in ~20 s. **3.99 bytes/token on validation vs. 4.11 for GPT-2's 50,257-token tokenizer.** Tests check exact equality with a naive reference trainer/encoder and round-tripping (including non-ASCII).

**Encoded** (`src/encode_data.py`): train **484.1M tokens**, validation **3.26M tokens**. Mean 228 tokens/story, p99 664; 3.5% of train stories exceed 512 tokens.

**Note for canary experiments:** the tokenizer is trained on clean data *before* canaries are inserted, so secrets don't get dedicated tokens. That's realistic (tokenizers are usually trained once and reused).

---

## 5b. Model + GPU benchmark (done 2026-11-15)
`src/model.py`: GPT-2-style decoder, no biases, dropout 0 by default (keeps per-example gradients well defined for DP-SGD). `forward(..., reduction="none")` gives per-token losses for the memorization experiments.

Training throughput on the RX 7800 XT, bf16 autocast, batch 32 x 512 tokens (`experiments/benchmark.py`, results in `experiments/results/benchmark.json`):

| Model | Params | Explicit attn | Torch SDPA | Peak mem | ~Time per epoch (484M tok) |
|---|---|---|---|---|---|
| S 6L/384d | 12.4M | 132K tok/s | 116K | 4.4 GB | 1.0 h |
| M 8L/512d | 27.5M | 70K tok/s | 62K | 7.1 GB | 1.9 h |
| L 10L/640d | 52.1M | 46K tok/s | 41K | 10.6 GB | 2.9 h |
| XL 12L/768d | 88.5M | 12K tok/s | 23K | 15.0 GB | 5.8 h (SDPA) |

- Our explicit attention is **faster** than torch's SDPA here (except XL, which is memory-bound). Default `use_sdpa=False`.
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` makes SDPA **crash** (hipErrorInvalidValue). Don't set it.
- `torch.compile` **doesn't work** (needs Triton, unavailable for ROCm on Windows). Run eager.
- DP-SGD will be several times slower per token (per-example gradients), so DP runs may need the S model and/or fewer tokens.

---

## 6. Next steps (in order)
1. **Submit the proposal by Wed Nov 18** (paragraph in section 7; switch to "I" if solo).
2. ~~`model.py`~~ done. ~~Benchmark~~ done.
3. `train.py`: memmap data loader, AdamW + cosine LR schedule, bf16 autocast, grad clipping, periodic val loss + sample generations, checkpoints, logs to `experiments/results/`.
4. Pick model size (recommendation: M, 27.5M params, ~2 h/epoch) and train the baseline.
5. `canaries.py` / `extract.py` → `defense.py` (dedup, then DP-SGD) → plots → report.

---

## 7. Draft proposal paragraph (for Gradescope, Nov 18)

> **Topic:** Memorization and privacy leakage in language models we build ourselves.
>
> Large language models do not only learn general patterns; they memorize specific examples from their training data and can be made to regurgitate them, which means a model trained on text containing private information can leak that information. For our project we will implement a small GPT-style language model from scratch (our own byte-pair tokenizer, self-attention, transformer blocks, and training loop in PyTorch) and train it on the TinyStories dataset so that it produces coherent text. We will then plant unique "canary" secrets in the training data at varying repetition counts, train the model, and implement an extraction attack that measures how reliably those secrets can be recovered — quantifying how memorization grows with repetition. Finally, we will train versions of the model with privacy defenses (training-data deduplication and differentially private training via DP-SGD) and measure how much they reduce leakage and at what cost to model quality. Our report will present generation samples, loss curves, and memorization/leakage measurements, and discuss what they imply for the privacy of people whose data is used to train real language models and for practitioners deciding how to train responsibly.

(Adjust to "I" if solo. Add partner's name.)

---

## 8. Notes / reminders
- Big data and model checkpoints are NOT committed (`.gitignore` covers `data/`, `*.pt`, `checkpoints/`). Repo: https://github.com/samuelbingham07/sams-lm (public).
- Keep every result (loss logs, sample generations, metrics) — they're the report's figures.
- Track hours spent and how AI was used, as we go, for `reflection.txt`.
- Attribution for any git commits made via Claude Code: end commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
