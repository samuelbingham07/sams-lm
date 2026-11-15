# Final Project — Handoff / Catch-Up Doc

Self-contained context for continuing this project on the Windows PC (training machine).
If you're an AI assistant reading this: this file is the source of truth for what we decided and why. Start here.

Author: Samuel Bingham. Course: Ethical Software Systems. Partner: TBD (project is 2-person; strongly recommended by instructor).

---

## 1. The assignment (what has to be delivered)

Final project for an "Ethical Software Systems" course. Requirements:
- Must relate to **software/computer systems with societal/ethical impact**.
- Must have a **significant programming component** (build/implement something, not a philosophy paper).
- Scope ≈ **twice a normal weekly assignment**. Grader considers scope.
- Can be done with a partner (recommended).

**Deadlines:**
- Proposal: **Wed Nov 18, 2025** — short Gradescope form: (1) topic, (2) group members, (3) one-paragraph description.
- Full project: **Wed Dec 9, 2025, 11:59pm** — submit (1) all code, (2) `report.pdf`, (3) `reflection.txt`.
- Presentation: in class **Dec 7–9**; submit slide deck on Gradescope by Dec 9.

**`reflection.txt` must answer:** how long it took; how AI was used; whether AI helped / saved time / affected learning; any other feedback.

---

## 2. What we're building (the decision)

**Build a small GPT-style language model from scratch, then study how it memorizes and leaks its training data — and whether a privacy defense reduces that leakage.**

Two intertwined goals:
- **Portfolio goal:** a genuinely-works, built-from-scratch LLM. Trained on **TinyStories** (dataset of simple children's stories, small vocabulary) so a small model produces *coherent English short stories* — the impressive, demo-able result. Clean repo + README + training curves + sample generations + (stretch) a demo UI.
- **Course / ethics goal:** language models memorize specific training examples and can regurgitate private data ("training data extraction"). We plant unique "canary" secrets in the training data, train the model, then try to extract them back out, measuring memorization. Then we train a version **with a privacy defense** (DP-SGD and/or data deduplication) and show leakage drops — at some cost to model quality.

**Ethical argument:** LLMs don't just learn patterns, they memorize. If training data contains private info, the model can leak it. This is a concrete, current harm and the reason differential privacy in ML matters. We demonstrate the harm on our own model and measure a mitigation.

### Build philosophy: MAX FROM-SCRATCH
Samuel writes the tokenizer (byte-pair encoding), multi-head self-attention, transformer blocks, and training loop **himself**. Allowed: keep a known-good reference (nanoGPT) open purely as a sanity-check oracle when a result looks wrong — but write every line ourselves. Use PyTorch as the tensor/autograd engine (that's expected, not "cheating").

### What we measure / the key graphs
- Sample generations at several training checkpoints (quality over time).
- Training/validation loss curves.
- **Memorization vs. repetition:** extraction success for canaries that appear 1, 5, 10, 50… times in training data (classic rising curve).
- **Leakage with no defense vs. with defense** (DP-SGD and/or dedup).
- **Utility cost of the defense** (quality/loss with vs. without) — the privacy/utility tradeoff, now on our own LLM.

### Partner split (once partner confirmed)
- **Partner A:** build + train the GPT (tokenizer, architecture, training loop, generation).
- **Partner B:** canary design, extraction attack, the defense (DP/dedup), analysis + plots + report.

### Open decisions (not yet locked)
- [ ] Confirm partner + their name for the proposal.
- [ ] GPU confirmed? (fills in model size — see HARDWARE.md). Need `nvidia-smi` output.
- [ ] Final model size (depends on VRAM; target 10–60M params).
- [ ] Defense: DP-SGD (Opacus) vs. dedup vs. both. Start with dedup (simpler), add DP-SGD if time.
- [ ] Stretch: Gradio/web demo for portfolio.

---

## 3. PC environment setup (Windows)

Run in order. Assumes Windows + NVIDIA GPU.

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

### 3.4 Install PyTorch with CUDA
Get the exact command from https://pytorch.org/get-started/locally/ (pick Stable / Windows / Pip / Python / CUDA). As of this writing it looks like:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Then verify the GPU is visible (see the python one-liner in HARDWARE.md). `CUDA available: True` is the goal.

### 3.5 Other dependencies
```powershell
pip install numpy matplotlib tqdm datasets tiktoken
```
- `datasets` → download TinyStories (`roneneldan/TinyStories` on Hugging Face).
- `tiktoken` → reference tokenizer for comparison (we still build our own BPE).
- Later, if doing DP-SGD: `pip install opacus`.
- Later, if doing a demo: `pip install gradio`.

Freeze versions when stable: `pip freeze > requirements.txt`

---

## 4. Planned repo structure
```
Final Project/
  PROJECT_HANDOFF.md        <- this file
  HARDWARE.md               <- machine specs
  requirements.txt
  data/                     <- TinyStories download + prepared token files (gitignore large files)
  src/
    tokenizer.py            <- BPE tokenizer, from scratch
    model.py                <- transformer: attention, blocks, GPT (from scratch)
    train.py                <- training loop, checkpointing, loss logging
    generate.py             <- sampling / text generation
    canaries.py             <- insert canary secrets into training data
    extract.py              <- membership/extraction attack, memorization measurement
    defense.py              <- dedup and/or DP-SGD training variant
  experiments/
    plots.py                <- generate the report figures
    results/                <- loss logs, generations, metrics
  report/
    report.pdf
    reflection.txt
  slides/
```

---

## 5. Immediate next steps (in order)
1. Move this folder to the PC (or git clone if we set up a repo).
2. Fill in `HARDWARE.md` (run `nvidia-smi` etc.). **Report the VRAM back so we can pick model size.**
3. Do env setup (section 3). Confirm `torch.cuda.is_available()` is True.
4. Download TinyStories, inspect it, decide vocab size for BPE.
5. Start building `tokenizer.py` (BPE from scratch) — first real code.
6. In parallel: confirm partner, submit the proposal (paragraph draft below).

---

## 6. Draft proposal paragraph (for Gradescope, Nov 18)

> **Topic:** Memorization and privacy leakage in language models we build ourselves.
>
> Large language models do not only learn general patterns; they memorize specific examples from their training data and can be made to regurgitate them, which means a model trained on text containing private information can leak that information. For our project we will implement a small GPT-style language model from scratch (our own byte-pair tokenizer, self-attention, transformer blocks, and training loop in PyTorch) and train it on the TinyStories dataset so that it produces coherent text. We will then plant unique "canary" secrets in the training data at varying repetition counts, train the model, and implement an extraction attack that measures how reliably those secrets can be recovered — quantifying how memorization grows with repetition. Finally, we will train versions of the model with privacy defenses (training-data deduplication and differentially private training via DP-SGD) and measure how much they reduce leakage and at what cost to model quality. Our report will present generation samples, loss curves, and memorization/leakage measurements, and discuss what they imply for the privacy of people whose data is used to train real language models and for practitioners deciding how to train responsibly.

(Adjust to "I" if solo. Add partner's name.)

---

## 7. Notes / reminders
- Big data and model checkpoints should NOT be committed to git — add a `.gitignore`.
- Keep every result (loss logs, sample generations, metrics) — they're the report's figures.
- Track hours spent and how AI was used, as we go, for `reflection.txt`.
- Attribution for any git commits made via Claude Code: end commit messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
