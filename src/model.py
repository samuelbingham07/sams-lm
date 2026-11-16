"""GPT-style decoder-only transformer, implemented from scratch in PyTorch.

Architecture (same family as GPT-2, so nanoGPT can serve as a reference):

    tokens -> token embedding + learned position embedding
           -> N x Block:  x = x + Attention(LayerNorm(x))
                          x = x + MLP(LayerNorm(x))
           -> final LayerNorm -> linear head (weights tied to token embedding)
           -> logits over the vocabulary for the NEXT token at every position

Design notes
------------
* Attention is written out explicitly (QK^T / sqrt(d) -> causal mask -> softmax
  -> weighted sum of V). `use_sdpa=True` swaps in PyTorch's fused kernel for
  speed; tests check both paths give the same result.
* No BatchNorm or anything else that mixes examples within a batch, and dropout
  defaults to 0. That keeps per-example gradients well defined, which DP-SGD
  (src/defense.py) needs: it clips each example's gradient separately.
* forward(..., reduction="none") returns the loss of every token, which the
  memorization experiments use to score individual stories and canaries.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512   # maximum context length
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False      # bias in Linear/LayerNorm (GPT-2 had them; slightly better/faster without)
    use_sdpa: bool = False  # use torch's fused attention instead of the explicit implementation

    def to_dict(self):
        return asdict(self)


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (torch's built-in can't disable only the bias)."""

    def __init__(self, n: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_sdpa = cfg.use_sdpa
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)   # all heads' Q, K, V in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)      # mixes the heads back together
        self.proj.is_residual_proj = True
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # mask[i, j] is True where position i may attend to position j (j <= i).
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            # Similarity of every query with every key, scaled so the softmax
            # doesn't saturate as head_dim grows.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)        # (B, nh, T, T)
            # Causal mask: a token may not look at tokens after it.
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(q.dtype)                  # softmax in fp32 for stability
            att = self.attn_drop(att)
            y = att @ v                                                        # (B, nh, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)                      # concatenate heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.proj.is_residual_proj = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # tokens exchange information
        x = x + self.mlp(self.ln2(x))    # each token is processed on its own
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output layer reuses the token embedding matrix.
        # Saves vocab_size * n_embd parameters and usually helps small models.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # GPT-2 init: N(0, 0.02); residual output projections scaled down by
        # 1/sqrt(2 * n_layer) so the residual stream doesn't grow with depth.
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "is_residual_proj", False):
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. By convention, position embeddings are excluded for
        the 'non-embedding' count (the tied token embedding is counted once)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, reduction: str = "mean"):
        """idx: (B, T) token ids. targets: (B, T) next-token ids, or None.

        Returns (logits (B, T, vocab), loss). loss is a scalar for
        reduction="mean"/"sum", or (B, T) per-token losses for "none".
        Targets equal to -1 are ignored.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1, reduction=reduction)
        if reduction == "none":
            loss = loss.view(B, T)
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95), device_type="cuda"):
        """AdamW with weight decay on matrices only (not on LayerNorm gains or biases)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        kwargs = {"fused": True} if device_type == "cuda" else {}
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)
        except (RuntimeError, TypeError):  # fused not supported on this backend
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
                 eot_id: int | None = None, generator: torch.Generator | None = None):
        """Autoregressively extend idx (B, T) by up to max_new_tokens.

        temperature=0 means greedy decoding (always pick the most likely token),
        which the extraction attack uses. If eot_id is given, generation stops
        once every sequence in the batch has produced it.
        """
        was_training = self.training
        self.eval()
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]          # crop to the context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            if eot_id is not None:
                nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, eot_id), nxt)
                done |= nxt.squeeze(1) == eot_id
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and done.all():
                break
        self.train(was_training)
        return idx
