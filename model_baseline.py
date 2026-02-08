# model_baseline.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


@dataclass
class InitArtifacts:
    pi: torch.Tensor  # (num_items,)
    num_items: int


def initialize(
    train_path: str,
    num_items: Optional[int] = None,
    item_col: str = "item_id",
    label_col: str = "correct",
    sep: str = "\t",
    smooth: float = 0.5,
    global_prior: Optional[float] = None,
) -> InitArtifacts:
    """
    Compute fixed p_i from train split (item-wise accuracy) and return as torch tensor.

    - smooth: Laplace-like smoothing strength. Uses:
        p_i = (sum_correct + smooth * prior) / (count + smooth)
      where prior defaults to global train accuracy unless global_prior is provided.
    """
    df = pd.read_csv(train_path, sep=sep)

    if num_items is None:
        num_items = int(df[item_col].max() + 1)

    # Global prior for smoothing
    if global_prior is None:
        global_prior = float(df[label_col].mean()) if len(df) > 0 else 0.5

    grp = df.groupby(item_col)[label_col].agg(["sum", "count"])
    sums = grp["sum"].to_dict()
    cnts = grp["count"].to_dict()

    pi = torch.full((num_items,), float(global_prior), dtype=torch.float32)
    for i, c in cnts.items():
        s = float(sums[i])
        c = float(c)
        p = (s + smooth * float(global_prior)) / (c + smooth)
        pi[int(i)] = float(p)

    return InitArtifacts(pi=pi, num_items=num_items)


class BaselineKT(nn.Module):
    """
    Implements Decoupled Key-Value KT:
      P(q_i=1|H) = alpha * p_i + (1-alpha) * Σ_{k in H} beta_{ik} * c_{ik}

    - Key (Attention): Determines 'beta' (Relevance)
      beta_{ik} = softmax( <q_i, k_k> / sqrt(r) )
      
    - Value (Prediction): Determines 'c' (Conditional Probability)
      c_{ik} = sigmoid( b_i + <q_i, v_k> / sqrt(r) )

    This decoupling allows the model to attend strongly (high beta) to a history item 
    that predicts failure (low c), effectively capturing P(q_i=1 | q_j=0) signals.
    """
    def __init__(
        self,
        pi: torch.Tensor,
        rank: int = 128,
        use_4bit: bool = False, # Kept for compatibility, ignored
        pad_id: int = -1,
        init_embed_std: float = 1e-3,
        dropout: float = 0.2, # [NEW] Added dropout probability
    ):
        super().__init__()
        assert pi.ndim == 1
        self.num_items = int(pi.shape[0])
        self.rank = int(rank)
        self.pad_id = int(pad_id)

        self.register_buffer("pi", pi.float(), persistent=True)

        self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # [NEW] Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

        # Query Embedding
        self.q_emb = nn.Embedding(self.num_items, self.rank)

        # [KEY Embeddings] for Attention (Relevance)
        self.k_emb_correct = nn.Embedding(self.num_items, self.rank)
        self.k_emb_wrong = nn.Embedding(self.num_items, self.rank)

        # [VALUE Embeddings] for Prediction (Probability)
        # This allows capturing P(qi|qj) distinct from relevance
        self.v_emb_correct = nn.Embedding(self.num_items, self.rank)
        self.v_emb_wrong = nn.Embedding(self.num_items, self.rank)

        # Per-target bias (applied to Value logits for difficulty adjustment)
        self.b_i = nn.Embedding(self.num_items, 1)

        self._init_parameters(init_embed_std=init_embed_std)

    def _init_parameters(self, init_embed_std: float = 1e-3):
        # Initialize all embeddings
        for mod in [self.q_emb, 
                    self.k_emb_correct, self.k_emb_wrong,
                    self.v_emb_correct, self.v_emb_wrong]:
            w = getattr(mod, "weight", None)
            if w is not None and torch.is_floating_point(w):
                nn.init.normal_(w, mean=0.0, std=float(init_embed_std))

        with torch.no_grad():
            b = _safe_logit(self.pi).unsqueeze(-1)  # (num_items, 1)
            self.b_i.weight.copy_(b)

    def forward(
        self,
        hist_items: torch.LongTensor,     # (B, L) padded with pad_id
        hist_correct: torch.LongTensor,   # (B, L) values in {0,1}, padded with -1
        target_items: torch.LongTensor,   # (B,)
        return_logits: bool = False,
    ) -> torch.Tensor:
        device = target_items.device

        # p_i lookup
        p_i = self.pi[target_items].to(device)  # (B,)

        # no history
        if hist_items.numel() == 0 or hist_items.shape[1] == 0:
            alpha = torch.sigmoid(self.alpha_logit)
            probs = alpha * p_i + (1.0 - alpha) * p_i
            probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            return _safe_logit(probs) if return_logits else probs

        # valid history positions
        mask = (hist_items != self.pad_id) & (hist_correct >= 0)  # (B, L)

        safe_hist = hist_items.clone()
        safe_hist[~mask] = 0

        # Query & Bias
        q = self.q_emb(target_items)              # (B, r)
        b = self.b_i(target_items).squeeze(-1)    # (B,)
        
        # [NEW] Apply Dropout to Query
        q = self.dropout(q)

        # ---------------------------------------------------------
        # 1. Attention Mechanism (Using Keys)
        # ---------------------------------------------------------
        k_c = self.k_emb_correct(safe_hist)
        k_w = self.k_emb_wrong(safe_hist)
        corr = (hist_correct == 1).unsqueeze(-1)  # (B, L, 1)
        
        k = torch.where(corr, k_c, k_w)           # (B, L, r)
        
        # [NEW] Apply Dropout to selected Keys
        k = self.dropout(k)

        # Attention Logits: <q, k> / sqrt(r)
        att_logits = torch.einsum("br,blr->bl", q, k) / math.sqrt(self.rank)
        
        # Softmax over valid history
        # [FIX] Use -1e4 instead of -1e9 for float16 compatibility (overflow prevention)
        att_logits = att_logits.masked_fill(~mask, -1e4)
        beta = torch.softmax(att_logits, dim=1)  # (B, L)

        # ---------------------------------------------------------
        # 2. Prediction Mechanism (Using Values)
        # ---------------------------------------------------------
        val_c = self.v_emb_correct(safe_hist)
        val_w = self.v_emb_wrong(safe_hist)
        
        v_emb = torch.where(corr, val_c, val_w)   # (B, L, r)
        
        # [NEW] Apply Dropout to selected Values
        v_emb = self.dropout(v_emb)
        
        # Value Logits: b_i + <q, v> / sqrt(r)
        # (Bias implies 'difficulty' of target, so it belongs to prediction)
        val_logits = torch.einsum("br,blr->bl", q, v_emb) / math.sqrt(self.rank)
        val_logits = val_logits + b.unsqueeze(1)
        
        c = torch.sigmoid(val_logits)  # (B, L)

        # ---------------------------------------------------------
        # 3. Aggregation
        # ---------------------------------------------------------
        alpha = torch.sigmoid(self.alpha_logit)
        hist_term = torch.sum(beta * c, dim=1)  # (B,)
        
        probs = alpha * p_i + (1.0 - alpha) * hist_term

        if return_logits:
            probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
            return _safe_logit(probs)
        return probs

    @torch.no_grad()
    def diagnostics(self) -> dict:
        alpha = float(torch.sigmoid(self.alpha_logit).item())
        return {
            "alpha": alpha,
            "num_items": self.num_items,
            "rank": self.rank,
            "pad_id": self.pad_id,
            "architecture": "Decoupled Key-Value",
        }