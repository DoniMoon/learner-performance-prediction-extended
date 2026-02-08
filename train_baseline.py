# train_baseline.py
import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_

from model_baseline import BaselineKT, initialize
from utils import Logger, Saver, Metrics  # same utilities used by other trainers


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sequences(df: pd.DataFrame, max_seq_len: int, pad_id: int = -1):
    """
    Returns list of (item_seq, label_seq) torch tensors, chunked by max_seq_len.
    Assumes df has columns: user_id, item_id, timestamp, correct
    """
    # Ensure chronological order per user
    if "timestamp" in df.columns:
        df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    else:
        df = df.sort_values(["user_id"], kind="mergesort")

    seqs = []
    for _, u_df in df.groupby("user_id", sort=False):
        items = torch.tensor(u_df["item_id"].values, dtype=torch.long)
        labels = torch.tensor(u_df["correct"].values, dtype=torch.long)
        if len(items) == 0:
            continue
        # Chunk
        for s in range(0, len(items), max_seq_len):
            it = items[s:s + max_seq_len]
            lb = labels[s:s + max_seq_len]
            if len(it) > 0:
                seqs.append((it, lb))
    return seqs


def split_train_val_ensure_all_items(
    seqs,
    all_items_set,
    train_ratio: float = 0.8,
    seed: int = 0,
):
    """
    Split at sequence level, then move sequences from val to train until
    every item in all_items_set appears in train at least once.
    """
    rng = random.Random(seed)
    seqs = list(seqs)
    rng.shuffle(seqs)

    n_train = int(train_ratio * len(seqs))
    train = seqs[:n_train]
    val = seqs[n_train:]

    def items_in(seqs_):
        s = set()
        for it, _ in seqs_:
            s.update(it.tolist())
        return s

    train_items = items_in(train)
    missing = set(all_items_set) - train_items

    if missing:
        # Greedy move: scan val, move sequences that cover missing items
        moved = 0
        i = 0
        while missing and i < len(val):
            it, lb = val[i]
            covers = missing.intersection(it.tolist())
            if covers:
                train.append((it, lb))
                train_items.update(it.tolist())
                missing = set(all_items_set) - train_items
                val.pop(i)
                moved += 1
            else:
                i += 1
        if missing:
            print(f"[WARN] Could not cover all missing items in train split. Remaining: {len(missing)}")
        else:
            print(f"[INFO] Moved {moved} sequences from val→train to cover all items.")

    return train, val


def make_batches(seqs, batch_size: int, pad_id: int = -1, randomize: bool = True, seed: int = 0):
    if randomize:
        rng = random.Random(seed)
        seqs = list(seqs)
        rng.shuffle(seqs)

    batches = []
    for k in range(0, len(seqs), batch_size):
        chunk = seqs[k:k + batch_size]
        item_list = [it for it, _ in chunk]
        label_list = [lb for _, lb in chunk]
        items = pad_sequence(item_list, batch_first=True, padding_value=pad_id)      # (B,T)
        labels = pad_sequence(label_list, batch_first=True, padding_value=-1)       # (B,T) label pad=-1
        batches.append((items, labels))
    return batches


def compute_auc_from_probs(probs: torch.Tensor, labels: torch.Tensor):
    """
    probs: (N,) float in [0,1]
    labels: (N,) float {0,1}
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    if len(np.unique(labels_np)) == 1:
        return accuracy_score(labels_np, (probs_np >= 0.5).astype(np.int32))
    return roc_auc_score(labels_np, probs_np)


def train_one_epoch(
    model: BaselineKT,
    batches,
    optimizer,
    criterion,
    max_hist: int,
    grad_clip: float,
    use_amp: bool,
):
    model.train()
    metrics = Metrics()
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    for items, labels in tqdm(batches):
        items = items.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        B, T = items.shape
        pad_id = model.pad_id

        # Left-pad items with pad_id to build fixed-length history windows
        left_pad_items = torch.full((B, max_hist), pad_id, device=items.device, dtype=items.dtype)
        items_pad = torch.cat([left_pad_items, items], dim=1)
        
        left_pad_labels = torch.full((B, max_hist), -1, device=labels.device, dtype=labels.dtype)
        labels_pad = torch.cat([left_pad_labels, labels], dim=1)

        total_loss = 0.0
        total_steps = 0

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda',enabled=use_amp):
            # Unroll time; each step uses a (B, max_hist) history window
            for t in range(T):
                y = labels[:, t]
                target = items[:, t]
                
                # history windows
                hist_items = items_pad[:, t:t + max_hist]
                hist_correct = labels_pad[:, t:t + max_hist]
                
                valid = (y >= 0) & (target >= 0) & (target < model.num_items)
                if not torch.any(valid):
                    continue
                
                hist_items_v = hist_items[valid]
                hist_correct_v = hist_correct[valid]
                target_v = target[valid]
                y_v = y[valid].float()
                
                logits_v = model(hist_items_v, hist_correct_v, target_v, return_logits=True)
                loss = criterion(logits_v, y_v)
                

                total_loss = total_loss + loss
                total_steps += 1

            if total_steps == 0:
                continue

            loss_mean = total_loss / total_steps

        scaler.scale(loss_mean).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metrics.store({"loss/train": float(loss_mean.detach().item())})

    return metrics


@torch.no_grad()
def eval_epoch(model: BaselineKT, batches, criterion, max_hist: int, use_amp: bool):
    model.eval()
    metrics = Metrics()

    all_probs = []
    all_labels = []

    for items, labels in batches:
        items = items.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        B, T = items.shape
        pad_id = model.pad_id

        left_pad_items = torch.full((B, max_hist), pad_id, device=items.device, dtype=items.dtype)
        items_pad = torch.cat([left_pad_items, items], dim=1)
        
        left_pad_labels = torch.full((B, max_hist), -1, device=labels.device, dtype=labels.dtype)
        labels_pad = torch.cat([left_pad_labels, labels], dim=1)


        total_loss = 0.0
        total_steps = 0

        with torch.amp.autocast('cuda',enabled=use_amp):
            for t in range(T):
                y = labels[:, t]
                target = items[:, t]
                
                hist_items = items_pad[:, t:t + max_hist]
                hist_correct = labels_pad[:, t:t + max_hist]
                
                valid = (y >= 0) & (target >= 0) & (target < model.num_items)
                if not torch.any(valid):
                    continue
                
                hist_items_v = hist_items[valid]
                hist_correct_v = hist_correct[valid]
                target_v = target[valid]
                y_v = y[valid].float()
                
                logits_v = model(hist_items_v, hist_correct_v, target_v, return_logits=True)
                loss = criterion(logits_v, y_v)
                
                total_loss += float(loss.detach().item())
                total_steps += 1
                
                probs = torch.sigmoid(logits_v)
                all_probs.append(probs.detach().cpu())
                all_labels.append(y_v.detach().cpu())


        
        if total_steps > 0:
            metrics.store({"loss/val": total_loss / total_steps})

    if all_probs:
        probs_cat = torch.cat(all_probs, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        auc = compute_auc_from_probs(probs_cat, labels_cat)
        metrics.store({"auc/val": float(auc)})

    return metrics


@torch.no_grad()
def predict_test(model: BaselineKT, seqs, batch_size: int, max_hist: int, use_amp: bool, seed: int = 0):
    """
    Predicts on test data using batching while preserving the correct user-sequence order.
    
    Instead of flattening results immediately (which interleaves users),
    we collect results per-sequence within the batch and then concatenate them.
    """
    model.eval()
    pad_id = model.pad_id

    batches = make_batches(seqs, batch_size, pad_id=pad_id, randomize=False, seed=seed)
    preds_all = []

    for items, labels in tqdm(batches):
        items = items.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        B, T = items.shape

        # Container to hold predictions for each sequence in the batch separately
        # batch_preds[b] will be a list of float probabilities for sequence b
        batch_preds = [[] for _ in range(B)]

        left_pad_items = torch.full((B, max_hist), pad_id, device=items.device, dtype=items.dtype)
        items_pad = torch.cat([left_pad_items, items], dim=1)
        
        left_pad_labels = torch.full((B, max_hist), -1, device=labels.device, dtype=labels.dtype)
        labels_pad = torch.cat([left_pad_labels, labels], dim=1)

        with torch.amp.autocast('cuda', enabled=use_amp):
            for t in range(T):
                y = labels[:, t]
                target = items[:, t]
                
                hist_items = items_pad[:, t:t + max_hist]
                hist_correct = labels_pad[:, t:t + max_hist]
                
                valid_mask = (y >= 0) & (target >= 0) & (target < model.num_items)
                
                if not torch.any(valid_mask):
                    continue
                
                # Model forward on valid indices only
                hist_items_v = hist_items[valid_mask]
                hist_correct_v = hist_correct[valid_mask]
                target_v = target[valid_mask]
                
                logits_v = model(hist_items_v, hist_correct_v, target_v, return_logits=True)
                probs_v = torch.sigmoid(logits_v).detach().cpu().numpy()
                
                # Map predictions back to their original batch index
                valid_indices = torch.nonzero(valid_mask).squeeze(-1).cpu().numpy()
                
                # Distribute results to the correct sequence buffer
                for i, batch_idx in enumerate(valid_indices):
                    batch_preds[batch_idx].append(probs_v[i])

        # Concatenate results in batch order (Seq 0 -> Seq 1 -> ... -> Seq B-1)
        # This reconstructs the original flattened order (User 1 all times -> User 2 all times)
        for b in range(B):
            preds_all.extend(batch_preds[b])

    if preds_all:
        return np.array(preds_all, dtype=np.float32)
    return np.empty((0,), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train baseline KT (alpha + beta + low-rank c_ik, fixed p_i).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="runs/baseline")
    parser.add_argument("--savedir", type=str, default="save/baseline")

    parser.add_argument("--max_seq_len", type=int, default=200, help="Chunk length per user sequence.")
    parser.add_argument("--max_hist", type=int, default=200, help="History window length used by the model.")
    parser.add_argument("--rank", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_4bit", action="store_true", help="Best-effort 4-bit via bitsandbytes when available.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this baseline."

    set_seed(args.seed)
    use_amp = (not args.no_amp)

    data_dir = os.path.join("data", args.dataset)
    train_path = os.path.join(data_dir, "preprocessed_data_train.csv")
    test_path = os.path.join(data_dir, "preprocessed_data_test.csv")

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    test_df["_orig_idx"] = np.arange(len(test_df))


    # Determine num_items from union of train/test (safer for embeddings)
    full_max_item = max(train_df["item_id"].max(), test_df["item_id"].max())
    num_items = int(full_max_item + 1)

    # Build sequences from the provided train file, then split train:val (~8:2)
    seqs_all = get_sequences(train_df, max_seq_len=args.max_seq_len, pad_id=-1)
    all_items_set = set(train_df["item_id"].unique().tolist())
    train_seqs, val_seqs = split_train_val_ensure_all_items(
        seqs_all, all_items_set, train_ratio=0.8, seed=args.seed
    )

    # Save the train split to a temp file so model_baseline.initialize can compute p_i
    tmp_train_split_path = os.path.join(data_dir, "_baseline_train_split.tsv")
    tmp_df_rows = []
    # Reconstruct a minimal df for initialize (item_id, correct)
    # We don’t need user_id/timestamp for p_i.
    for it, lb in train_seqs:
        tmp_df_rows.append(
            pd.DataFrame({"item_id": it.numpy(), "correct": lb.numpy()})
        )
    tmp_df = pd.concat(tmp_df_rows, axis=0, ignore_index=True)
    tmp_df.to_csv(tmp_train_split_path, sep="\t", index=False)

    init = initialize(
        train_path=tmp_train_split_path,
        num_items=num_items,
        item_col="item_id",
        label_col="correct",
        sep="\t",
        smooth=0.5,
        global_prior=None,
    )

    model = BaselineKT(
        pi=init.pi,
        rank=args.rank,
        use_4bit=args.use_4bit,
        pad_id=-1,
        init_embed_std=1e-3,
    ).cuda()

    print("[INFO] Model:", model.diagnostics())

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare batches
    train_batches = make_batches(train_seqs, args.batch_size, pad_id=-1, randomize=True, seed=args.seed)
    val_batches = make_batches(val_seqs, args.batch_size, pad_id=-1, randomize=False, seed=args.seed)

    param_str = (
        f"{args.dataset},"
        f"bs={args.batch_size},"
        f"seq={args.max_seq_len},"
        f"hist={args.max_hist},"
        f"r={args.rank},"
        f"4bit={args.use_4bit},"
        f"seed={args.seed}"
    )
    logger = Logger(os.path.join(args.logdir, param_str))
    saver = Saver(args.savedir, param_str)

    best_auc = -1.0
    no_improve = 0

    for epoch in range(args.num_epochs):
        tr_metrics = train_one_epoch(
            model, train_batches, optimizer, criterion,
            max_hist=args.max_hist, grad_clip=args.grad_clip, use_amp=use_amp
        )
        va_metrics = eval_epoch(
            model, val_batches, criterion,
            max_hist=args.max_hist, use_amp=use_amp
        )

        avg = {}
        avg.update(tr_metrics.average())
        avg.update(va_metrics.average())
        avg["epoch"] = epoch

        logger.log_scalars(avg, epoch)
        print(f"[Epoch {epoch}] " + ", ".join([f"{k}={v:.5f}" for k, v in avg.items() if k != "epoch"]))

        auc_val = avg.get("auc/val", None)
        if auc_val is not None:
            if auc_val > best_auc + args.min_delta:
                best_auc = auc_val
                no_improve = 0
                saver.save(best_auc, model)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[EarlyStopping] No improvement for {args.patience} epochs. Stop.")
                    break


    logger.close()

    # Predict on test split and write back to preprocessed_data_test.csv
    # NOTE: predict_test interleaves data if batch_size > 1. 
    # Must use batch_size=1 to maintain row-level alignment with the sorted DataFrame.
    test_df_sorted = test_df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    test_seqs = get_sequences(test_df_sorted, max_seq_len=args.max_seq_len, pad_id=-1)
    
    test_preds = predict_test(
        model, test_seqs,
        batch_size=args.batch_size, # Optimized: safe to use batch_size now
        max_hist=args.max_hist,
        use_amp=use_amp,
        seed=args.seed,
    )


    if len(test_preds) != len(test_df_sorted):
        raise RuntimeError(f"Pred length mismatch: preds={len(test_preds)} vs test_df_sorted={len(test_df_sorted)}")
    
    test_df_sorted["BASELINE"] = test_preds
    
    test_out = test_df_sorted.sort_values("_orig_idx", kind="mergesort").drop(columns=["_orig_idx"])
    test_out.to_csv(test_path, sep="\t", index=False)
    
    print("auc_test =", roc_auc_score(test_out["correct"].values, test_out["BASELINE"].values))


if __name__ == "__main__":
    main()