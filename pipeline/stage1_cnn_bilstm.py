"""
Stage 1 (alternative backend): CNN-BiLSTM on top of a FROZEN GraphCodeBERT.


Architecture
     │  GraphCodeBERT tokenizer + encoder, run under torch.no_grad()
     │  (ALL GraphCodeBERT weights are FROZEN — never updated)
     >  last hidden states            [batch, seq_len, 768]
     │  parallel Conv1d filters of sizes 2,3,4,5 (STAGE1_CNN_FILTERS each) + ReLU
     >  concatenated feature maps      [batch, seq_len, 4*filters]
     │  two-layer BiLSTM (hidden = STAGE1_HIDDEN_DIM)
     >  contextual sequence           [batch, seq_len, 2*hidden]
     │  self-attention pooling (weights the most informative timesteps)
     >  context vector                [batch, 2*hidden]
     │  dropout -> linear -> (sigmoid) -> score 0.0–1.0
     >  record.stage1_score

"""

import os
from typing import Optional

from pipeline.contract import WindowRecord
from config import (
    STAGE1_MODEL,
    MODELS_DIR,
    VULN_TYPES,
    STAGE1_HIDDEN_DIM,
    STAGE1_CNN_FILTERS,
    STAGE1_DROPOUT,
)

# Convolution filter widths (in tokens) used in parallel.
_FILTER_SIZES = (2, 3, 4, 5)

# Default sequence length (tokens) fed to the frozen encoder.
_MAX_LENGTH = 256

# Neutral score returned when no trained head exists for a type.
_FALLBACK_SCORE = 0.5

# Cache of loaded predict-time bundles per vuln type: {tokenizer, encoder, classifier, max_length} or None.
_CACHE: dict[str, Optional[dict]] = {}

# The frozen GraphCodeBERT encoder + tokenizer are identical for every type and
# every trial, so we load them ONCE and reuse. (Big speed win during search.)
_ENCODER = None  # (tokenizer, encoder) or None

# The nn.Module class is built lazily so importing this file stays cheap (no torch).
_MODEL_CLASS = None


# Model definition (built lazily)


def _get_model_class():
    #Define (once) and return the CNNBiLSTMClassifier class. Imports torch lazily.
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    import torch
    import torch.nn as nn

    class CNNBiLSTMClassifier(nn.Module):
        #CNN -> BiLSTM -> self-attention -> linear head over frozen GraphCodeBERT embeddings.

        def __init__(self, embed_dim=768, cnn_filters=STAGE1_CNN_FILTERS,
                     filter_sizes=_FILTER_SIZES, hidden_dim=STAGE1_HIDDEN_DIM,
                     lstm_layers=2, dropout=STAGE1_DROPOUT):
            super().__init__()
            # Parallel convolutions over the token axis. padding=k//2 keeps the
            # output length ~ the input length so we retain a real sequence.
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, cnn_filters, kernel_size=k, padding=k // 2)
                for k in filter_sizes
            ])
            conv_out = cnn_filters * len(filter_sizes)
            self.bilstm = nn.LSTM(
                input_size=conv_out,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.attn = nn.Linear(hidden_dim * 2, 1)   # self-attention score per timestep
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, 1)     # final linear; sigmoid applied outside

        def forward(self, embeddings, attention_mask=None):
            # embeddings: [B, T, E]  (E = 768)
            x = embeddings.transpose(1, 2)                  # [B, E, T] for Conv1d
            conv_outs = [torch.relu(conv(x)) for conv in self.convs]  # each [B, F, ~T]
            min_len = min(c.size(2) for c in conv_outs)
            conv_outs = [c[:, :, :min_len] for c in conv_outs]
            feats = torch.cat(conv_outs, dim=1)             # [B, 4F, L]
            feats = feats.transpose(1, 2)                   # [B, L, 4F]

            lstm_out, _ = self.bilstm(feats)                # [B, L, 2H]

            scores = self.attn(lstm_out).squeeze(-1)        # [B, L]
            if attention_mask is not None:
                mask = attention_mask[:, :scores.size(1)]
                scores = scores.masked_fill(mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=1)          # [B, L]
            # Guard against all-(-inf) rows (fully padded) producing NaNs.
            weights = torch.nan_to_num(weights)
            context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # [B, 2H]

            logits = self.fc(self.dropout(context)).squeeze(-1)          # [B]
            return logits

    _MODEL_CLASS = CNNBiLSTMClassifier
    return _MODEL_CLASS


# ---------------------------------------------------------------------------
# Frozen encoder + embedding helpers (shared by predict, train, and the search)
# ---------------------------------------------------------------------------

def _load_encoder(device="cpu"):
    #Load and FREEZE the GraphCodeBERT encoder + tokenizer once; cache and reuse.
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
    encoder = AutoModel.from_pretrained(STAGE1_MODEL)
    for p in encoder.parameters():
        p.requires_grad = False          # freeze every GraphCodeBERT weight
    encoder.eval().to(device)
    _ENCODER = (tokenizer, encoder)
    return _ENCODER


def _embed_batch(tokenizer, encoder, texts, device, max_length):
    #Tokenize texts and return (last_hidden_states [B,T,768], attention_mask [B,T]).
    import torch
    enc = tokenizer(
        texts, truncation=True, max_length=max_length,
        padding=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        hidden = encoder(**enc).last_hidden_state
    return hidden, enc["attention_mask"]


def embed_records(records, device, max_length, batch_size=32, label=""):
    """
    Precompute frozen embeddings for a list of records ONCE.

    Returns (emb_list, labels) where emb_list[i] is a CPU tensor [len_i, 768]
    (padding stripped). Used by the search so each trial trains only the small
    head on cached embeddings instead of re-running the encoder every epoch.
    """
    import torch
    tokenizer, encoder = _load_encoder(device)
    emb_list, labels = [], []
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        texts = [r.code or "" for r in batch]
        hidden, mask = _embed_batch(tokenizer, encoder, texts, device, max_length)
        for i in range(len(batch)):
            length = int(mask[i].sum().item())
            emb_list.append(hidden[i, :length, :].cpu())
            labels.append(int(batch[i].label))
        if label and (start // batch_size) % 20 == 0:
            print(f"    embedding {label}: {min(start + batch_size, total)}/{total}", flush=True)
    return emb_list, labels


def collate(emb_list, device):
    """Pad a list of [len_i,768] tensors into (padded [B,Tmax,768], mask [B,Tmax])."""
    import torch
    lengths = [e.size(0) for e in emb_list]
    t_max = max(lengths) if lengths else 1
    embed_dim = emb_list[0].size(1) if emb_list else 768
    padded = torch.zeros(len(emb_list), t_max, embed_dim)
    mask = torch.zeros(len(emb_list), t_max, dtype=torch.long)
    for i, e in enumerate(emb_list):
        length = e.size(0)
        padded[i, :length, :] = e
        mask[i, :length] = 1
    return padded.to(device), mask.to(device)


def build_classifier(hp, device):
    """Construct a CNNBiLSTMClassifier from a hyperparameter dict and move to device."""
    cls = _get_model_class()
    model = cls(
        embed_dim=768,
        cnn_filters=hp.get("cnn_filters", STAGE1_CNN_FILTERS),
        filter_sizes=_FILTER_SIZES,
        hidden_dim=hp.get("hidden_dim", STAGE1_HIDDEN_DIM),
        lstm_layers=hp.get("lstm_layers", 2),
        dropout=hp.get("dropout", STAGE1_DROPOUT),
    )
    return model.to(device)


def compute_pos_weight(labels, device):
    """pos_weight for weighted BCE = (#safe / #vulnerable), to counter class imbalance."""
    import torch
    pos = sum(1 for l in labels if l == 1)
    neg = len(labels) - pos
    weight = (neg / pos) if pos > 0 else 1.0
    return torch.tensor([weight], device=device, dtype=torch.float32)


def run_classifier(classifier, emb_list, device, batch_size=64):
    """Return predicted probabilities (after sigmoid) for a list of cached embeddings."""
    import torch
    classifier.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(emb_list), batch_size):
            padded, mask = collate(emb_list[start:start + batch_size], device)
            logits = classifier(padded, mask)
            probs.extend(torch.sigmoid(logits).cpu().reshape(-1).tolist())
    return probs


def fit_classifier(classifier, emb_list, labels, hp, epochs, batch_size, device,
                   val_emb=None, val_labels=None, trial=None, optuna_mod=None,
                   beta=2.0, verbose=False, label=""):
    """
    Train the head on cached embeddings with weighted BCEWithLogitsLoss.

    After each epoch, if a validation set is given, computes the best achievable
    F-beta (beta>1 favors recall over precision — we care more about catching
    every vulnerability than about a few false positives). That value is printed
    (when verbose) and reported to the Optuna pruner, which kills hopeless trials.
    """
    import random
    import torch

    pos_weight = compute_pos_weight(labels, device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=hp["learning_rate"])

    n = len(emb_list)
    for epoch in range(epochs):
        classifier.train()
        order = list(range(n))
        random.Random(epoch).shuffle(order)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            padded, mask = collate([emb_list[k] for k in idx], device)
            y = torch.tensor([float(labels[k]) for k in idx], device=device)
            optimizer.zero_grad()
            logits = classifier(padded, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        if val_emb is not None:
            fb = _best_fbeta_from_probs(run_classifier(classifier, val_emb, device), val_labels, beta)
            if verbose:
                print(f"      {label} epoch {epoch + 1}/{epochs}: val_F{beta:g}={fb:.3f}", flush=True)
            if trial is not None:
                trial.report(fb, epoch)
                if trial.should_prune():
                    import optuna as _opt
                    raise (optuna_mod or _opt).TrialPruned()
    return classifier


def _best_fbeta_from_probs(probs, labels, beta=2.0):
    """Best F-beta over thresholds 0.05..0.95 (beta>1 weights recall higher)."""
    from sklearn.metrics import precision_score, recall_score
    b2 = beta * beta
    best = 0.0
    for i in range(5, 96, 5):
        th = i / 100.0
        preds = [1 if p >= th else 0 for p in probs]
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        denom = b2 * prec + rec
        fb = (1 + b2) * prec * rec / denom if denom > 0 else 0.0
        if fb > best:
            best = fb
    return float(best)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _checkpoint_path(vuln_type: str) -> str:
    return os.path.join(MODELS_DIR, f"cnn_bilstm_{vuln_type}.pt")


def save_checkpoint(path, classifier, hp, max_length=_MAX_LENGTH, threshold=0.5):
    """
    Save the head's weights, the hyperparameters needed to rebuild it, and the
    chosen decision threshold (the F-beta-optimal cutoff found during the search).
    """
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": classifier.state_dict(),
        "hp": {
            "cnn_filters": hp.get("cnn_filters", STAGE1_CNN_FILTERS),
            "hidden_dim": hp.get("hidden_dim", STAGE1_HIDDEN_DIM),
            "lstm_layers": hp.get("lstm_layers", 2),
            "dropout": hp.get("dropout", STAGE1_DROPOUT),
        },
        "max_length": max_length,
        "threshold": threshold,
    }, path)


# ---------------------------------------------------------------------------
# Public API: predict + train  (identical signatures to stage1_graphcodebert)
# ---------------------------------------------------------------------------

def _load(vuln_type: str) -> Optional[dict]:
    """Lazily load the predict-time bundle for a type; cache it. None = use fallback."""
    if vuln_type in _CACHE:
        return _CACHE[vuln_type]
    path = _checkpoint_path(vuln_type)
    if not os.path.exists(path):
        _CACHE[vuln_type] = None
        return None
    try:
        import torch
        device = "cpu"
        ckpt = torch.load(path, map_location=device)
        tokenizer, encoder = _load_encoder(device)
        classifier = build_classifier(ckpt["hp"], device)
        classifier.load_state_dict(ckpt["state_dict"])
        classifier.eval()
        _CACHE[vuln_type] = {
            "tokenizer": tokenizer, "encoder": encoder,
            "classifier": classifier, "max_length": ckpt.get("max_length", _MAX_LENGTH),
        }
    except Exception as e:
        print(f"[cnn_bilstm] Could not load model for {vuln_type!r}: {e}. "
              f"Using fallback score {_FALLBACK_SCORE}.")
        _CACHE[vuln_type] = None
    return _CACHE[vuln_type]


def predict(record: WindowRecord) -> None:
    """Score record.code with the CNN-BiLSTM head; set record.stage1_score in-place."""
    bundle = _load(record.vulnerability_type)
    if bundle is None:
        record.stage1_score = _FALLBACK_SCORE
        return
    try:
        import torch
        hidden, mask = _embed_batch(
            bundle["tokenizer"], bundle["encoder"], [record.code or ""],
            "cpu", bundle["max_length"],
        )
        with torch.no_grad():
            logits = bundle["classifier"](hidden, mask)
            prob = torch.sigmoid(logits).reshape(-1)[0]
        record.stage1_score = float(prob)
    except Exception as e:
        print(f"[cnn_bilstm] predict failed for {record.vulnerability_type!r}: {e}. "
              f"Using fallback score {_FALLBACK_SCORE}.")
        record.stage1_score = _FALLBACK_SCORE


def train(vuln_type: str, epochs: int = 3, batch_size: int = 16,
          learning_rate: float = 1e-3, hidden_dim: Optional[int] = None,
          cnn_filters: Optional[int] = None, dropout: Optional[float] = None,
          lstm_layers: int = 2, max_length: int = _MAX_LENGTH,
          records: Optional[list] = None) -> None:
    """
    Train the CNN-BiLSTM head on data/processed/{vuln_type}_train.json and save it
    to models/cnn_bilstm_{vuln_type}.pt. GraphCodeBERT stays frozen throughout.

    The required public signature is train(vuln_type, epochs, batch_size); the
    extra keyword args let the search pass specific hyperparameters (and tests
    pass `records`). Class imbalance is handled via weighted BCEWithLogitsLoss.
    """
    if vuln_type not in VULN_TYPES:
        raise ValueError(f"Unknown vuln type: {vuln_type!r}. Must be one of {VULN_TYPES}")

    hp = {
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim if hidden_dim is not None else STAGE1_HIDDEN_DIM,
        "cnn_filters": cnn_filters if cnn_filters is not None else STAGE1_CNN_FILTERS,
        "dropout": dropout if dropout is not None else STAGE1_DROPOUT,
        "lstm_layers": lstm_layers,
    }

    if records is None:
        records = _load_train_records(vuln_type)
    records = [r for r in records if r.label in (0, 1) and r.code and r.code.strip()]
    if len(records) < 2:
        raise ValueError(f"Need at least 2 labeled records to train {vuln_type!r}, got {len(records)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_list, labels = embed_records(records, device, max_length)
    classifier = build_classifier(hp, device)
    fit_classifier(classifier, emb_list, labels, hp, epochs, batch_size, device)

    save_checkpoint(_checkpoint_path(vuln_type), classifier, hp, max_length)
    print(f"[cnn_bilstm] Saved head -> {_checkpoint_path(vuln_type)}")
    _CACHE.pop(vuln_type, None)


def _load_train_records(vuln_type: str) -> list:
    """Read the processed training split into WindowRecords."""
    import json
    from data.loader import PROCESSED_DIR
    train_path = os.path.join(PROCESSED_DIR, f"{vuln_type}_train.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"No training split at {train_path!r}. Run scripts/setup_data.py first."
        )
    with open(train_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [WindowRecord.from_dict(d) for d in rows]
