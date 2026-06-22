"""
Stage 1: GraphCodeBERT binary classifier (one per vulnerability type).
"""

import os
from typing import Optional

from pipeline.contract import WindowRecord
from config import STAGE1_MODEL, MODELS_DIR, VULN_TYPES

# GraphCodeBERT's maximum input length (in tokens).
_MAX_TOKENS = 512

# Returned when no fine-tuned model is available for a type — a neutral "unsure".
_FALLBACK_SCORE = 0.5

# dictionary remembering loaded models per type
_CACHE: dict[str, Optional[dict]] = {}


def _weights_path(vuln_type: str) -> str:
    #Return the on-disk path of the fine-tuned weights for a vuln type.
    return os.path.join(MODELS_DIR, "graphcodebert", f"graphcodebert_{vuln_type}.pt")


def _load_classifier(vuln_type: str) -> Optional[dict]:
    #Load the tokenizer + fine-tuned model for `vuln_type`.

    if vuln_type in _CACHE:
        return _CACHE[vuln_type]

    path = _weights_path(vuln_type)
    if not os.path.exists(path):
        # No fine-tuned weights yet — don't download the base model; use fallback.
        _CACHE[vuln_type] = None
        return None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            STAGE1_MODEL, num_labels=2
        )
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _CACHE[vuln_type] = {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        print(f"[stage1] Could not load GraphCodeBERT for {vuln_type!r}: {e}. "
              f"Using fallback score {_FALLBACK_SCORE}.")
        _CACHE[vuln_type] = None

    return _CACHE[vuln_type]


def predict(record: WindowRecord) -> None:
    #Run GraphCodeBERT on record.code for record.vulnerability_type
    vuln_type = record.vulnerability_type
    clf = _load_classifier(vuln_type)

    if clf is None:
        record.stage1_score = _FALLBACK_SCORE
        return

    try:
        import torch

        tokenizer = clf["tokenizer"]
        model = clf["model"]
        code = record.code or ""

        inputs = tokenizer(
            code,
            truncation=True,
            max_length=_MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        # probs[1] = probability of class 1 (vulnerable)
        record.stage1_score = float(probs[1])
    except Exception as e:
        print(f"[stage1] predict failed for {vuln_type!r}: {e}. "
              f"Using fallback score {_FALLBACK_SCORE}.")
        record.stage1_score = _FALLBACK_SCORE


def train(
    vuln_type: str,
    epochs: int = 3,
    batch_size: int = 16,
    records: Optional[list] = None,
) -> None:
    """
    Fine-tune GraphCodeBERT on data/processed/{vuln_type}_train.json.

    Args:
        vuln_type:  which classifier to train (one of VULN_TYPES).
        epochs:     number of passes over the training data.
        batch_size: examples per optimization step.
        records:    optional list[WindowRecord] to train on instead of reading
                    from disk. Used by tests to fit a tiny synthetic dataset
                    without needing the full processed splits.
    """
    if vuln_type not in VULN_TYPES:
        raise ValueError(f"Unknown vuln type: {vuln_type!r}. Must be one of {VULN_TYPES}")

    if records is None:
        records = _load_train_records(vuln_type)

    # Keep only labeled records (label must be 0 or 1).
    records = [r for r in records if r.label in (0, 1) and r.code and r.code.strip()]
    if len(records) < 2:
        raise ValueError(
            f"Need at least 2 labeled records to train {vuln_type!r}, got {len(records)}"
        )

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL, num_labels=2)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    texts = [r.code for r in records]
    labels = [int(r.label) for r in records]

    for epoch in range(epochs):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = torch.tensor(labels[i:i + batch_size])
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=_MAX_TOKENS,
                padding=True,
                return_tensors="pt",
            )
            optimizer.zero_grad()
            outputs = model(**enc, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()
        print(f"[stage1] {vuln_type!r} epoch {epoch + 1}/{epochs} done")

    out_path = _weights_path(vuln_type)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[stage1] Saved fine-tuned weights -> {out_path}")

    # Drop any cached (now-stale) model for this type.
    _CACHE.pop(vuln_type, None)


def _load_train_records(vuln_type: str) -> list:
    """Read the processed training split for a vuln type into WindowRecords."""
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
