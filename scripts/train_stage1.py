"""
Stage 1 training with Bayesian hyperparameter search (Optuna).

For EACH of the 7 vulnerability types this script:
  1. Loads the processed train/val/test splits from data/processed/.
  2. Runs an Optuna study (Bayesian optimization via the TPE sampler) that
     tries several hyperparameter combinations ("experiments"/trials).
  3. Each trial fine-tunes GraphCodeBERT and measures VALIDATION F1.
  4. Whenever a trial beats the best so far, its weights are saved to
     models/graphcodebert_{type}.pt  (so the pipeline's predict() picks them up).
  5. After the search, the best model is evaluated once on the TEST split.

"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STAGE1_MODEL, MODELS_DIR, RESULTS_DIR, VULN_TYPES


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 GraphCodeBERT hyperparameter search.")
    p.add_argument("--types", nargs="+", default=VULN_TYPES,
                   help="Which vulnerability types to train (default: all 7).")
    p.add_argument("--trials", type=int, default=40,
                   help="Max trials per type (the time budget usually stops you first).")
    p.add_argument("--hours-per-type", type=float, default=3.0,
                   help="Time budget per type in hours (0 = no time limit).")
    p.add_argument("--max-train-samples", type=int, default=4000,
                   help="Cap on training windows per trial (class-balanced subsample).")
    p.add_argument("--max-eval-samples", type=int, default=2000,
                   help="Cap on validation windows used to score each trial.")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max tokens fed to the model (lower = faster).")
    p.add_argument("--save-all-trials", action="store_true",
                   help="Also save every trial's weights under models/trials/.")
    p.add_argument("--results-file", default=os.path.join(RESULTS_DIR, "stage1_experiments.json"),
                   help="Where the full results JSON is written.")
    p.add_argument("--storage", default=os.path.join(RESULTS_DIR, "optuna_stage1.db"),
                   help="SQLite file for Optuna (enables resume). Empty string = in-memory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _subsample(records, max_n, seed):
    """Return at most max_n records, keeping the vulnerable/safe ratio roughly intact."""
    import random
    if max_n is None or len(records) <= max_n:
        return list(records)
    rng = random.Random(seed)
    pos = [r for r in records if r.label == 1]
    neg = [r for r in records if r.label == 0]
    if not pos or not neg:
        # single class — just take a flat sample
        return rng.sample(list(records), max_n)
    ratio = len(pos) / len(records)
    n_pos = min(len(pos), max(1, round(max_n * ratio)))
    n_neg = min(len(neg), max_n - n_pos)
    sample = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(sample)
    return sample


# ---------------------------------------------------------------------------
# Train / evaluate one model
# ---------------------------------------------------------------------------

def _train_model(train_recs, hp, device, max_length):
    """Fine-tune a fresh GraphCodeBERT with the given hyperparameters. Returns (tokenizer, model)."""
    import random
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL, num_labels=2).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"]
    )

    texts = [r.code for r in train_recs]
    labels = [int(r.label) for r in train_recs]
    bs = hp["batch_size"]

    for epoch in range(hp["epochs"]):
        order = list(range(len(texts)))
        random.Random(epoch).shuffle(order)  # reshuffle each epoch
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            batch_texts = [texts[k] for k in idx]
            batch_labels = torch.tensor([labels[k] for k in idx]).to(device)
            enc = tokenizer(
                batch_texts, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(device)
            optimizer.zero_grad()
            outputs = model(**enc, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()

    return tokenizer, model


def _evaluate(tokenizer, model, recs, device, max_length, batch_size=32):
    """Run the model over recs and return F1/precision/recall/accuracy (all floats)."""
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for start in range(0, len(recs), batch_size):
            batch = recs[start:start + batch_size]
            texts = [r.code for r in batch]
            enc = tokenizer(
                texts, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(int(r.label) for r in batch)

    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": len(recs),
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_results(path, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _cleanup(*objs):
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# One Optuna study per vulnerability type
# ---------------------------------------------------------------------------

def run_study_for_type(vuln_type, args, device, results, optuna):
    from data.loader import load_vudenc

    print(f"\n{'=' * 70}\n[{vuln_type}] starting\n{'=' * 70}", flush=True)

    # --- load data ---
    try:
        train, val, test = load_vudenc(vuln_type)
    except FileNotFoundError as e:
        print(f"[{vuln_type}] SKIP: {e}", flush=True)
        results["types"][vuln_type] = {"status": "skipped: no processed data"}
        _save_results(args.results_file, results)
        return

    train = _subsample(train, args.max_train_samples, seed=args.seed)
    val_eval = _subsample(val, args.max_eval_samples, seed=args.seed)

    train_classes = set(r.label for r in train)
    if train_classes != {0, 1}:
        print(f"[{vuln_type}] SKIP: training subsample has classes {train_classes} (need both 0 and 1).",
              flush=True)
        results["types"][vuln_type] = {"status": f"skipped: single-class train ({train_classes})"}
        _save_results(args.results_file, results)
        return

    model_path = os.path.join(MODELS_DIR, f"graphcodebert_{vuln_type}.pt")
    state = {"best_f1": -1.0, "best_params": None, "best_trial": None}

    results["types"][vuln_type] = {
        "status": "running",
        "n_train": len(train), "n_val": len(val_eval), "n_test": len(test),
        "trials": [], "model_path": model_path, "best": None, "test": None,
    }
    type_results = results["types"][vuln_type]

    # --- Optuna study (resumable via SQLite storage) ---
    storage = f"sqlite:///{args.storage}" if args.storage else None
    study = optuna.create_study(
        direction="maximize",
        study_name=f"stage1_{vuln_type}",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    # If resuming a study that already has results, restore the best score so we
    # don't overwrite an already-good saved model with a worse one.
    if study.trials and study.best_value is not None:
        state["best_f1"] = study.best_value
        state["best_trial"] = study.best_trial.number
        state["best_params"] = dict(study.best_trial.params)

    def objective(trial):
        hp = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "epochs": trial.suggest_int("epochs", 1, 3),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        }
        t0 = time.time()
        try:
            tokenizer, model = _train_model(train, hp, device, args.max_length)
            metrics = _evaluate(tokenizer, model, val_eval, device, args.max_length)
            f1 = metrics["f1"]

            improved = f1 > state["best_f1"]
            if improved:
                state.update(best_f1=f1, best_params=hp, best_trial=trial.number)
                os.makedirs(MODELS_DIR, exist_ok=True)
                import torch
                torch.save(model.state_dict(), model_path)

            if args.save_all_trials:
                import torch
                trials_dir = os.path.join(MODELS_DIR, "trials")
                os.makedirs(trials_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(trials_dir, f"graphcodebert_{vuln_type}_trial{trial.number}.pt"))

            _cleanup(model, tokenizer)
        except Exception as e:
            print(f"[{vuln_type}] trial {trial.number} FAILED: {e}", flush=True)
            metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "n": len(val_eval)}
            f1 = 0.0
            improved = False

        secs = round(time.time() - t0, 1)
        type_results["trials"].append({
            "number": trial.number,
            "params": hp,
            "val_f1": round(metrics["f1"], 4),
            "val_precision": round(metrics["precision"], 4),
            "val_recall": round(metrics["recall"], 4),
            "seconds": secs,
            "improved": improved,
        })
        type_results["best"] = {
            "trial": state["best_trial"], "params": state["best_params"],
            "val_f1": round(state["best_f1"], 4),
        }
        _save_results(args.results_file, results)  # crash-safe after every trial

        star = "  <-- BEST" if improved else ""
        print(f"[{vuln_type}] trial {trial.number}: val_f1={f1:.3f}  "
              f"lr={hp['learning_rate']:.1e} bs={hp['batch_size']} "
              f"ep={hp['epochs']} wd={hp['weight_decay']:.3f}  ({secs:.0f}s){star}", flush=True)
        return f1

    timeout = args.hours_per_type * 3600 if args.hours_per_type and args.hours_per_type > 0 else None
    study.optimize(objective, n_trials=args.trials, timeout=timeout)

    # --- final: evaluate the best saved model on the full test split ---
    test_metrics = None
    if os.path.exists(model_path) and len(test) > 0:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL, num_labels=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            test_eval = _subsample(test, max(args.max_eval_samples * 3, args.max_eval_samples), seed=7)
            test_metrics = _evaluate(tokenizer, model, test_eval, device, args.max_length)
            _cleanup(model, tokenizer)
        except Exception as e:
            print(f"[{vuln_type}] test evaluation failed: {e}", flush=True)

    type_results["status"] = "done"
    type_results["test"] = test_metrics
    _save_results(args.results_file, results)

    best_f1 = state["best_f1"]
    test_f1 = test_metrics["f1"] if test_metrics else float("nan")
    print(f"[{vuln_type}] DONE  best_val_f1={best_f1:.3f}  test_f1={test_f1:.3f}", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(results) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("STAGE 1 — HYPERPARAMETER SEARCH SUMMARY")
    lines.append("=" * 78)

    # One detailed block per type.
    for vt, info in results["types"].items():
        lines.append(f"\n### {vt}  —  {info.get('status', '?')}")
        trials = info.get("trials") or []
        if trials:
            lines.append(f"  {'trial':>5} {'val_f1':>7} {'prec':>6} {'rec':>6} "
                         f"{'lr':>9} {'bs':>3} {'ep':>3} {'wd':>6} {'sec':>6}")
            for t in trials:
                p = t["params"]
                lines.append(
                    f"  {t['number']:>5} {t['val_f1']:>7.3f} {t['val_precision']:>6.3f} "
                    f"{t['val_recall']:>6.3f} {p['learning_rate']:>9.1e} {p['batch_size']:>3} "
                    f"{p['epochs']:>3} {p['weight_decay']:>6.3f} {t['seconds']:>6.0f}"
                    + ("  *" if t.get("improved") else "")
                )
        best = info.get("best")
        if best and best.get("params"):
            lines.append(f"  BEST: trial {best['trial']}  val_f1={best['val_f1']:.3f}  params={best['params']}")
        test = info.get("test")
        if test:
            lines.append(f"  TEST: f1={test['f1']:.3f}  precision={test['precision']:.3f}  "
                         f"recall={test['recall']:.3f}  acc={test['accuracy']:.3f}  (n={test['n']})")

    # One-line overview at the bottom.
    lines.append("\n" + "-" * 78)
    lines.append(f"{'type':>22} {'best_val_f1':>12} {'test_f1':>9} {'trials':>7}")
    lines.append("-" * 78)
    for vt, info in results["types"].items():
        best = info.get("best") or {}
        test = info.get("test") or {}
        bf1 = best.get("val_f1")
        tf1 = test.get("f1")
        lines.append(
            f"{vt:>22} "
            f"{(f'{bf1:.3f}' if isinstance(bf1, (int, float)) else '-'):>12} "
            f"{(f'{tf1:.3f}' if isinstance(tf1, (int, float)) else '-'):>9} "
            f"{len(info.get('trials') or []):>7}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Heavy/optional imports happen AFTER argparse so --help always works.
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna is not installed. Run:  pip install optuna")
        return 1

    import torch
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate requested types.
    bad = [t for t in args.types if t not in VULN_TYPES]
    if bad:
        print(f"Unknown vulnerability type(s): {bad}. Must be from: {VULN_TYPES}")
        return 1

    results = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "config": {
            "trials": args.trials,
            "hours_per_type": args.hours_per_type,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_length": args.max_length,
            "model": STAGE1_MODEL,
        },
        "types": {},
    }

    print(f"Device: {device}  |  types: {args.types}")
    print(f"Budget: {args.hours_per_type} h/type, up to {args.trials} trials/type")
    print(f"Results -> {args.results_file}")
    if args.storage:
        print(f"Optuna storage (resumable) -> {args.storage}")

    try:
        for vuln_type in args.types:
            run_study_for_type(vuln_type, args, device, results, optuna)
    except KeyboardInterrupt:
        print("\nInterrupted by user — saving partial results...", flush=True)

    results["finished"] = datetime.now().isoformat(timespec="seconds")
    _save_results(args.results_file, results)

    summary = build_summary(results)
    print("\n" + summary)
    summary_path = os.path.join(RESULTS_DIR, "stage1_summary.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved full results -> {args.results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
