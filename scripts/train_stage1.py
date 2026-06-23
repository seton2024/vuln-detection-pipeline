"""
Stage 1 training with Bayesian hyperparameter search (Optuna).

Trains EITHER backend (choose with --model):
  - graphcodebert : fine-tune GraphCodeBERT end-to-end
  - cnn_bilstm    : train a CNN-BiLSTM head on FROZEN GraphCodeBERT embeddings


Outputs:
  - best model per type  -> models/cnn/cnn_bilstm_{type}.pt  OR  models/graphcodebert/graphcodebert_{type}.pt
  - best hyperparameters -> results/stage1/best_hyperparams_{type}.json
  - full experiment log  -> results/stage1/stage1_experiments.json   (crash-safe)
  - readable summary     -> results/stage1/stage1_summary.txt

USAGE
  python scripts/train_stage1.py                              # cnn_bilstm, all types
  python scripts/train_stage1.py --model graphcodebert        # the other backend
  python scripts/train_stage1.py --types sql --hours-per-type 6
  python scripts/train_stage1.py --model cnn_bilstm --types sql xss

Requires processed data (scripts/setup_data.py) and optuna (pip install optuna).
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    STAGE1_MODEL, MODELS_DIR, RESULTS_DIR, VULN_TYPES,
    STAGE1_HIDDEN_DIM, STAGE1_CNN_FILTERS, STAGE1_DROPOUT,
    MODEL_SUBDIR, STAGE1_RESULTS_DIR,
)



# Arguments


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 hyperparameter search (GraphCodeBERT or CNN-BiLSTM).")
    p.add_argument("--model", choices=["graphcodebert", "cnn_bilstm"], default="cnn_bilstm",
                   help="Which Stage 1 backend to train (default: cnn_bilstm).")
    p.add_argument("--types", nargs="+", default=VULN_TYPES,
                   help="Which vulnerability types to train (default: all 7).")
    p.add_argument("--hours-per-type", type=float, default=None,
                   help="Time budget per type in hours (default: 1 for cnn_bilstm, 3 for graphcodebert).")
    p.add_argument("--max-train-samples", type=int, default=6000,
                   help="Cap on training windows per type (class-balanced subsample).")
    p.add_argument("--max-eval-samples", type=int, default=2000,
                   help="Cap on validation windows used to score trials.")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max tokens fed to the encoder (lower = faster).")
    p.add_argument("--cnn-epochs", type=int, default=8,
                   help="Epochs per cnn_bilstm trial (epochs are not searched for that model).")
    p.add_argument("--beta", type=float, default=1.0,
                   help="F-beta to optimize. 1.0 = balanced F1 (recommended with --balance). "
                        ">1 favors recall over precision. Default: 1.0.")
    p.add_argument("--trials", type=int, default=None,
                   help="Max Optuna trials per type. None = unlimited (controlled by --hours-per-type).")
    p.add_argument("--balance", action="store_true", default=True,
                   help="Force 50/50 vulnerable/safe training split (default: True). "
                        "Use --no-balance to preserve the natural VUDENC ratio (~10-18%% vuln).")
    p.add_argument("--no-balance", dest="balance", action="store_false")
    p.add_argument("--pos-weight-max", type=float, default=1.0,
                   help="Cap on BCE pos_weight. 1.0 = no class weighting (best with --balance). "
                        "Raise only if using --no-balance. Default: 1.0.")
    p.add_argument("--results-file", default=os.path.join(STAGE1_RESULTS_DIR, "stage1_experiments.json"))
    p.add_argument("--storage", default=os.path.join(STAGE1_RESULTS_DIR, "optuna_stage1.db"),
                   help="SQLite file for Optuna (enables resume). Empty string = in-memory.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



# Shared helpers


def _subsample(records, max_n, seed, balance=False):
    """Return at most max_n records.

    balance=True forces 50/50 vulnerable/safe sampling regardless of the natural
    ratio. This is the main lever against the VUDENC imbalance (~10-18% vulnerable)
    which otherwise drives pos_weight to 6-9x and causes the model to flag
    everything as vulnerable.
    """
    import random
    rng = random.Random(seed)
    pos = [r for r in records if r.label == 1]
    neg = [r for r in records if r.label == 0]
    if not pos or not neg:
        pool = list(records)
        return rng.sample(pool, min(max_n, len(pool))) if max_n else pool

    if balance:
        # Equal classes: each side capped at min(available, max_n//2).
        half = (max_n // 2) if max_n else min(len(pos), len(neg))
        n_pos = min(len(pos), half)
        n_neg = min(len(neg), half)
    else:
        if max_n is None or len(records) <= max_n:
            return list(records)
        ratio = len(pos) / len(records)
        n_pos = min(len(pos), max(1, round(max_n * ratio)))
        n_neg = min(len(neg), max_n - n_pos)

    sample = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(sample)
    return sample


def _fbeta(precision, recall, beta):
    b2 = beta * beta
    denom = b2 * precision + recall
    return ((1 + b2) * precision * recall / denom) if denom > 0 else 0.0


def _eval_probs(probs, labels, beta=2.0):
    """
    Maximize F-beta (beta>1 favors recall, i.e. min FN s while keeping some precision) and return all metrics
    AT that threshold: f2, f1, precision, recall, accuracy, threshold, n.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    best = None
    for i in range(5, 96, 5):                      # thresholds 0.05 .. 0.95
        th = i / 100.0
        preds = [1 if p >= th else 0 for p in probs]
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        fb = _fbeta(prec, rec, beta)
        if best is None or fb > best["f2"]:
            best = {
                "threshold": th, "f2": float(fb),
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "precision": float(prec), "recall": float(rec),
                "accuracy": float(accuracy_score(labels, preds)),
                "n": len(labels),
            }
    return best or {"threshold": 0.5, "f2": 0.0, "f1": 0.0, "precision": 0.0,
                    "recall": 0.0, "accuracy": 0.0, "n": len(labels)}


def _save_results(path, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _cleanup(*objs):
    import torch
    for _ in objs:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# GraphCodeBERT backend (fine-tune end-to-end) old model

def _gcb_suggest(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        "epochs": trial.suggest_int("epochs", 1, 3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }


def _gcb_train(train_recs, hp, device, max_length):
    import random
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL, num_labels=2).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    texts = [r.code for r in train_recs]
    labels = [int(r.label) for r in train_recs]
    bs = hp["batch_size"]
    for epoch in range(hp["epochs"]):
        order = list(range(len(texts)))
        random.Random(epoch).shuffle(order)
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            batch_labels = torch.tensor([labels[k] for k in idx]).to(device)
            enc = tokenizer([texts[k] for k in idx], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            optimizer.zero_grad()
            out = model(**enc, labels=batch_labels)
            out.loss.backward()
            optimizer.step()
    return tokenizer, model


def _gcb_probs(tokenizer, model, recs, device, max_length, batch_size=32):
    """Return (probs_of_class_1, labels) so the caller can threshold-select."""
    import torch
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for start in range(0, len(recs), batch_size):
            batch = recs[start:start + batch_size]
            enc = tokenizer([r.code for r in batch], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
            probs.extend(p)
            labels.extend(int(r.label) for r in batch)
    return probs, labels



# CNN-BiLSTM backend (frozen GraphCodeBERT + trained head), prefered model

def _cnn_suggest(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "cnn_filters": trial.suggest_categorical("cnn_filters", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2]),
    }


# One study per type


def run_study_for_type(vuln_type, args, device, results, optuna):
    from data.loader import load_vudenc

    print(f"\n{'=' * 70}\n[{vuln_type}] starting ({args.model})\n{'=' * 70}", flush=True)

    try:
        train, val, test = load_vudenc(vuln_type)
    except FileNotFoundError as e:
        print(f"[{vuln_type}] SKIP: {e}", flush=True)
        results["types"][vuln_type] = {"status": "skipped: no processed data"}
        _save_results(args.results_file, results)
        return

    train = _subsample(train, args.max_train_samples, seed=args.seed, balance=args.balance)
    # Val/test keep the natural ratio so evaluation metrics reflect real-world performance.
    val_sub = _subsample(val, args.max_eval_samples, seed=args.seed, balance=False)
    test_sub = _subsample(test, max(args.max_eval_samples * 3, args.max_eval_samples), seed=7, balance=False)
    vuln_pct = 100.0 * sum(1 for r in train if r.label == 1) / len(train) if train else 0
    print(f"[{vuln_type}] train balance: {vuln_pct:.1f}% vulnerable "
          f"({'balanced 50/50' if args.balance else 'natural ratio'})", flush=True)

    if set(r.label for r in train) != {0, 1}:
        print(f"[{vuln_type}] SKIP: training subsample is single-class.", flush=True)
        results["types"][vuln_type] = {"status": "skipped: single-class train"}
        _save_results(args.results_file, results)
        return

    model_path = os.path.join(MODELS_DIR, MODEL_SUBDIR[args.model], f"{args.model}_{vuln_type}.pt")
    state = {"best_score": -1.0, "best_threshold": 0.5}
    results["types"][vuln_type] = {
        "status": "running", "model": args.model,
        "n_train": len(train), "n_val": len(val_sub), "n_test": len(test_sub),
        "trials": [], "model_path": model_path, "best": None, "test": None,
    }
    type_results = results["types"][vuln_type]

    # --- precompute frozen embeddings once (CNN-BiLSTM only) ---
    cnn = None
    train_emb = val_emb = train_lab = val_lab = None
    if args.model == "cnn_bilstm":
        from pipeline import stage1_cnn_bilstm as cnn
        print(f"[{vuln_type}] embedding train/val with frozen GraphCodeBERT (one-time)...", flush=True)
        train_emb, train_lab = cnn.embed_records(train, device, args.max_length, label="train")
        val_emb, val_lab = cnn.embed_records(val_sub, device, args.max_length, label="val")

        # Short initial run to confirm the model works before the real study.
        try:
            hp0 = {"learning_rate": 1e-3, "hidden_dim": STAGE1_HIDDEN_DIM,
                   "cnn_filters": STAGE1_CNN_FILTERS, "dropout": STAGE1_DROPOUT, "lstm_layers": 2}
            probe = cnn.build_classifier(hp0, device)
            cnn.fit_classifier(probe, train_emb[:64], train_lab[:64], hp0, epochs=1, batch_size=16, device=device)
            print(f"[{vuln_type}] sanity check OK — CNN-BiLSTM trains without error.", flush=True)
            del probe
        except Exception as e:
            print(f"[{vuln_type}] SANITY CHECK FAILED: {e}", flush=True)
            type_results["status"] = f"error: sanity check failed ({e})"
            _save_results(args.results_file, results)
            return

    # --- Optuna study ---
    pruner = optuna.pruners.MedianPruner() if args.model == "cnn_bilstm" else optuna.pruners.NopPruner()
    storage = f"sqlite:///{args.storage}" if args.storage else None
    # Study name encodes the objective so changing beta starts a FRESH search
    # rather than mixing with trials scored under a different metric.
    study = optuna.create_study(
        direction="maximize",
        study_name=f"stage1_{args.model}_{vuln_type}_F{args.beta:g}",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    try:
        if study.best_value is not None:
            state["best_score"] = study.best_value
    except ValueError:
        pass  # no completed trials yet

    empty = {"f2": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
             "accuracy": 0.0, "threshold": 0.5, "n": 0}

    def objective(trial):
        t0 = time.time()
        pruned = False
        if args.model == "cnn_bilstm":
            hp = _cnn_suggest(trial)
            try:
                classifier = cnn.build_classifier(hp, device)
                cnn.fit_classifier(
                    classifier, train_emb, train_lab, hp,
                    epochs=args.cnn_epochs, batch_size=hp["batch_size"], device=device,
                    val_emb=val_emb, val_labels=val_lab, trial=trial, optuna_mod=optuna,
                    beta=args.beta, verbose=True, label=f"trial {trial.number}",
                    pos_weight_max=args.pos_weight_max,
                )
                probs = cnn.run_classifier(classifier, val_emb, device)
                metrics = _eval_probs(probs, val_lab, args.beta)
                score = metrics["f2"]
                if score > state["best_score"]:
                    state["best_score"] = score
                    state["best_threshold"] = metrics["threshold"]
                    cnn.save_checkpoint(model_path, classifier, hp, args.max_length,
                                        threshold=metrics["threshold"])
                _cleanup(classifier)
            except optuna.TrialPruned:
                pruned, metrics, score = True, empty, 0.0
            except Exception as e:
                print(f"[{vuln_type}] trial {trial.number} FAILED: {e}", flush=True)
                metrics, score = empty, 0.0
        else:
            hp = _gcb_suggest(trial)
            try:
                tokenizer, model = _gcb_train(train, hp, device, args.max_length)
                probs, labels = _gcb_probs(tokenizer, model, val_sub, device, args.max_length)
                metrics = _eval_probs(probs, labels, args.beta)
                score = metrics["f2"]
                if score > state["best_score"]:
                    state["best_score"] = score
                    state["best_threshold"] = metrics["threshold"]
                    import torch
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                _cleanup(model, tokenizer)
            except Exception as e:
                print(f"[{vuln_type}] trial {trial.number} FAILED: {e}", flush=True)
                metrics, score = empty, 0.0

        secs = round(time.time() - t0, 1)
        type_results["trials"].append({
            "number": trial.number, "params": hp,
            "val_f2": round(metrics["f2"], 4),
            "val_f1": round(metrics["f1"], 4),
            "val_precision": round(metrics["precision"], 4),
            "val_recall": round(metrics["recall"], 4),
            "val_accuracy": round(metrics["accuracy"], 4),
            "threshold": metrics["threshold"],
            "seconds": secs, "pruned": pruned,
        })
        _save_results(args.results_file, results)
        tag = " PRUNED" if pruned else (" <-- BEST" if score >= state["best_score"] and not pruned else "")
        print(f"[{vuln_type}] trial {trial.number}: F{args.beta:g}={score:.3f} "
              f"rec={metrics['recall']:.3f} prec={metrics['precision']:.3f} "
              f"acc={metrics['accuracy']:.3f} th={metrics['threshold']}  "
              f"{_fmt_params(hp)}  ({secs:.0f}s){tag}", flush=True)

        if pruned:
            raise optuna.TrialPruned()
        return score

    hours = args.hours_per_type
    if hours is None:
        hours = 1.0 if args.model == "cnn_bilstm" else 3.0
    timeout = hours * 3600 if hours > 0 else None
    study.optimize(objective, n_trials=args.trials, timeout=timeout)

    # free embeddings before the test pass to reduce peak memory
    train_emb = val_emb = None
    _cleanup()

    # --- best hyperparameters ---
    best_params = None
    best_value = None
    try:
        best_params = dict(study.best_trial.params)
        best_value = float(study.best_value)
    except (ValueError, AttributeError):
        pass
    type_results["best"] = {"params": best_params,
                            "val_f2": round(best_value, 4) if best_value is not None else None,
                            "threshold": state["best_threshold"]}
    bh_path = os.path.join(STAGE1_RESULTS_DIR, f"best_hyperparams_{vuln_type}.json")
    os.makedirs(STAGE1_RESULTS_DIR, exist_ok=True)
    with open(bh_path, "w", encoding="utf-8") as f:
        json.dump({"vuln_type": vuln_type, "model": args.model,
                   "objective": f"F{args.beta:g}", "best_val_f2": best_value,
                   "best_threshold": state["best_threshold"],
                   "best_params": best_params, "n_trials": len(study.trials)}, f, indent=2)
    print(f"[{vuln_type}] best params -> {best_params}  "
          f"(val_F{args.beta:g}={best_value}, threshold={state['best_threshold']})")
    print(f"[{vuln_type}] saved -> {bh_path}")

    # --- final test evaluation at the threshold chosen on validation ---
    test_metrics = _evaluate_best_on_test(args, vuln_type, model_path, test_sub, device, cnn,
                                          state["best_threshold"])
    type_results["status"] = "done"
    type_results["test"] = test_metrics
    _save_results(args.results_file, results)

    if test_metrics:
        print(f"[{vuln_type}] DONE  best_val_F{args.beta:g}={state['best_score']:.3f}  "
              f"TEST: F{args.beta:g}={test_metrics['f2']:.3f} recall={test_metrics['recall']:.3f} "
              f"prec={test_metrics['precision']:.3f} acc={test_metrics['accuracy']:.3f} "
              f"(threshold={test_metrics['threshold']})", flush=True)
    else:
        print(f"[{vuln_type}] DONE  best_val_F{args.beta:g}={state['best_score']:.3f}  (no test metrics)", flush=True)


def _metrics_at(probs, labels, threshold, beta=2.0):
    """All metrics at a FIXED threshold (the one chosen on validation)."""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    preds = [1 if p >= threshold else 0 for p in probs]
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    return {"threshold": threshold, "f2": _fbeta(prec, rec, beta),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(prec), "recall": float(rec),
            "accuracy": float(accuracy_score(labels, preds)), "n": len(labels)}


def _evaluate_best_on_test(args, vuln_type, model_path, test_recs, device, cnn, threshold):
    if not os.path.exists(model_path) or not test_recs:
        return None
    try:
        import torch
        if args.model == "cnn_bilstm":
            ckpt = torch.load(model_path, map_location=device)
            classifier = cnn.build_classifier(ckpt["hp"], device)
            classifier.load_state_dict(ckpt["state_dict"])
            test_emb, test_lab = cnn.embed_records(test_recs, device, args.max_length, label="test")
            probs = cnn.run_classifier(classifier, test_emb, device)
            return _metrics_at(probs, test_lab, threshold, args.beta)
        else:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL, num_labels=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            probs, labels = _gcb_probs(tokenizer, model, test_recs, device, args.max_length)
            return _metrics_at(probs, labels, threshold, args.beta)
    except Exception as e:
        print(f"[{vuln_type}] test evaluation failed: {e}", flush=True)
        return None



# Summary


def _fmt_params(p):
    if not p:
        return "-"
    parts = []
    for k, v in p.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2e}" if v < 0.01 else f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"


def build_summary(results) -> str:
    beta = results.get("config", {}).get("beta", 2.0)
    fb = f"F{beta:g}"
    lines = ["=" * 96,
             f"STAGE 1 SEARCH SUMMARY  (model: {results.get('model','?')}, objective: {fb} — recall-favoring)",
             "=" * 96]
    for vt, info in results["types"].items():
        lines.append(f"\n### {vt}  —  {info.get('status', '?')}")
        for t in (info.get("trials") or []):
            tag = " PRUNED" if t.get("pruned") else ""
            score = t.get("val_f2", t.get("val_f1", 0.0))
            lines.append(f"  trial {t['number']:>3}  {fb}={score:.3f} "
                         f"rec={t.get('val_recall', 0):.3f} prec={t.get('val_precision', 0):.3f} "
                         f"acc={t.get('val_accuracy', 0):.3f} th={t.get('threshold', '-')}  "
                         f"{_fmt_params(t['params'])}  ({t['seconds']:.0f}s){tag}")
        best = info.get("best") or {}
        if best.get("params"):
            lines.append(f"  BEST: val_{fb}={best.get('val_f2')}  threshold={best.get('threshold')}  "
                         f"{_fmt_params(best['params'])}")
        test = info.get("test")
        if test:
            lines.append(f"  TEST @th={test.get('threshold')}: {fb}={test['f2']:.3f}  recall={test['recall']:.3f}  "
                         f"precision={test['precision']:.3f}  f1={test['f1']:.3f}  acc={test['accuracy']:.3f}  (n={test['n']})")

    lines.append("\n" + "-" * 96)
    lines.append(f"{'type':>22} {'val_'+fb:>9} {'test_'+fb:>9} {'test_rec':>9} {'test_prec':>10} {'test_acc':>9} {'trials':>7}")
    lines.append("-" * 96)
    for vt, info in results["types"].items():
        best = info.get("best") or {}
        test = info.get("test") or {}
        lines.append(f"{vt:>22} {_fmt(best.get('val_f2')):>9} {_fmt(test.get('f2')):>9} "
                     f"{_fmt(test.get('recall')):>9} {_fmt(test.get('precision')):>10} "
                     f"{_fmt(test.get('accuracy')):>9} {len(info.get('trials') or []):>7}")
    lines.append("=" * 96)
    return "\n".join(lines)



# Main


def main() -> int:
    args = parse_args()

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

    bad = [t for t in args.types if t not in VULN_TYPES]
    if bad:
        print(f"Unknown vulnerability type(s): {bad}. Must be from: {VULN_TYPES}")
        return 1

    results = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "device": device, "model": args.model,
        "config": {"max_trials": args.trials, "hours_per_type": args.hours_per_type,
                   "max_train_samples": args.max_train_samples,
                   "max_eval_samples": args.max_eval_samples,
                   "max_length": args.max_length, "cnn_epochs": args.cnn_epochs,
                   "beta": args.beta, "balance": args.balance,
                   "pos_weight_max": args.pos_weight_max},
        "types": {},
    }

    print(f"Model: {args.model}  |  device: {device}  |  types: {args.types}")
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
    summary_path = os.path.join(STAGE1_RESULTS_DIR, "stage1_summary.txt")
    os.makedirs(STAGE1_RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved full results -> {args.results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
