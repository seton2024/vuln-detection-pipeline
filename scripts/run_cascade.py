"""
Run the FULL cascade (Stage 0 -> Stage 1 -> 1.5 -> Stage 2 -> Stage 3) on a file.

Command-line entry point referenced in the README. It windows a source file,
runs every stage per the escalation thresholds in config.py, and prints a
per-window report plus a summary.

USAGE
    python scripts/run_cascade.py --vuln_type sql --file input_data/flask_app.py
    python scripts/run_cascade.py --all --file input_data/flask_app.py
    python scripts/run_cascade.py --all --file input_data/flask_app.py --flagged-only

Stage 2 (Ollama) and Stage 3 (Claude) honour the env toggles in config.py:
    OLLAMA_MOCK=0      -> actually call Ollama (otherwise mock scores)
    OLLAMA_MODEL=...   -> pick a model you have pulled (`ollama list`)
    STAGE3_ENABLED=1   -> enable the Claude adjudicator (needs ANTHROPIC_API_KEY)
"""

import argparse
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.windowing import extract_windows
from pipeline.runner import run_pipeline


def _fmt(value, nd=3):
    return "-" if value is None else (f"{value:.{nd}f}" if isinstance(value, float) else str(value))


def _run_one(vuln_type, code, file_path, window_size, stride, flagged_only):
    records = extract_windows(code, file_path, vuln_type, window_size=window_size, stride=stride)
    if not records:
        print(f"[{vuln_type}] no windows (empty file?)")
        return

    results = run_pipeline(records)

    print(f"\n=== {vuln_type}  ({len(results)} consolidated windows) ===")
    print(f"{'#':>3}  {'bandit':>6}  {'stage1':>7}  {'stage2':>7}  {'stage3':>14}")
    shown = 0
    flagged = 0
    for i, r in enumerate(results):
        escalated = r.stage1_score is not None and r.stage1_score > config.STAGE1_ESCALATION_THRESHOLD
        if escalated:
            flagged += 1
        if flagged_only and not escalated:
            continue
        print(f"{i:>3}  {str(r.bandit_flag):>6}  {_fmt(r.stage1_score):>7}  "
              f"{_fmt(r.stage2_score):>7}  {_fmt(r.stage3_verdict):>14}")
        shown += 1

    print(f"summary: {flagged}/{len(results)} windows escalated past Stage 1 "
          f"(threshold {config.STAGE1_ESCALATION_THRESHOLD})")
    if flagged_only and shown == 0:
        print("  (nothing escalated - try without --flagged-only, or train Stage 1 models)")


def main() -> int:
    p = argparse.ArgumentParser(description="Run the full vulnerability-detection cascade on a file.")
    p.add_argument("--file", required=True, help="Path to the .py file to scan.")
    p.add_argument("--vuln_type", choices=config.VULN_TYPES, help="Single vulnerability type to check.")
    p.add_argument("--all", action="store_true", help="Check all seven vulnerability types.")
    p.add_argument("--window-size", type=int, default=10, help="Window size in lines (default 10).")
    p.add_argument("--stride", type=int, default=5, help="Window stride in lines (default 5).")
    p.add_argument("--flagged-only", action="store_true", help="Only print windows that escalated past Stage 1.")
    args = p.parse_args()

    if not args.all and not args.vuln_type:
        p.error("give either --vuln_type <type> or --all")

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return 1
    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()

    print(f"Scanning {args.file}")
    print(f"config: OLLAMA_MOCK={config.OLLAMA_MOCK}  OLLAMA_MODEL={config.OLLAMA_MODEL}  "
          f"STAGE3_ENABLED={config.STAGE3_ENABLED}  backend={config.STAGE1_BACKEND}")

    types = config.VULN_TYPES if args.all else [args.vuln_type]
    for vt in types:
        _run_one(vt, code, args.file, args.window_size, args.stride, args.flagged_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
