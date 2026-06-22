"""
Full end-to-end pipeline demo runner.

Loads a Python file from input_data/, runs it through ALL stages
(Stage 0 Bandit -> Stage 1 model -> Stage 1.5 consolidation -> Stage 2 Llama ->
Stage 3 Claude), and produces TWO outputs:

  1. A per-stage text report, printed to the terminal AND saved as
     results/pipeline_demo_<file>_<timestamp>.txt
  2. A per-line PNG saved to results/m_demo/, coloured by Stage 1 score:
       green -> yellow -> red   = safe -> vulnerable
       grey                     = comment lines
       black                    = code escalated to the next stage (undetermined)
       red dot                  = code the pipeline finally judged vulnerable

Separate from scripts/demo.py (the per-line visualizer), which is left untouched.

USAGE
    python scripts/run_pipeline_demo.py --list
    python scripts/run_pipeline_demo.py --file 0
    python scripts/run_pipeline_demo.py --file 0 --backend graphcodebert --window 10 --stride 1
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from pipeline import stage0_bandit, stage1_cnn_bilstm, stage1_graphcodebert, stage2_llama, stage3_claude
from pipeline.stage15_consolidator import consolidate
from data.windowing import extract_all_vuln_types
from pipeline.contract import WindowRecord
from config import (STAGE1_ESCALATION_THRESHOLD, STAGE2_SAFE_THRESHOLD,
                    STAGE2_ESCALATION_THRESHOLD, STAGE3_ENABLED, STAGE1_BACKEND,
                    VULN_TYPES, RESULTS_DIR, DEMO_RESULTS_DIR)

INPUT_DIR = PROJECT_ROOT / "input_data"


# ---------------------------------------------------------------------------
# Per-record verdict
# ---------------------------------------------------------------------------

def classify(record: WindowRecord) -> str:
    """Return 'safe', 'vulnerable', or 'ambiguous' for a record's current state."""
    label = record.final_label()
    if label == "not_vulnerable":
        return "safe"
    if label == "vulnerable":
        return "vulnerable"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Per-stage, per-type counts  (each row: safe, vulnerable, ambiguous)
# ---------------------------------------------------------------------------

def stage0_counts(records: list) -> dict:
    """Per type for Bandit: vulnerable = flagged, safe = not flagged (no ambiguous)."""
    out = {}
    for vt in VULN_TYPES:
        recs = [r for r in records if r.vulnerability_type == vt]
        vuln = sum(1 for r in recs if r.bandit_flag)
        out[vt] = (len(recs) - vuln, vuln, 0)
    return out


def stage1_counts(records: list) -> dict:
    """Per type for Stage 1: safe = score<=thr, ambiguous = score>thr (escalated)."""
    out = {}
    for vt in VULN_TYPES:
        recs = [r for r in records if r.vulnerability_type == vt and r.stage1_score is not None]
        safe = sum(1 for r in recs if r.stage1_score <= STAGE1_ESCALATION_THRESHOLD)
        amb = sum(1 for r in recs if r.stage1_score > STAGE1_ESCALATION_THRESHOLD)
        out[vt] = (safe, 0, amb)
    return out


def stage2_counts(records: list) -> dict:
    """Per type for Stage 2: safe<thr, vulnerable>esc, ambiguous in between."""
    out = {}
    for vt in VULN_TYPES:
        recs = [r for r in records if r.vulnerability_type == vt and r.stage2_score is not None]
        safe = sum(1 for r in recs if r.stage2_score < STAGE2_SAFE_THRESHOLD)
        vuln = sum(1 for r in recs if r.stage2_score > STAGE2_ESCALATION_THRESHOLD)
        amb = sum(1 for r in recs if STAGE2_SAFE_THRESHOLD <= r.stage2_score <= STAGE2_ESCALATION_THRESHOLD)
        out[vt] = (safe, vuln, amb)
    return out


def stage3_counts(records: list) -> dict:
    """Per type for Stage 3: safe / vulnerable by Claude's verdict."""
    out = {}
    for vt in VULN_TYPES:
        recs = [r for r in records if r.vulnerability_type == vt and r.stage3_verdict is not None]
        safe = sum(1 for r in recs if r.stage3_verdict == "not_vulnerable")
        vuln = sum(1 for r in recs if r.stage3_verdict == "vulnerable")
        out[vt] = (safe, vuln, 0)
    return out


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def _counts_table(counts: dict) -> list:
    """Format a per-type counts dict into report lines (only types with any window)."""
    lines = [f"  {'Vuln Type':<22} {'Safe':>6} {'Vulnerable':>11} {'Ambiguous':>10}",
             "  " + "-" * 52]
    for vt, (safe, vuln, amb) in counts.items():
        if safe + vuln + amb == 0:
            continue
        lines.append(f"  {vt:<22} {safe:>6} {vuln:>11} {amb:>10}")
    if len(lines) == 2:
        lines.append("  (no windows at this stage)")
    return lines


def _example(record, label: str) -> list:
    """Format one example code snippet (the window's code), indented."""
    lines = [f"  {label}:"]
    if record is None:
        lines.append("    (none)")
        return lines
    for ln in record.code.split("\n"):
        lines.append(f"    | {ln}")
    return lines


def _max_by(records, attr):
    """Return the record with the highest non-None attribute value, or None."""
    pool = [r for r in records if getattr(r, attr) is not None]
    return max(pool, key=lambda r: getattr(r, attr), default=None)


def build_report(filename, backend, records, consolidated, window, stride) -> list:
    """Build the full per-stage text report as a list of lines."""
    R = []
    R.append("=" * 66)
    R.append(f" PIPELINE REPORT — {filename}")
    R.append(f" backend: {backend}    Stage 3: {'ENABLED' if STAGE3_ENABLED else 'disabled'}"
             f"    {datetime.now():%Y-%m-%d %H:%M:%S}")
    R.append("=" * 66)

    # --- Stage 0 ---
    R.append("\nSTAGE 0 — Bandit (static analysis)")
    R += _counts_table(stage0_counts(records))
    R += _example(next((r for r in records if r.bandit_flag), None),
                  "Example Bandit-flagged snippet")

    # --- Stage 1 ---
    R.append(f"\nSTAGE 1 — {backend}")
    R.append(f"  Total windows scanned: {len(records)}")
    R += _counts_table(stage1_counts(records))
    R += _example(_max_by(records, "stage1_score"), "Example highest-scoring window")

    # --- Stage 1.5 ---
    R.append("\nSTAGE 1.5 — Consolidation")
    flagged = [r for r in records if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD]
    cons_flagged = [r for r in consolidated if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD]
    R.append(f"  Flagged windows received   : {len(flagged)}")
    R.append(f"  Consolidated windows produced: {len(cons_flagged)}")
    R += _example(cons_flagged[len(cons_flagged) // 2] if cons_flagged else None,
                  "Example consolidated window (sent to Stage 2)")

    # --- Stage 2 ---
    R.append("\nSTAGE 2 — Llama")
    R += _counts_table(stage2_counts(consolidated))
    R += _example(_max_by(consolidated, "stage2_score"), "Example Stage 2 section")

    # --- Stage 3 ---
    R.append("\nSTAGE 3 — Claude")
    received = [r for r in consolidated if r.stage2_score is not None
                and STAGE2_SAFE_THRESHOLD <= r.stage2_score <= STAGE2_ESCALATION_THRESHOLD]
    R.append(f"  Received snippets (ambiguous from Stage 2): {len(received)}")
    if STAGE3_ENABLED:
        R += _counts_table(stage3_counts(consolidated))
        R += _example(next((r for r in consolidated if r.stage3_verdict is not None), None),
                      "Example Stage 3 snippet")
    else:
        R.append("  (Stage 3 disabled — set STAGE3_ENABLED=1 to run Claude)")
        R += _example(received[0] if received else None,
                      "Example snippet that would go to Claude")

    R.append("\n" + "=" * 66)
    return R


# ---------------------------------------------------------------------------
# Per-line PNG
# ---------------------------------------------------------------------------

def _per_line_status(code: str, records: list, window: int, stride: int) -> list:
    """For every source line, aggregate the worst verdict and the max Stage 1 score
    across the windows covering it (window j starts at line j*stride)."""
    lines = code.split("\n")
    n = len(lines)
    score = [None] * n
    verdict = [None] * n
    rank = {"safe": 0, "ambiguous": 1, "vulnerable": 2}
    groups = defaultdict(list)
    for r in records:
        groups[r.vulnerability_type].append(r)
    for recs in groups.values():
        if len(recs) == 1 and window > n:
            spans = [(0, n)]
        else:
            spans = [(j * stride, min(j * stride + window, n)) for j in range(len(recs))]
        for (a, b), r in zip(spans, recs):
            v = classify(r)
            for i in range(a, b):
                if 0 <= i < n:
                    if r.stage1_score is not None and (score[i] is None or r.stage1_score > score[i]):
                        score[i] = r.stage1_score
                    if verdict[i] is None or rank[v] > rank[verdict[i]]:
                        verdict[i] = v
    return [{"text": t, "comment": t.strip().startswith("#"),
             "score": score[i], "verdict": verdict[i]} for i, t in enumerate(lines)]


def render_png(code, records, window, stride, out_path: Path, title: str):
    """Render the coloured per-line PNG and save it. Returns the path, or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        print("[demo] matplotlib not installed — skipping PNG.")
        return None

    cmap = colormaps["RdYlGn_r"]          # 0.0 -> green (safe), 1.0 -> red (vulnerable)
    rows = _per_line_status(code, records, window, stride)
    n = len(rows)

    fig, ax = plt.subplots(figsize=(15, max(2.6, 0.23 * n + 1.8)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, row in enumerate(rows):
        y = n - 1 - i
        text = row["text"] if row["text"].strip() else " "
        if len(text) > 200:
            text = text[:197] + "..."
        if row["comment"]:
            colour = "#9aa0a6"                       # grey  = comment
        elif row["verdict"] == "vulnerable":
            colour = "#c0392b"                       # red   = finally vulnerable
        elif row["verdict"] == "ambiguous":
            colour = "#000000"                       # black = escalated to next stage
        elif row["score"] is not None:
            colour = cmap(row["score"])              # gradient green->red by score
        else:
            colour = "#cfd2d6"                       # uncovered / blank
        if row["verdict"] == "vulnerable":
            ax.text(0.004, y, "●", va="center", ha="left", fontsize=10, color="#ff0000")
        ax.text(0.03, y, text, va="center", ha="left", fontsize=9, family="monospace", color=colour)
        if row["score"] is not None and not row["comment"]:
            ax.text(0.995, y, f"{row['score']:.2f}", va="center", ha="right",
                    fontsize=8, family="monospace", color=colour if not isinstance(colour, str) else colour)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.4, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, loc="left")

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, fraction=0.03)
    cbar.set_label("Stage 1 score: safe (green) → vulnerable (red)")

    fig.text(0.5, 0.01,
             "grey = comment      black = escalated to next stage (undetermined)      "
             "● red dot = finally vulnerable",
             ha="center", fontsize=9, color="#444")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Input handling + main
# ---------------------------------------------------------------------------

def list_input_files() -> list:
    """Return the alphabetically-sorted .py files in input_data/ (raises if missing)."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(str(INPUT_DIR))
    return sorted(INPUT_DIR.glob("*.py"))


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Full pipeline demo over a file from input_data/.")
    p.add_argument("--list", action="store_true", help="List .py files in input_data/ and exit.")
    p.add_argument("--file", type=int, default=None, help="Index of the file to scan (see --list).")
    p.add_argument("--backend", choices=["cnn_bilstm", "graphcodebert"], default=STAGE1_BACKEND,
                   help=f"Stage 1 backend (default: config STAGE1_BACKEND = {STAGE1_BACKEND}).")
    p.add_argument("--window", type=int, default=10, help="Sliding window size in lines (default 10).")
    p.add_argument("--stride", type=int, default=1, help="Stride between windows in lines (default 1).")
    return p.parse_args()


def main() -> int:
    """Run the full pipeline demo. Returns a process exit code."""
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        files = list_input_files()
    except FileNotFoundError:
        print(f"ERROR: input directory '{INPUT_DIR}' does not exist.")
        print("Create it and drop some .py files in there, then re-run with --list.")
        return 1

    if args.list:
        if not files:
            print(f"No .py files found in {INPUT_DIR}.")
            return 0
        print(f"Python files in {INPUT_DIR}:")
        for i, f in enumerate(files):
            print(f"  [{i}] {f.name}")
        return 0

    if args.file is None:
        print("ERROR: --file INDEX is required (or use --list to see the choices).")
        return 1
    if not files:
        print(f"No .py files found in {INPUT_DIR}.")
        return 1
    if args.file < 0 or args.file >= len(files):
        print(f"--file {args.file} is out of range (0..{len(files) - 1}). Use --list.")
        return 1

    target = files[args.file]
    backend = args.backend
    stage1 = stage1_cnn_bilstm if backend == "cnn_bilstm" else stage1_graphcodebert

    code = target.read_text(encoding="utf-8", errors="replace")
    records = extract_all_vuln_types(code, target.name, args.window, args.stride)
    print(f"Scanning {target.name}: {code.count(chr(10)) + 1} lines → "
          f"{len(records)} windows (window={args.window}, stride={args.stride}). Running stages...",
          flush=True)
    if not records:
        print("File too short for the window size — nothing to scan.")
        return 0

    # Run the cascade (progress to terminal; the report is built afterwards).
    print("  Stage 0 (Bandit)...", flush=True)
    for r in records:
        stage0_bandit.run_bandit(r)
    print("  Stage 1 (model)...", flush=True)
    for r in records:
        stage1.predict(r)
    consolidated = consolidate(records)
    print("  Stage 2 (Llama)...", flush=True)
    for r in consolidated:
        if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD:
            stage2_llama.predict(r)
    if STAGE3_ENABLED:
        print("  Stage 3 (Claude)...", flush=True)
        for r in consolidated:
            if r.stage2_score is not None and \
               STAGE2_SAFE_THRESHOLD <= r.stage2_score <= STAGE2_ESCALATION_THRESHOLD:
                stage3_claude.predict(r)

    # --- Output 1: the text report (terminal + .txt file) ---
    report = build_report(target.name, backend, records, consolidated, args.window, args.stride)
    text = "\n".join(report)
    print("\n" + text)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_dir = PROJECT_ROOT / RESULTS_DIR
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"pipeline_demo_{target.stem}_{ts}.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")

    # --- Output 2: the PNG ---
    png_path = render_png(code, records, args.window, args.stride,
                          PROJECT_ROOT / DEMO_RESULTS_DIR / f"pipeline_demo_{target.stem}_{ts}.png",
                          f"{target.stem}.py  —  Stage 1 per-line view (backend: {backend})")

    print(f"\nSaved report → {txt_path}")
    if png_path:
        print(f"Saved PNG    → {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
