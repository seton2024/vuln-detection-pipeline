"""
Full end-to-end pipeline demo runner.

Loads a Python file from input_data/, runs it through ALL stages
(Stage 0 Bandit -> Stage 1 model -> Stage 1.5 consolidation -> Stage 2 Llama ->
Stage 3 Claude), shows a progress bar per stage, and produces:

  1. A per-stage text report — printed to the terminal AND saved to
     results/pipeline_demo_<file>_<timestamp>.txt
  2. One PNG PER STAGE in results/m_demo/, where each source line is coloured by
     that stage's outcome:
       green -> yellow -> red  = safe -> vulnerable (Stage 1 score)
       grey                    = comment lines
       black                   = code escalated to the next stage (undetermined)
       red dot (●)             = TRUE vulnerable line (from `# VULN` labels)

Ground-truth labels: lines tagged with a trailing `# VULN` comment in the input
are treated as truly-vulnerable (red dot). The marker is stripped before
scanning so the model never sees it. This labelling is for THIS demo only — in
production there are no such labels.

Separate from scripts/demo.py (the per-line visualizer), which is left untouched.

USAGE
    python scripts/run_pipeline_demo.py --list
    python scripts/run_pipeline_demo.py --file 0
    python scripts/run_pipeline_demo.py --file 0 --backend graphcodebert --window 10 --stride 1
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

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

# A trailing `# VULN` comment marks a truly-vulnerable line (ground truth for the demo).
LABEL_MARKER = re.compile(r"\s*#\s*VULN\b.*$", re.IGNORECASE)

# Rank for "worst wins" when aggregating a line's status across windows.
_RANK = {"safe": 0, "escalated": 1, "vulnerable": 2}


# ---------------------------------------------------------------------------
# Input loading (with ground-truth labels)
# ---------------------------------------------------------------------------

def list_input_files() -> list:
    """Return the alphabetically-sorted .py files in input_data/ (raises if missing)."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(str(INPUT_DIR))
    return sorted(INPUT_DIR.glob("*.py"))


def load_labeled_code(path: Path):
    """Read a file, strip `# VULN` markers, and return (clean_code, true_vuln_line_set)."""
    clean_lines, true_vuln = [], set()
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n")):
        if LABEL_MARKER.search(line):
            true_vuln.add(i)
            line = LABEL_MARKER.sub("", line)
        clean_lines.append(line)
    return "\n".join(clean_lines), true_vuln


# ---------------------------------------------------------------------------
# Per-record verdicts
# ---------------------------------------------------------------------------

def classify(record: WindowRecord) -> str:
    """Final 'safe' / 'vulnerable' / 'ambiguous' for the text report."""
    label = record.final_label()
    if label == "not_vulnerable":
        return "safe"
    if label == "vulnerable":
        return "vulnerable"
    return "ambiguous"


def status_at_stage(r: WindowRecord, stage: int):
    """Return (category, score_for_gradient) for a record AT a given stage, or None.

    category: 'safe' (decided safe here), 'vulnerable' (decided vulnerable here),
              'escalated' (sent to the next stage, undetermined here).
    """
    if stage == 0:
        return ("escalated", None) if r.bandit_flag else ("safe", 0.0)
    if stage == 1:
        if r.stage1_score is None:
            return None
        return ("escalated", r.stage1_score) if r.stage1_score > STAGE1_ESCALATION_THRESHOLD \
            else ("safe", r.stage1_score)
    if stage == 2:
        if r.stage2_score is None:
            # never reached Stage 2 -> Stage 1 already called it safe (or flagged but Llama gave nothing)
            if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD:
                return ("escalated", None)
            return ("safe", r.stage1_score)
        if r.stage2_score < STAGE2_SAFE_THRESHOLD:
            return ("safe", r.stage2_score)
        if r.stage2_score > STAGE2_ESCALATION_THRESHOLD:
            return ("vulnerable", r.stage2_score)
        return ("escalated", r.stage2_score)
    # stage 3
    if r.stage3_verdict == "vulnerable":
        return ("vulnerable", None)
    if r.stage3_verdict == "not_vulnerable":
        return ("safe", None)
    if r.stage2_score is not None and STAGE2_SAFE_THRESHOLD <= r.stage2_score <= STAGE2_ESCALATION_THRESHOLD:
        return ("escalated", None)   # was sent to Stage 3 but no verdict (disabled / unresolved)
    return status_at_stage(r, 2)     # otherwise inherit the Stage 2 outcome


# ---------------------------------------------------------------------------
# Per-stage, per-type counts for the text report  (safe, vulnerable, ambiguous)
# ---------------------------------------------------------------------------

def _counts(records, fn) -> dict:
    out = {}
    for vt in VULN_TYPES:
        out[vt] = fn([r for r in records if r.vulnerability_type == vt])
    return out


def stage0_counts(records):
    return _counts(records, lambda rs: (sum(1 for r in rs if not r.bandit_flag),
                                        sum(1 for r in rs if r.bandit_flag), 0))


def stage1_counts(records):
    def f(rs):
        rs = [r for r in rs if r.stage1_score is not None]
        return (sum(1 for r in rs if r.stage1_score <= STAGE1_ESCALATION_THRESHOLD), 0,
                sum(1 for r in rs if r.stage1_score > STAGE1_ESCALATION_THRESHOLD))
    return _counts(records, f)


def stage2_counts(records):
    def f(rs):
        rs = [r for r in rs if r.stage2_score is not None]
        return (sum(1 for r in rs if r.stage2_score < STAGE2_SAFE_THRESHOLD),
                sum(1 for r in rs if r.stage2_score > STAGE2_ESCALATION_THRESHOLD),
                sum(1 for r in rs if STAGE2_SAFE_THRESHOLD <= r.stage2_score <= STAGE2_ESCALATION_THRESHOLD))
    return _counts(records, f)


def stage3_counts(records):
    def f(rs):
        rs = [r for r in rs if r.stage3_verdict is not None]
        return (sum(1 for r in rs if r.stage3_verdict == "not_vulnerable"),
                sum(1 for r in rs if r.stage3_verdict == "vulnerable"), 0)
    return _counts(records, f)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _counts_table(counts: dict) -> list:
    lines = [f"  {'Vuln Type':<22} {'Safe':>6} {'Vulnerable':>11} {'Ambiguous':>10}",
             "  " + "-" * 52]
    for vt, (safe, vuln, amb) in counts.items():
        if safe + vuln + amb:
            lines.append(f"  {vt:<22} {safe:>6} {vuln:>11} {amb:>10}")
    if len(lines) == 2:
        lines.append("  (no windows at this stage)")
    return lines


def _example(record, label: str) -> list:
    lines = [f"  {label}:"]
    if record is None:
        lines.append("    (none)")
        return lines
    for ln in record.code.split("\n"):
        lines.append(f"    | {ln}")
    return lines


def _max_by(records, attr):
    pool = [r for r in records if getattr(r, attr) is not None]
    return max(pool, key=lambda r: getattr(r, attr), default=None)


def build_report(filename, backend, records, consolidated) -> list:
    """Build the per-stage text report as a list of lines."""
    R = ["=" * 66, f" PIPELINE REPORT — {filename}",
         f" backend: {backend}    Stage 3: {'ENABLED' if STAGE3_ENABLED else 'disabled'}"
         f"    {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 66]

    R.append("\nSTAGE 0 — Bandit (static analysis)")
    R += _counts_table(stage0_counts(records))
    R += _example(next((r for r in records if r.bandit_flag), None), "Example Bandit-flagged snippet")

    R.append(f"\nSTAGE 1 — {backend}")
    R.append(f"  Total windows scanned: {len(records)}")
    R += _counts_table(stage1_counts(records))
    R += _example(_max_by(records, "stage1_score"), "Example highest-scoring window")

    R.append("\nSTAGE 1.5 — Consolidation")
    flagged = [r for r in records if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD]
    cons_flagged = [r for r in consolidated if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD]
    R.append(f"  Flagged windows received    : {len(flagged)}")
    R.append(f"  Consolidated windows produced: {len(cons_flagged)}")
    R += _example(cons_flagged[len(cons_flagged) // 2] if cons_flagged else None,
                  "Example consolidated window (sent to Stage 2)")

    R.append("\nSTAGE 2 — Llama")
    R += _counts_table(stage2_counts(consolidated))
    R += _example(_max_by(consolidated, "stage2_score"), "Example Stage 2 section")

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
        R += _example(received[0] if received else None, "Example snippet that would go to Claude")

    R.append("\n" + "=" * 66)
    return R


# ---------------------------------------------------------------------------
# Per-stage PNGs
# ---------------------------------------------------------------------------

def _comment_lines(code) -> set:
    """Return line indices that are `#` comments OR inside a triple-quoted block (docstrings)."""
    out = set()
    in_block = None
    for i, line in enumerate(code.split("\n")):
        s = line.strip()
        if in_block:
            out.add(i)
            if in_block in line:
                in_block = None
            continue
        if s.startswith("#"):
            out.add(i)
            continue
        for q in ('"""', "'''"):
            if s.startswith(q):
                out.add(i)
                if q not in s[3:]:        # not closed on the same line
                    in_block = q
                break
    return out


def _line_status(code, records, stage, window, stride, true_vuln, comment_set) -> list:
    """Per source line: worst category + max score AT a given stage, plus comment/true_vuln flags."""
    lines = code.split("\n")
    n = len(lines)
    cat, score = [None] * n, [None] * n
    groups = defaultdict(list)
    for r in records:
        groups[r.vulnerability_type].append(r)
    for recs in groups.values():
        spans = [(0, n)] if (len(recs) == 1 and window > n) else \
            [(j * stride, min(j * stride + window, n)) for j in range(len(recs))]
        for (a, b), r in zip(spans, recs):
            st = status_at_stage(r, stage)
            if st is None:
                continue
            c, s = st
            for i in range(a, b):
                if 0 <= i < n:
                    if cat[i] is None or _RANK[c] > _RANK[cat[i]]:
                        cat[i] = c
                    if s is not None and (score[i] is None or s > score[i]):
                        score[i] = s
    return [{"text": t, "comment": i in comment_set,
             "category": cat[i], "score": score[i], "true_vuln": i in true_vuln}
            for i, t in enumerate(lines)]


def render_stage_png(rows, out_path: Path, title: str):
    """Render one stage's coloured per-line PNG. Returns the path or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        return None

    cmap = colormaps["RdYlGn_r"]              # 0=green (safe), 1=red (vulnerable)
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
            colour = "#9aa0a6"                              # grey
        elif row["category"] == "vulnerable":
            colour = "#c0392b"                              # red
        elif row["category"] == "escalated":
            colour = "#000000"                              # black
        elif row["category"] == "safe":
            colour = cmap(row["score"]) if row["score"] is not None else "#2ecc40"
        else:
            colour = "#cfd2d6"                              # uncovered / blank
        if row["true_vuln"]:
            ax.text(0.004, y, "●", va="center", ha="left", fontsize=10, color="#ff0000")
        ax.text(0.03, y, text, va="center", ha="left", fontsize=9, family="monospace", color=colour)
        if row["score"] is not None and not row["comment"]:
            ax.text(0.995, y, f"{row['score']:.2f}", va="center", ha="right",
                    fontsize=8, family="monospace", color=colour)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, loc="left")

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, fraction=0.03)
    cbar.set_label("score: safe (green) → vulnerable (red)")
    fig.text(0.5, 0.01,
             "grey = comment      black = escalated to next stage (undetermined)      "
             "● red dot = TRUE vulnerable line (label)",
             ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_all_stage_pngs(code, records, window, stride, true_vuln, stem, ts, backend) -> list:
    """Render one PNG per stage (0–3). Returns the list of saved paths."""
    names = {0: "Stage 0 — Bandit", 1: f"Stage 1 — {backend}",
             2: "Stage 2 — Llama", 3: "Stage 3 — Claude"}
    comment_set = _comment_lines(code)
    out = []
    for stage in (0, 1, 2, 3):
        rows = _line_status(code, records, stage, window, stride, true_vuln, comment_set)
        path = render_stage_png(
            rows, PROJECT_ROOT / DEMO_RESULTS_DIR / f"pipeline_demo_{stem}_{ts}_stage{stage}.png",
            f"{stem}.py  —  {names[stage]}")
        if path:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    if not files or not (0 <= args.file < len(files)):
        print(f"--file {args.file} is out of range (0..{len(files) - 1}). Use --list.")
        return 1

    target = files[args.file]
    backend = args.backend
    stage1 = stage1_cnn_bilstm if backend == "cnn_bilstm" else stage1_graphcodebert

    code, true_vuln = load_labeled_code(target)
    records = extract_all_vuln_types(code, target.name, args.window, args.stride)
    print(f"Scanning {target.name}: {code.count(chr(10)) + 1} lines, {len(true_vuln)} labelled-vulnerable "
          f"→ {len(records)} windows (window={args.window}, stride={args.stride}).\n")
    if not records:
        print("File too short for the window size — nothing to scan.")
        return 0

    # Run the cascade with a progress bar per stage.
    for r in tqdm(records, desc="Stage 0  Bandit  ", unit="win"):
        stage0_bandit.run_bandit(r)
    for r in tqdm(records, desc="Stage 1  Model   ", unit="win"):
        stage1.predict(r)
    consolidated = consolidate(records)
    to_stage2 = [r for r in consolidated
                 if r.stage1_score is not None and r.stage1_score > STAGE1_ESCALATION_THRESHOLD]
    for r in tqdm(to_stage2, desc="Stage 2  Llama   ", unit="sec"):
        stage2_llama.predict(r)
    if STAGE3_ENABLED:
        to_stage3 = [r for r in consolidated if r.stage2_score is not None
                     and STAGE2_SAFE_THRESHOLD <= r.stage2_score <= STAGE2_ESCALATION_THRESHOLD]
        for r in tqdm(to_stage3, desc="Stage 3  Claude  ", unit="sec"):
            stage3_claude.predict(r)

    # Output 1: text report (terminal + .txt)
    text = "\n".join(build_report(target.name, backend, records, consolidated))
    print("\n" + text)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_dir = PROJECT_ROOT / RESULTS_DIR
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"pipeline_demo_{target.stem}_{ts}.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")

    # Output 2: one PNG per stage
    pngs = render_all_stage_pngs(code, records, args.window, args.stride, true_vuln,
                                 target.stem, ts, backend)

    print(f"\nSaved report → {txt_path}")
    for p in pngs:
        print(f"Saved PNG    → {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
