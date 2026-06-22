"""
Full end-to-end pipeline demo runner.

Loads a Python file from input_data/, runs it through ALL stages
(Stage 0 Bandit -> Stage 1 model -> Stage 1.5 consolidation -> Stage 2 Llama ->
Stage 3 Claude), shows a progress bar per stage, and produces:

  1. A per-stage text report — printed to the terminal AND saved to
     results/pipeline_demo_<file>_<timestamp>.txt
  2. One PNG per stage in results/m_demo/:
       - Stage 0 (Bandit): the code with the Bandit test ID (e.g. B608) shown
         next to each flagged line.
       - Stages 1/2/3: a heatmap colouring each line by how vulnerable that
         stage scored it (green -> yellow -> orange -> red), grey for comments.

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

# Optional `# VULN` markers in the input are stripped before scanning (so the
# model never sees them). They are no longer drawn — kept only as input hygiene.
LABEL_MARKER = re.compile(r"\s*#\s*VULN\b.*$", re.IGNORECASE)

# Heatmap colour bands (same scheme as demo.py).
_BOUNDS = [0.0, 0.3, 0.6, 0.8, 1.0]
_MPL = {"green": "#2ecc40", "yellow": "#ffdc00", "orange": "#ff851b",
        "red": "#ff4136", "grey": "#9aa0a6", "blank": "#d4d7db", "code": "#1d2330"}


def _band(score) -> str:
    """Return the colour-band name for a score (None -> blank)."""
    if score is None:
        return "blank"
    if score < 0.3:
        return "green"
    if score < 0.6:
        return "yellow"
    if score < 0.8:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def list_input_files() -> list:
    """Return the alphabetically-sorted .py files in input_data/ (raises if missing)."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(str(INPUT_DIR))
    return sorted(INPUT_DIR.glob("*.py"))


def load_code(path: Path) -> str:
    """Read a file and strip any `# VULN` markers so the model never sees them."""
    return "\n".join(LABEL_MARKER.sub("", line)
                     for line in path.read_text(encoding="utf-8", errors="replace").split("\n"))


def comment_lines(code: str) -> set:
    """Line indices that are `#` comments OR inside a triple-quoted block (docstrings)."""
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
                if q not in s[3:]:
                    in_block = q
                break
    return out


# ---------------------------------------------------------------------------
# Per-record verdicts / scores
# ---------------------------------------------------------------------------

def classify(record: WindowRecord) -> str:
    """Final 'safe' / 'vulnerable' / 'ambiguous' for the text report."""
    label = record.final_label()
    if label == "not_vulnerable":
        return "safe"
    if label == "vulnerable":
        return "vulnerable"
    return "ambiguous"


def stage_score(r: WindowRecord, stage: int):
    """The 0–1 'how vulnerable' score to colour by, AT a given stage (or None)."""
    if stage == 1:
        return r.stage1_score
    if stage == 2:
        return r.stage2_score if r.stage2_score is not None else r.stage1_score
    if stage == 3:
        if r.stage3_verdict == "vulnerable":
            return 1.0
        if r.stage3_verdict == "not_vulnerable":
            return 0.0
        return r.stage2_score if r.stage2_score is not None else r.stage1_score
    return None


# ---------------------------------------------------------------------------
# Per-stage, per-type counts for the text report  (safe, vulnerable, ambiguous)
# ---------------------------------------------------------------------------

def _counts(records, fn) -> dict:
    return {vt: fn([r for r in records if r.vulnerability_type == vt]) for vt in VULN_TYPES}


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
# PNGs
# ---------------------------------------------------------------------------

def _line_scores(code, records, stage, window, stride) -> list:
    """Per source line: the max 'how vulnerable' score across covering windows AT a stage."""
    lines = code.split("\n")
    n = len(lines)
    best = [None] * n
    groups = defaultdict(list)
    for r in records:
        groups[r.vulnerability_type].append(r)
    for recs in groups.values():
        spans = [(0, n)] if (len(recs) == 1 and window > n) else \
            [(j * stride, min(j * stride + window, n)) for j in range(len(recs))]
        for (a, b), r in zip(spans, recs):
            s = stage_score(r, stage)
            if s is None:
                continue
            for i in range(a, b):
                if 0 <= i < n and (best[i] is None or s > best[i]):
                    best[i] = s
    return best


def _new_fig(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, max(2.6, 0.23 * n + 1.8)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return plt, fig, ax


def render_heatmap_png(code, scores, comments, out_path: Path, title: str):
    """Stages 1/2/3: colour each line by its score (green→red), grey for comments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.cm import ScalarMappable
    except ImportError:
        return None
    lines = code.split("\n")
    n = len(lines)
    plt, fig, ax = _new_fig(n)
    for i, line in enumerate(lines):
        y = n - 1 - i
        text = (line if line.strip() else " ")[:200]
        band = "grey" if i in comments else _band(scores[i])
        colour = _MPL[band]
        ax.text(0.03, y, text, va="center", ha="left", fontsize=9, family="monospace", color=colour)
        if i not in comments and scores[i] is not None:
            ax.text(0.995, y, f"{scores[i]:.2f}", va="center", ha="right",
                    fontsize=8, family="monospace", color=colour)
    ax.set_title(title, fontsize=12, loc="left")
    cmap = ListedColormap([_MPL["green"], _MPL["yellow"], _MPL["orange"], _MPL["red"]])
    sm = ScalarMappable(norm=BoundaryNorm(_BOUNDS, cmap.N), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", boundaries=_BOUNDS,
                        ticks=_BOUNDS, pad=0.01, fraction=0.03)
    cbar.set_label("score: safe (green) → vulnerable (red)")
    fig.text(0.5, 0.01, "grey = comment", ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def bandit_findings(code: str) -> dict:
    """Run Bandit once on the whole file; return {line_index: [test_id, ...]}."""
    from pipeline.stage0_bandit import _write_temp_py, _run_bandit_on_file, _parse_bandit_json
    path = _write_temp_py(code)
    try:
        issues = _parse_bandit_json(_run_bandit_on_file(path))
    except Exception:
        issues = []
    finally:
        if os.path.exists(path):
            os.unlink(path)
    by_line = defaultdict(list)
    for it in issues:
        ln = it.get("line_number")
        if ln is not None:
            by_line[ln - 1].append(it.get("test_id", "?"))
    return by_line


def render_bandit_png(code, by_line, comments, out_path: Path, title: str):
    """Stage 0: show each line with the Bandit test ID(s) it triggered (flagged lines in red)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return None
    lines = code.split("\n")
    n = len(lines)
    plt, fig, ax = _new_fig(n)
    for i, line in enumerate(lines):
        y = n - 1 - i
        text = (line if line.strip() else " ")[:200]
        ids = by_line.get(i, [])
        if i in comments:
            colour = _MPL["grey"]
        elif ids:
            colour = _MPL["red"]
        else:
            colour = _MPL["code"]
        ax.text(0.03, y, text, va="center", ha="left", fontsize=9, family="monospace", color=colour)
        if ids:
            ax.text(0.995, y, ", ".join(sorted(set(ids))), va="center", ha="right",
                    fontsize=8, family="monospace", color=_MPL["red"])
    ax.set_title(title, fontsize=12, loc="left")
    fig.text(0.5, 0.01,
             "red = Bandit-flagged line (its test ID is shown on the right)      grey = comment",
             ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_all_pngs(code, records, window, stride, stem, ts, backend) -> list:
    """Render the Bandit PNG (stage 0) + heatmap PNGs (stages 1/2/3)."""
    comments = comment_lines(code)
    out = []
    p0 = render_bandit_png(code, bandit_findings(code), comments,
                           PROJECT_ROOT / DEMO_RESULTS_DIR / f"pipeline_demo_{stem}_{ts}_stage0.png",
                           f"{stem}.py  —  Stage 0 (Bandit)")
    if p0:
        out.append(p0)
    names = {1: f"Stage 1 — {backend}", 2: "Stage 2 — Llama", 3: "Stage 3 — Claude"}
    for stage in (1, 2, 3):
        scores = _line_scores(code, records, stage, window, stride)
        p = render_heatmap_png(code, scores, comments,
                               PROJECT_ROOT / DEMO_RESULTS_DIR / f"pipeline_demo_{stem}_{ts}_stage{stage}.png",
                               f"{stem}.py  —  {names[stage]}")
        if p:
            out.append(p)
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

    code = load_code(target)
    records = extract_all_vuln_types(code, target.name, args.window, args.stride)
    print(f"Scanning {target.name}: {code.count(chr(10)) + 1} lines → "
          f"{len(records)} windows (window={args.window}, stride={args.stride}).\n")
    if not records:
        print("File too short for the window size — nothing to scan.")
        return 0

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

    # Output 1: text report
    text = "\n".join(build_report(target.name, backend, records, consolidated))
    print("\n" + text)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_dir = PROJECT_ROOT / RESULTS_DIR
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"pipeline_demo_{target.stem}_{ts}.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")

    # Output 2: PNGs
    pngs = render_all_pngs(code, records, args.window, args.stride, target.stem, ts, backend)

    print(f"\nSaved report → {txt_path}")
    for p in pngs:
        print(f"Saved PNG    → {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
