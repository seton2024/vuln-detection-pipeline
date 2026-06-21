"""
Demo: scan a WHOLE source file with a sliding window and color every line by its
local vulnerability score (like the VUDENC GraphCodeBERT visualization).

USAGE
    python scripts/demo.py --vuln_type command_injection --index 0 --backend graphcodebert
"""

import argparse
import json
import os
import sys

# Quiet the transformers loading chatter (must be set before transformers loads).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import VULN_TYPES, STAGE1_BACKEND, RESULTS_DIR, VUDENC_DATA_DIR, VUDENC_BLOCK_LENGTH
from pipeline.contract import WindowRecord
from pipeline import stage0_bandit, stage1_graphcodebert, stage1_cnn_bilstm
from data.vudenc_blocks import getcontextPos, findpositions

# --- color bands (shared by terminal + PNG) ---
_BOUNDS = [0.0, 0.3, 0.6, 0.8, 1.0]
_ANSI = {"green": "\033[32m", "yellow": "\033[33m", "orange": "\033[38;5;208m",
         "red": "\033[31m", "grey": "\033[90m"}
_RESET = "\033[0m"
_MPL = {"green": "#2ecc40", "yellow": "#ffdc00", "orange": "#ff851b",
        "red": "#ff4136", "grey": "#aaaaaa"}

_BACKENDS = {"graphcodebert": stage1_graphcodebert, "cnn_bilstm": stage1_cnn_bilstm}


def _band(score):
    if score is None:
        return "grey"
    if score < 0.3:
        return "green"
    if score < 0.6:
        return "yellow"
    if score < 0.8:
        return "orange"
    return "red"


def _enable_ansi():
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


# Source selection

def _load_source_from_dataset(vuln_type, index):
    #Return (source_with_comments, filename, badparts) for the index-th dataset file.
    path = os.path.join(VUDENC_DATA_DIR, f"plain_{vuln_type}")
    if not os.path.exists(path):
        return None, None, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    i = 0
    for repo, commits in data.items():
        for sha, entry in commits.items():
            for fname, info in entry.get("files", {}).items():
                src = info.get("sourceWithComments") or info.get("source")
                if not src or not src.strip():
                    continue
                if i == index:
                    badparts = []
                    for change in info.get("changes", []):
                        badparts.extend(b for b in change.get("badparts", []) if b and b.strip())
                    return src, fname, badparts
                i += 1
    return None, None, None


def _line_spans(source):
    #Return [(line_text, start_char, end_char), ...] for each line in source.
    spans = []
    pos = 0
    for line in source.split("\n"):
        start = pos
        end = pos + len(line)
        spans.append((line, start, end))
        pos = end + 1  # account for the '\n'
    return spans


def _overlaps_bad(start, end, badpositions):
    return any(start <= b[1] and end >= b[0] for b in badpositions)


# Scoring


def score_lines(source, vuln_type, stage1, context_len, max_lines):
    """
    Score every (non-blank) line via a context window centered on it.
    Returns a list of dicts: {text, score, band, start, end}.
    """
    spans = _line_spans(source)
    truncated = len(spans) > max_lines
    spans = spans[:max_lines]

    rows = []
    for n, (text, start, end) in enumerate(spans):
        score = None
        if text.strip():
            middle = (start + end) // 2
            ctx = getcontextPos(source, middle, context_len)
            window_text = source[ctx[0]:ctx[1]] if ctx is not None else source[start:end]
            if window_text.strip():
                rec = WindowRecord(file="demo", vulnerability_type=vuln_type, code=window_text)
                stage1.predict(rec)
                score = rec.stage1_score
        rows.append({"text": text, "score": score, "band": _band(score), "start": start, "end": end})
        if (n + 1) % 25 == 0:
            print(f"    scored {n + 1}/{len(spans)} lines...", flush=True)
    return rows, truncated



# PNG output

def save_png(rows, badpositions, vuln_type, source_id, backend, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("[demo] matplotlib not installed — skipping PNG. (pip install matplotlib)")
        return False

    n = len(rows)
    cmap = ListedColormap([_MPL["green"], _MPL["yellow"], _MPL["orange"], _MPL["red"]])
    norm = BoundaryNorm(_BOUNDS, cmap.N)

    fig_h = max(2.5, 0.23 * n + 1.4)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, row in enumerate(rows):
        y = n - 1 - i  # first line on top
        color = _MPL[row["band"]]
        text = row["text"] if row["text"].strip() else " "
        if len(text) > 200:
            text = text[:197] + "..."
        # known-vulnerable marker
        if _overlaps_bad(row["start"], row["end"], badpositions):
            ax.text(0.004, y, "●", va="center", ha="left", fontsize=9, color="#b10dc9")
        ax.text(0.02, y, text, va="center", ha="left", fontsize=9, family="monospace", color=color)
        if row["score"] is not None:
            ax.text(0.995, y, f"{row['score']:.2f}", va="center", ha="right",
                    fontsize=8, family="monospace", color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{vuln_type}   {source_id}   (backend: {backend})   "
                 f"● = known-vulnerable line", fontsize=12, loc="left")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        boundaries=_BOUNDS, ticks=_BOUNDS, pad=0.01, fraction=0.03)
    cbar.set_label("P(vulnerable)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True



# Main


def main() -> int:
    parser = argparse.ArgumentParser(description="Color a whole source file by per-line vulnerability score.")
    parser.add_argument("--vuln_type", required=True, choices=VULN_TYPES,
                        help="Which type's Stage 1 model to use.")
    parser.add_argument("--index", type=int, default=0,
                        help="Which source file from the raw VUDENC data (ignored if --file given).")
    parser.add_argument("--file", default=None,
                        help="Scan an arbitrary .py file instead of the dataset.")
    parser.add_argument("--backend", choices=["graphcodebert", "cnn_bilstm"], default=None,
                        help=f"Stage 1 backend to use (default: config STAGE1_BACKEND = {STAGE1_BACKEND}).")
    parser.add_argument("--context", type=int, default=VUDENC_BLOCK_LENGTH,
                        help="Context window length in characters (default: VUDENC_BLOCK_LENGTH).")
    parser.add_argument("--max-lines", type=int, default=150,
                        help="Cap on lines scored (keeps the demo fast). Default 150.")
    args = parser.parse_args()

    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # --- pick the source ---
    badparts = []
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return 1
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        source_id = os.path.basename(args.file)
        out_id = os.path.splitext(source_id)[0]
    else:
        source, fname, badparts = _load_source_from_dataset(args.vuln_type, args.index)
        if source is None:
            print(f"No source #{args.index} found for {args.vuln_type!r} in {VUDENC_DATA_DIR}. "
                  f"(Need the raw data/vudenc/plain_{args.vuln_type} file.)")
            return 1
        source_id = f"{fname}  [#{args.index}]"
        out_id = str(args.index)

    badpositions = findpositions(badparts, source) if badparts else []

    backend = args.backend or STAGE1_BACKEND
    stage1 = _BACKENDS[backend]

    # --- Stage 0 on the whole file (informational) ---
    whole = WindowRecord(file="demo", vulnerability_type=args.vuln_type, code=source)
    stage0_bandit.run_bandit(whole)

    # --- score every line ---
    print(f"Scoring {args.vuln_type} source with backend={backend} ...", flush=True)
    rows, truncated = score_lines(source, args.vuln_type, stage1, args.context, args.max_lines)
    if truncated:
        print(f"(note: source truncated to first {args.max_lines} lines — raise --max-lines to see more)")


    out_path = os.path.join(RESULTS_DIR, "m_demo", f"demo_{args.vuln_type}_{out_id}.png")
    if save_png(rows, badpositions, args.vuln_type, source_id, backend, out_path):
        print(f"Saved visualization -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
