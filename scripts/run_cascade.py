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
import io
import os
import re
import sys
import tokenize
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.windowing import extract_windows, extract_char_windows
from pipeline.runner import run_pipeline


_CODE_COL = 100   # characters reserved for code; longer lines are truncated with …


def strip_comments_and_docstrings(code: str) -> str:
    """Remove # line comments and triple-quoted strings from Python source.

    Uses Python's tokenizer so # characters inside string literals are never
    accidentally stripped. Falls back to regex if the source has syntax errors.
    """
    try:
        out_tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                continue  # drop  # ...  comments
            if tok.type == tokenize.STRING and tok.string[:3] in ('"""', "'''"):
                continue  # drop triple-quoted docstrings / block comments
            out_tokens.append(tok)
        return tokenize.untokenize(out_tokens)
    except tokenize.TokenError:
        # Regex fallback for files with syntax errors
        code = re.sub(r"#[^\n]*", "", code)
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        code = re.sub(r"'''[\s\S]*?'''", "", code)
        return code


def _fmt(value, nd=3):
    return "-" if value is None else (f"{value:.{nd}f}" if isinstance(value, float) else str(value))


_ADVICE_COL = 80   # max chars for the advice column before truncation


def _build_findings_map(record) -> dict[int, str]:
    """Map line_in_window (1-indexed) → advice text from stage2 and/or stage3 findings."""
    fmap: dict[int, list[str]] = {}

    for finding in (record.stage2_findings or []):
        ln = finding.get("line_in_window", 0)
        reason = finding.get("reason", "")
        fix = finding.get("fix", "")
        fmap.setdefault(ln, []).append(f"[S2] {reason} → {fix}")

    for finding in (record.stage3_findings or []):
        ln = finding.get("line_in_window", 0)
        reason = finding.get("reason", "")
        fix = finding.get("fix", "")
        fmap.setdefault(ln, []).append(f"[S3] {reason} → {fix}")

    return {ln: " | ".join(texts) for ln, texts in fmap.items()}


def _window_to_txt_rows(r) -> list[str]:
    """Return one row per code line, with scores and advice repeated/placed per row.

    Every row looks like:
        <code padded/truncated to _CODE_COL>  <bandit>  <s1>  <s2>  <s3>  <advice>
    A blank row is appended so windows are visually separated.
    """
    scores = (f"  {str(r.bandit_flag):<5}  {_fmt(r.stage1_score):>7}  "
              f"{_fmt(r.stage2_score):>7}  {_fmt(r.stage3_verdict):>14}")
    findings_map = _build_findings_map(r)
    rows = []
    for line_no, line in enumerate((r.code or "<empty>").splitlines(), start=1):
        if len(line) > _CODE_COL:
            display = line[:_CODE_COL - 1] + "…"
        else:
            display = line
        advice = findings_map.get(line_no, "-")
        if len(advice) > _ADVICE_COL:
            advice = advice[:_ADVICE_COL - 1] + "…"
        rows.append(f"{display:<{_CODE_COL}}{scores}  {advice}")
    rows.append("")   # blank separator between windows
    return rows


def _run_one(vuln_type, code, file_path, window_size, stride, flagged_only, char_windows, char_stride):
    """Run the cascade for one vuln type.

    Returns:
        txt_lines: list of strings for the .txt report (code blocks instead of indexes)
        console_lines: list of strings for the terminal (indexes, same as before)
    """
    if char_windows:
        records = extract_char_windows(code, file_path, vuln_type,
                                       char_length=config.VUDENC_BLOCK_LENGTH,
                                       char_stride=char_stride)
    else:
        records = extract_windows(code, file_path, vuln_type, window_size=window_size, stride=stride)
    if not records:
        msg = f"[{vuln_type}] no windows (empty file?)"
        return [msg], [msg]

    results = run_pipeline(records)

    n_raw = len(records)
    n_cons = len(results)
    header = f"\n=== {vuln_type}  ({n_raw} windows → {n_cons} after consolidation) ==="
    col_header = f"{'#':>3}  {'bandit':>6}  {'stage1':>7}  {'stage2':>7}  {'stage3':>14}"
    txt_col_header = (f"{'':>{_CODE_COL}}  {'bandit':<5}  {'stage1':>7}  "
                      f"{'stage2':>7}  {'stage3':>14}  {'advice'}")

    console_lines = [header, col_header]
    txt_lines = [header, txt_col_header]

    flagged_s1 = 0
    reached_s2 = 0
    escalated_s2 = 0
    for i, r in enumerate(results):
        s1_escalated = (r.stage1_score is not None
                        and r.stage1_score > config.STAGE1_ESCALATION_THRESHOLD)
        if s1_escalated:
            flagged_s1 += 1
        if flagged_only and not s1_escalated:
            continue

        if r.stage2_score is not None:
            reached_s2 += 1
            if config.STAGE2_SAFE_THRESHOLD <= r.stage2_score <= config.STAGE2_ESCALATION_THRESHOLD:
                escalated_s2 += 1

        # console: one row per window (index + scores)
        console_lines.append(
            f"{i:>3}  {str(r.bandit_flag):>6}  {_fmt(r.stage1_score):>7}  "
            f"{_fmt(r.stage2_score):>7}  {_fmt(r.stage3_verdict):>14}"
        )

        # txt: one row per code line, scores repeated on every row
        txt_lines.extend(_window_to_txt_rows(r))

    s1_summary = (f"summary stage1: {flagged_s1}/{n_cons} windows escalated past Stage 1 "
                  f"(threshold {config.STAGE1_ESCALATION_THRESHOLD})")
    s2_summary = (f"summary stage2: {reached_s2} reached Stage 2, "
                  f"{escalated_s2} escalated to Stage 3 "
                  f"(uncertain band {config.STAGE2_SAFE_THRESHOLD}–{config.STAGE2_ESCALATION_THRESHOLD})")
    for line in (s1_summary, s2_summary):
        console_lines.append(line)
        txt_lines.append(f"\n{line}" if line == s1_summary else line)

    if flagged_only and flagged_s1 == 0:
        note = "  (nothing escalated - try without --flagged-only, or train Stage 1 models)"
        console_lines.append(note)
        txt_lines.append(note)

    return txt_lines, console_lines


def main() -> int:
    p = argparse.ArgumentParser(description="Run the full vulnerability-detection cascade on a file.")
    p.add_argument("--file", required=True, help="Path to the .py file to scan.")
    p.add_argument("--vuln_type", choices=config.VULN_TYPES, help="Single vulnerability type to check.")
    p.add_argument("--all", action="store_true", help="Check all seven vulnerability types.")
    p.add_argument("--char-windows", action="store_true", default=True,
                   help="Use 200-char windows matching VUDENC training distribution (default: on).")
    p.add_argument("--no-char-windows", dest="char_windows", action="store_false",
                   help="Use line-based windows instead.")
    p.add_argument("--char-stride", type=int, default=50,
                   help="Char stride in char-window mode (default 50).")
    p.add_argument("--window-size", type=int, default=10, help="Window size in lines (line mode only, default 10).")
    p.add_argument("--stride", type=int, default=5, help="Stride in lines (line mode only, default 5).")
    p.add_argument("--flagged-only", action="store_true", help="Only print windows that escalated past Stage 1.")
    p.add_argument("--outdir", default=None,
                   help="Directory for per-type .txt reports. Defaults to results/.")
    args = p.parse_args()

    if not args.all and not args.vuln_type:
        p.error("give either --vuln_type <type> or --all")

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return 1
    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()

    original_len = len(code)
    code = strip_comments_and_docstrings(code)
    print(f"stripped comments/docstrings: {original_len} → {len(code)} chars")

    mode = (f"char({config.VUDENC_BLOCK_LENGTH} chars, stride={args.char_stride})"
            if args.char_windows else f"lines(size={args.window_size}, stride={args.stride})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.file).stem

    results_dir = (Path(args.outdir) if args.outdir
                   else Path(__file__).resolve().parent.parent / config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Shared header printed once to console and included in every per-type file
    report_header = [
        "=" * 72,
        f"CASCADE REPORT — {Path(args.file).name}",
        f"backend: {config.STAGE1_BACKEND}    windowing: {mode}",
        f"OLLAMA_MOCK={config.OLLAMA_MOCK}  STAGE3_ENABLED={config.STAGE3_ENABLED}",
        f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 72,
    ]
    for line in report_header:
        print(line)

    types = config.VULN_TYPES if args.all else [args.vuln_type]

    for vt in types:
        txt_lines, console_lines = _run_one(
            vt, code, args.file, args.window_size, args.stride,
            args.flagged_only, args.char_windows, args.char_stride,
        )
        for line in console_lines:
            print(line)

        out_path = results_dir / f"cascade_{stem}_{vt}_{ts}.txt"
        out_path.write_text(
            "\n".join(report_header + txt_lines) + "\n",
            encoding="utf-8",
        )
        print(f"  → saved {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
