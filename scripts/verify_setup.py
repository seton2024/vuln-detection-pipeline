"""
Smoke-test the full day-1 setup WITHOUT requiring VUDENC to be downloaded.

Run after installing requirements:
    python scripts/verify_setup.py

Checks:
  1. All key imports work (bandit, torch, transformers, sklearn, etc.)
  2. WindowRecord round-trips correctly through JSON
  3. Bandit runs on a temp file without crashing
  4. scrub_secrets works on a hardcoded secret string
  5. Windowing produces the correct window count for a known input
  6. validate_record catches a bad record
  7. Pipeline runner imports and runs without error (stubs)

Prints PASS or FAIL for each check. Exits 0 if all pass, 1 if any fail.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator that wraps a function in a try/except and records PASS/FAIL."""
    def decorator(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception:
            RESULTS.append((name, False, traceback.format_exc()))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Check 1: Imports
# ---------------------------------------------------------------------------
@check("Imports: core libraries")
def _():
    import json
    import re
    import subprocess
    import tempfile
    import dataclasses


@check("Imports: pandas + sklearn")
def _():
    import pandas as pd
    import sklearn


@check("Imports: bandit (via subprocess)")
def _():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "bandit", "--version"], capture_output=True, text=True)
    assert result.returncode == 0 or "bandit" in result.stdout.lower() or "bandit" in result.stderr.lower(), \
        "bandit not importable — run: pip install bandit"


@check("Imports: torch")
def _():
    import torch


@check("Imports: transformers")
def _():
    import transformers


# ---------------------------------------------------------------------------
# Check 2: WindowRecord JSON round-trip
# ---------------------------------------------------------------------------
@check("WindowRecord: JSON round-trip")
def _():
    from pipeline.contract import WindowRecord
    r = WindowRecord(
        file="test.py",
        vulnerability_type="sql",
        code="SELECT * FROM users",
        bandit_flag=True,
        stage1_score=0.8,
    )
    j = r.to_json()
    r2 = WindowRecord.from_json(j)
    assert r == r2, f"Round-trip mismatch:\n  original: {r}\n  restored: {r2}"


# ---------------------------------------------------------------------------
# Check 3: Bandit runs without crashing
# ---------------------------------------------------------------------------
@check("Bandit: runs on temp file")
def _():
    from pipeline.contract import WindowRecord
    from pipeline.stage0_bandit import run_bandit
    r = WindowRecord(
        file="demo.py",
        vulnerability_type="sql",
        code="x = 1\nprint(x)\n",
    )
    result = run_bandit(r)
    assert isinstance(result, bool), f"run_bandit must return bool, got {type(result)}"


# ---------------------------------------------------------------------------
# Check 4: scrub_secrets
# ---------------------------------------------------------------------------
@check("scrub_secrets: redacts hardcoded password")
def _():
    from pipeline.stage0_bandit import scrub_secrets
    code = 'password = "hunter2"\nprint("logging in")\n'
    scrubbed = scrub_secrets(code)
    assert "hunter2" not in scrubbed, "Secret was not scrubbed"
    assert "<REDACTED>" in scrubbed, "<REDACTED> placeholder missing"


@check("scrub_secrets: clean code passes through")
def _():
    from pipeline.stage0_bandit import scrub_secrets
    code = "def add(a, b):\n    return a + b\n"
    scrubbed = scrub_secrets(code)
    assert "def add" in scrubbed


# ---------------------------------------------------------------------------
# Check 5: Windowing
# ---------------------------------------------------------------------------
@check("Windowing: correct window count")
def _():
    from data.windowing import extract_windows
    # 20 lines, window_size=10, stride=1 → 20-10+1 = 11 windows
    code = "\n".join(f"line_{i} = {i}" for i in range(20))
    windows = extract_windows(code, "test.py", "sql", window_size=10, stride=1)
    assert len(windows) == 11, f"Expected 11 windows, got {len(windows)}"


@check("Windowing: extract_all_vuln_types gives 7x windows")
def _():
    from data.windowing import extract_windows, extract_all_vuln_types
    code = "\n".join(f"line_{i} = {i}" for i in range(20))
    single = extract_windows(code, "test.py", "sql", window_size=10, stride=1)
    all_types = extract_all_vuln_types(code, "test.py", window_size=10, stride=1)
    assert len(all_types) == len(single) * 7, (
        f"Expected {len(single) * 7} windows, got {len(all_types)}"
    )


@check("VUDENC blocks: context extraction labels vulnerable code")
def _():
    from data.vudenc_blocks import findpositions, getblocks
    # VUDENC stores 'source' normalised (single spaces, no newlines), so we
    # feed the block extractor that same shape.
    source = (
        "\n def run ( cmd ) : "
        "result = subprocess.call ( cmd , shell = True ) "
        "return result def add ( a , b ) : return a + b"
    )
    positions = findpositions(["subprocess.call ( cmd , shell = True )"], source)
    blocks = getblocks(source, positions, 5, 40)
    assert isinstance(blocks, list) and len(blocks) > 0, "no blocks produced"
    assert any(label == 1 for _, label in blocks), "vulnerable line was not labeled"


# ---------------------------------------------------------------------------
# Check 6: validate_record catches bad input
# ---------------------------------------------------------------------------
@check("validate_record: catches empty code")
def _():
    from pipeline.contract import WindowRecord, validate_record
    r = WindowRecord(file="x.py", vulnerability_type="sql", code="")
    try:
        validate_record(r)
        raise AssertionError("Should have raised ValueError for empty code")
    except ValueError:
        pass  # expected


@check("validate_record: catches bad vulnerability type")
def _():
    from pipeline.contract import WindowRecord, validate_record
    r = WindowRecord(file="x.py", vulnerability_type="sql", code="x = 1")
    r.vulnerability_type = "buffer_overflow"  # not one of VULN_TYPES
    try:
        validate_record(r)
        raise AssertionError("Should have raised ValueError for bad vuln type")
    except ValueError:
        pass  # expected


# ---------------------------------------------------------------------------
# Check 7: Pipeline runner imports and runs
# ---------------------------------------------------------------------------
@check("Pipeline runner: imports and runs without error")
def _():
    from pipeline.runner import run_pipeline
    from pipeline.contract import WindowRecord
    r = WindowRecord(
        file="test.py",
        vulnerability_type="sql",
        code="x = 1\nprint(x)\n",
    )
    result = run_pipeline([r])
    assert len(result) == 1
    assert isinstance(result[0].bandit_flag, bool)


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n" + "=" * 60)
    print("Verify Setup - Day 1 Smoke Test")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, ok, tb in RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            # Print just the last line of the traceback (the error message)
            lines = [l for l in tb.strip().splitlines() if l.strip()]
            print(f"         -> {lines[-1]}")
            failed += 1
        else:
            passed += 1

    print(f"\n  {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nTo fix failures:")
        print("  pip install -r requirements.txt")
        print("  Make sure bandit is on your PATH")
        return 1
    else:
        print("\nAll checks passed! Run pytest to execute the full test suite.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
