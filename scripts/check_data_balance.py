"""
Data balance report for the processed VUDENC splits.

Usage:
    python scripts/check_data_balance.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import VULN_TYPES
from data.loader import load_vudenc


def _counts(records):
    """Return (total, vulnerable, safe) for a list of WindowRecords."""
    vuln = sum(1 for r in records if r.label == 1)
    return len(records), vuln, len(records) - vuln


def _pct(vuln, total):
    return (100.0 * vuln / total) if total else 0.0


def _split_line(name, records):
    total, vuln, safe = _counts(records)
    pct = _pct(vuln, total)
    return f"    {name:<5} total={total:>7}   vulnerable={vuln:>7}   safe={safe:>7}   vulnerable%={pct:5.1f}%"


def main() -> int:
    print("=" * 72)
    print("VUDENC DATA BALANCE REPORT")
    print("=" * 72)

    # Running totals across all types/splits.
    grand_total = grand_vuln = 0
    any_warning = False
    loaded_types = 0

    for vuln_type in VULN_TYPES:
        print(f"\n### {vuln_type}")
        try:
            train, val, test = load_vudenc(vuln_type)
        except FileNotFoundError:
            print("    (no processed data — run scripts/setup_data.py)")
            continue

        loaded_types += 1
        print(_split_line("train", train))
        print(_split_line("val", val))
        print(_split_line("test", test))

        all_records = train + val + test
        total, vuln, safe = _counts(all_records)
        pct = _pct(vuln, total)
        grand_total += total
        grand_vuln += vuln

        print(f"    {'TOTAL':<5} total={total:>7}   vulnerable={vuln:>7}   safe={safe:>7}   vulnerable%={pct:5.1f}%")

        if pct < 20.0:
            print(f"    [WARNING] only {pct:.1f}% vulnerable (< 20%) — heavily imbalanced toward 'safe'.")
            any_warning = True
        elif pct > 80.0:
            print(f"    [WARNING] {pct:.1f}% vulnerable (> 80%) — heavily imbalanced toward 'vulnerable'.")
            any_warning = True

    # Overall summary.
    print("\n" + "=" * 72)
    print("OVERALL (all types combined)")
    print("=" * 72)
    if grand_total:
        overall_pct = _pct(grand_vuln, grand_total)
        print(f"  types with data : {loaded_types}/{len(VULN_TYPES)}")
        print(f"  total windows   : {grand_total}")
        print(f"  vulnerable      : {grand_vuln} ({overall_pct:.1f}%)")
        print(f"  safe            : {grand_total - grand_vuln} ({100 - overall_pct:.1f}%)")
    else:
        print("  No processed data found. Run scripts/setup_data.py first.")
    if any_warning:
        print("\n  Some types are outside the 20%-80% band — consider class weighting "
              "(the CNN-BiLSTM head already uses weighted BCE) or resampling.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
