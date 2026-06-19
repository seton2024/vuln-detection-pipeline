"""
Consolidation step — runs between Stage 1 and Stage 2.
a small snippet that pinpoints the suspicious code. Non-flagged
records pass through untouched.
"""

from typing import List

from pipeline.contract import WindowRecord
from config import STAGE1_ESCALATION_THRESHOLD


def consolidate(records: List[WindowRecord]) -> List[WindowRecord]:

    #Collapse runs of consecutive Stage-1-flagged windows into one record.
    result: List[WindowRecord] = []
    i = 0
    n = len(records)

    while i < n:
        rec = records[i]

        if not _is_flagged(rec):
            # Not flagged — keep as-is (won't reach Stage 2, but kept for stats).
            result.append(rec)
            i += 1
            continue

        # Start a run: consecutive flagged records sharing (file, vuln_type).
        key = (rec.file, rec.vulnerability_type)
        run = []
        j = i
        while (
            j < n
            and _is_flagged(records[j])
            and (records[j].file, records[j].vulnerability_type) == key
        ):
            run.append(records[j])
            j += 1

        result.append(_merge_run(run))
        i = j

    return result


def _is_flagged(record: WindowRecord) -> bool:
    #True if Stage 1 scored this record above the escalation threshold.
    return (
        record.stage1_score is not None
        and record.stage1_score > STAGE1_ESCALATION_THRESHOLD
    )


def _merge_run(run: List[WindowRecord]) -> WindowRecord:
    """
    the code is set to the lines that appear in EVERY record of the run (intersection), preserving the
    representative's own line order. If no line is shared (e.g. single-line
    windows that overlap only at the character level), we keep the
    representative's original code rather than emit an empty snippet.
    """
    representative = run[len(run) // 2]

    if len(run) == 1:
        return representative

    # Lines present in every window of the run.
    line_sets = [set(r.code.splitlines()) for r in run]
    common = set.intersection(*line_sets) if line_sets else set()

    intersected = [
        line for line in representative.code.splitlines() if line in common
    ]

    if intersected:
        representative.code = "\n".join(intersected)
    # else: leave representative.code as the middle window's own code (fallback).

    return representative
