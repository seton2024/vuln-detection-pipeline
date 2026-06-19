"""
Pipeline runner — chains Stage 0 through Stage 3 with a consolidation step.

Flow:
  Stage 0 (Bandit)          — runs on every record, sets bandit_flag
  Stage 1 (GraphCodeBERT)   — runs on every record, sets stage1_score
  Stage 1.5 (Consolidation) — merges runs of consecutive flagged windows
  Stage 2 (Llama)           — only for records Stage 1 flagged (score > threshold)
  Stage 3 (Claude)          — only for records Stage 2 left uncertain, if enabled

"""

from typing import List

from pipeline.contract import WindowRecord
from pipeline import stage0_bandit, stage1_graphcodebert, stage2_llama, stage3_claude
from pipeline.stage15_consolidator import consolidate
from config import (
    STAGE1_ESCALATION_THRESHOLD,
    STAGE2_SAFE_THRESHOLD,
    STAGE2_ESCALATION_THRESHOLD,
    STAGE3_ENABLED,
)


def run_pipeline(records: List[WindowRecord]) -> List[WindowRecord]:
    """
    Run the records through the full cascade

    Each stage writes its result back into the record. The returned list is the
    consolidated one (runs of flagged windows merged into single representatives).
    """
    # Stage 0: Bandit — always runs, sets bandit_flag.
    for record in records:
        stage0_bandit.run_bandit(record)

    # Stage 1: GraphCodeBERT — always runs, sets stage1_score.
    for record in records:
        stage1_graphcodebert.predict(record)

    # Consolidation: merge consecutive flagged windows before the expensive stages.
    consolidated = consolidate(records)

    # Stage 2: Llama — only for records Stage 1 flagged as suspicious.
    for record in consolidated:
        if record.stage1_score is not None and record.stage1_score > STAGE1_ESCALATION_THRESHOLD:
            stage2_llama.predict(record)

    # Stage 3: Claude — only for records Stage 2 left uncertain (and if enabled).
    for record in consolidated:
        if record.stage2_score is None:
            continue
        if record.stage2_score < STAGE2_SAFE_THRESHOLD:
            continue  # Stage 2 says safe — stop
        if record.stage2_score > STAGE2_ESCALATION_THRESHOLD:
            continue  # Stage 2 says clearly vulnerable — stop
        if STAGE3_ENABLED:
            stage3_claude.predict(record)

    return consolidated
