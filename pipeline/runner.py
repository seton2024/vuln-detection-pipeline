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
from pipeline import stage0_bandit, stage1_graphcodebert, stage1_cnn_bilstm, stage2_llama, stage3_claude
from pipeline.stage15_consolidator import consolidate
from config import (
    STAGE1_BACKEND,
    STAGE1_ESCALATION_THRESHOLD,
    STAGE2_SAFE_THRESHOLD,
    STAGE2_ESCALATION_THRESHOLD,
    STAGE3_ENABLED,
)

# Map the STAGE1_BACKEND config value to the module that implements predict().
_STAGE1_BACKENDS = {
    "graphcodebert": stage1_graphcodebert,
    "cnn_bilstm": stage1_cnn_bilstm,
}


def get_stage1():
    """Return the Stage 1 module selected by config.STAGE1_BACKEND."""
    return _STAGE1_BACKENDS.get(STAGE1_BACKEND, stage1_graphcodebert)


def run_pipeline(records: List[WindowRecord]) -> List[WindowRecord]:
    """
    Run the records through the full cascade

    Each stage writes its result back into the record. The returned list is the
    consolidated one (runs of flagged windows merged into single representatives).
    """
    # Stage 0: Bandit — always runs, sets bandit_flag.
    for record in records:
        stage0_bandit.run_bandit(record)

    # Stage 1: the backend selected by config.STAGE1_BACKEND — sets stage1_score.
    stage1 = get_stage1()
    for record in records:
        stage1.predict(record)

    # Consolidation: merge consecutive flagged windows before the expensive stages.
    consolidated = consolidate(records)

    # Stage 2: Llama — only for records Stage 1 flagged as suspicious.
    # Use the per-type threshold from the checkpoint when available (it is the
    # F-beta-optimal cutoff chosen during training). Fall back to the global
    # STAGE1_ESCALATION_THRESHOLD only when no checkpoint threshold exists.
    for record in consolidated:
        if record.stage1_score is None:
            continue
        th = (stage1.get_threshold(record.vulnerability_type)
              if hasattr(stage1, "get_threshold")
              else STAGE1_ESCALATION_THRESHOLD)
        if record.stage1_score > th:
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
