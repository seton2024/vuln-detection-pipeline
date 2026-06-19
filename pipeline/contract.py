"""
Data contract for the vulnerability detection pipeline.

we use windows for the code snippets. This code defines the Window class
"""

from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json

from config import (VULN_TYPES,
                    STAGE1_ESCALATION_THRESHOLD,
                    STAGE2_SAFE_THRESHOLD,
                    STAGE2_ESCALATION_THRESHOLD)


@dataclass
class WindowRecord:
    """

    Identity (written at creation, never changes):
      file              - path of the source file, or "vudenc" for training data
      vulnerability_type - which of the 7 vuln types we're checking for
      code              - the code snippet (~200 chars for training, ~50-300 inference)
      label             - ground-truth answer from the dataset (1=vuln, 0=safe), None for live code

    Fields filled in by each pipeline stage (default None = "not run yet"):
      bandit_flag       - Stage 0: True if Bandit found a relevant issue
      stage1_score      - Stage 1: 0.0–1.0 confidence from GraphCodeBERT
      stage2_score      - Stage 2: 0.0–1.0 confidence from Llama
      stage3_verdict    - Stage 3: final verdict from Claude ("vulnerable"/"not_vulnerable")
      stage3_findings   - Stage 3: optional list of line-level finding dicts
    """

    #Identity
    file: str
    vulnerability_type: str
    code: str

    # Ground-truth label from the dataset (VUDENC): 1 = vulnerable, 0 = safe.
    # None during live inference on new code (no known answer yet).
    label: Optional[int] = None

    # Stage outputs
    bandit_flag: bool = False
    stage1_score: Optional[float] = None
    stage2_score: Optional[float] = None
    stage3_verdict: Optional[Literal["vulnerable", "not_vulnerable"]] = None
    stage3_findings: Optional[list[dict]] = None

    #Serialization helpers

    def to_dict(self) -> dict:
        #Convert  record to a dictionary (for saving to JSON, etc.)
        return asdict(self)

    def to_json(self) -> str:
        #A nicely-formatted JSON string.
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "WindowRecord":
        #Recreate a WindowRecord from a dictionary.
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "WindowRecord":
        #Recreate a WindowRecord from a JSON string.
        return cls.from_dict(json.loads(s))

    #Pipeline logic helpers

    def is_complete(self) -> bool:
        """
        Returns True if this record has reached a final decision and no more
        stages need to run.

        A record is complete when:
          - Stage 3 has delivered a verdict, OR
          - Stage 1 scored the code safe (no need to escalate), OR
          - Stage 2 scored clearly enough (very safe OR very vulnerable) to stop.
        """
        if self.stage3_verdict is not None:
            return True
        if self.stage1_score is not None and self.stage1_score <= STAGE1_ESCALATION_THRESHOLD:
            return True   # Stage 1 said "safe enough" — stop here
        if self.stage2_score is not None:
            if self.stage2_score < STAGE2_SAFE_THRESHOLD or self.stage2_score > STAGE2_ESCALATION_THRESHOLD:
                return True  # Stage 2 gave a clear answer — stop here
        return False

    def final_label(self) -> Optional[Literal["vulnerable", "not_vulnerable"]]:
        """
        Returns the final verdict if the record is complete, otherwise None.
        """
        if self.stage3_verdict is not None:
            return self.stage3_verdict
        if self.stage1_score is not None and self.stage1_score <= STAGE1_ESCALATION_THRESHOLD:
            return "not_vulnerable"
        if self.stage2_score is not None:
            if self.stage2_score < STAGE2_SAFE_THRESHOLD:
                return "not_vulnerable"
            if self.stage2_score > STAGE2_ESCALATION_THRESHOLD:
                return "vulnerable"
        return None


def validate_record(record: WindowRecord) -> None:
    """
    Errors for format problems.

    """
    if record.vulnerability_type not in VULN_TYPES:
        raise ValueError(
            f"Unknown vulnerability type: '{record.vulnerability_type}'. "
            f"Must be one of: {VULN_TYPES}"
        )

    if not record.code or not record.code.strip():
        raise ValueError("WindowRecord.code must not be empty.")

    if not record.file:
        raise ValueError("WindowRecord.file must not be empty.")

    if record.stage1_score is not None and not (0.0 <= record.stage1_score <= 1.0):
        raise ValueError(
            f"stage1_score must be between 0.0 and 1.0, got {record.stage1_score}"
        )

    if record.stage2_score is not None and not (0.0 <= record.stage2_score <= 1.0):
        raise ValueError(
            f"stage2_score must be between 0.0 and 1.0, got {record.stage2_score}"
        )

    if record.label is not None and record.label not in (0, 1):
        raise ValueError(
            f"label must be 0, 1, or None, got {record.label!r}"
        )
