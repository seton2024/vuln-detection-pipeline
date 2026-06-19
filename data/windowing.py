"""
Sliding window logic for code analysis.

this module is used for inference on NEW code files (not from VUDENC). For training data,
use loader.py instead.
"""

from typing import List

from pipeline.contract import WindowRecord
from config import VULN_TYPES


def extract_windows(
    code: str,
    file_path: str,
    vuln_type: str,
    window_size: int = 10,
    stride: int = 1,
) -> List[WindowRecord]:
    """
    Args:
        code:        source code text.
        file_path:   Path to the source file.
        vuln_type:   Which vulnerability type to check for.
        window_size: (default 10).
        stride:      (default 1).

    Returns:
        A list of WindowRecord, one per window position.

    """
    if not code.strip():
        return []

    lines = code.splitlines()
    total_lines = len(lines)

    if window_size > total_lines:
        # File is smaller than one window — use the whole file as one window
        return [
            WindowRecord(
                file=file_path,
                vulnerability_type=vuln_type,
                code=code,
            )
        ]

    records = []
    start = 0
    while start + window_size <= total_lines:
        end = start + window_size
        window_lines = lines[start:end]
        window_code = "\n".join(window_lines)

        records.append(
            WindowRecord(
                file=file_path,
                vulnerability_type=vuln_type,
                code=window_code,
            )
        )
        start += stride

    return records


def extract_all_vuln_types(
    code: str,
    file_path: str,
    window_size: int = 10,
    stride: int = 1,
) -> List[WindowRecord]:
    """
    Extract windows for ALL 7 vulnerability types at once.

    The length is:  num_windows × 7  (one per vuln type).
    """
    all_records = []
    for vuln_type in VULN_TYPES:
        windows = extract_windows(
            code=code,
            file_path=file_path,
            vuln_type=vuln_type,
            window_size=window_size,
            stride=stride,
        )
        all_records.extend(windows)
    return all_records
