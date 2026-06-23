"""
Sliding window logic for code analysis.

this module is used for inference on NEW code files (not from VUDENC). For training data,
use loader.py instead.

Two windowing modes:
  - line-based  : extract_windows()       — original, window_size in lines
  - char-based  : extract_char_windows()  — matches VUDENC training distribution
                                            (200-char windows, configurable char stride)

The model was trained on ~205-char windows from VUDENC. Prefer extract_char_windows()
at inference so the model sees the same text size it was trained on.
"""

from typing import List

from pipeline.contract import WindowRecord
from config import VULN_TYPES, VUDENC_BLOCK_LENGTH, VUDENC_BLOCK_STEP


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


def extract_char_windows(
    code: str,
    file_path: str,
    vuln_type: str,
    char_length: int = VUDENC_BLOCK_LENGTH,
    char_stride: int = 50,
) -> List[WindowRecord]:
    """Extract windows using the same character-based method as VUDENC training data.

    This matches the ~200-char window size the model was trained on.
    char_stride=50 ≈ one line of Python, giving a similar window count to the
    10-line line-based approach without the size mismatch.

    Args:
        char_length: window size in characters (default: VUDENC_BLOCK_LENGTH = 200).
        char_stride: how many characters to advance between windows (default: 50 ≈ 1 line).
    """
    from data.vudenc_blocks import getblocks

    if not code.strip():
        return []

    # No known bad positions at inference time — all windows start unlabeled.
    blocks = getblocks(code, [], char_stride, char_length)
    return [
        WindowRecord(file=file_path, vulnerability_type=vuln_type, code=snippet)
        for snippet, _ in blocks
        if snippet and snippet.strip()
    ]


def extract_all_vuln_types_char(
    code: str,
    file_path: str,
    char_length: int = VUDENC_BLOCK_LENGTH,
    char_stride: int = 50,
) -> List[WindowRecord]:
    """Character-based extraction for ALL 7 vulnerability types.

    Use this instead of extract_all_vuln_types() when the model was trained on
    VUDENC data (200-char windows). The two functions have the same call site
    signature except window_size/stride are replaced by char_length/char_stride.
    """
    all_records = []
    for vuln_type in VULN_TYPES:
        windows = extract_char_windows(
            code=code,
            file_path=file_path,
            vuln_type=vuln_type,
            char_length=char_length,
            char_stride=char_stride,
        )
        all_records.extend(windows)
    return all_records
