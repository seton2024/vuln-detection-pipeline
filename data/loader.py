"""
VUDENC data loader.

  PHASE 1 — Data preparation (run once):
      -python scripts/setup_data.py
      -Downloads the VUDENC plain_* files from Zenodo, 
        -turns each one into labeled code windows, 
        -splits 70/15/15, 
        -saves the result to data/processed/

  PHASE 2 — Training / inference (run any time):
      Reads directly from data/processed/.

"""

import os
import json
from typing import Optional

from sklearn.model_selection import train_test_split

from pipeline.contract import WindowRecord
from config import (
    VUDENC_DATA_DIR,
    VULN_TYPES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    VUDENC_BLOCK_LENGTH,
    VUDENC_BLOCK_STEP,
)
from data.vudenc_blocks import findpositions, getblocks

#where the processed splits will be saved
PROCESSED_DIR = os.path.join("data", "processed")

# Skip change-hunks with an unusually large number of vulnerable lines — these are big refactors/renames, not focused security fixes, and add noise.
MAX_BADPARTS_PER_CHANGE = 20



# Public API — Phase 2 (called by training code, reads from data/processed)


def load_vudenc(
    vuln_type: str,
) -> tuple[list[WindowRecord], list[WindowRecord], list[WindowRecord]]:
    #Load the pre-processed data for a single vulnerability type.

    if vuln_type not in VULN_TYPES:
        raise ValueError(f"Unknown vuln type: {vuln_type!r}. Must be one of {VULN_TYPES}")

    result = _load_processed(vuln_type)
    if result is None:
        raise FileNotFoundError(
            f"No processed data found for {vuln_type!r} in {PROCESSED_DIR!r}.\n"
            f"Run:  python scripts/setup_data.py"
        )
    return result


def load_all_types() -> dict[str, tuple[list, list, list]]:

    #Load pre-processed data for all 7 vulnerability types.

    result = {}
    for vuln_type in VULN_TYPES:
        print(f"[loader] Loading {vuln_type!r}...")
        result[vuln_type] = load_vudenc(vuln_type)
    return result



# Public API — Phase 1 (called only by scripts/setup_data.py)


def prepare_and_save(vuln_type: str) -> None:
    """
    Process one vuln type's raw plain_* file into labeled context windows,
    split 70/15/15, and save to data/processed/.

    """
    if vuln_type not in VULN_TYPES:
        raise ValueError(f"Unknown vuln type: {vuln_type!r}")

    # Skip if already processed
    if _processed_exists(vuln_type):
        print(f"[loader] {vuln_type!r} already processed — skipping")
        return

    records = _load_raw_records(vuln_type)
    if len(records) < 3:
        print(f"[loader] WARNING: Only {len(records)} windows for {vuln_type!r} — skipping")
        return

    train, val, test = _split(records)
    _save_processed(vuln_type, train, val, test)
    _print_split_stats(vuln_type, train, val, test)


def prepare_all_types() -> None:
    #Run prepare_and_save for all 7 vulnerability types.
    for vuln_type in VULN_TYPES:
        print(f"\n[loader] Preparing {vuln_type!r}...")
        prepare_and_save(vuln_type)



# Processed file I/O


def _processed_path(vuln_type: str, split: str) -> str:
    return os.path.join(PROCESSED_DIR, f"{vuln_type}_{split}.json")


def _processed_exists(vuln_type: str) -> bool:
    #True if all three split files exist for this vuln type.
    return all(
        os.path.exists(_processed_path(vuln_type, s))
        for s in ("train", "val", "test")
    )


def _save_processed(vuln_type: str, train, val, test) -> None:
    #Serialize and save the three splits to data/processed/.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for split_name, records in [("train", train), ("val", val), ("test", test)]:
        path = _processed_path(vuln_type, split_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
    print(f"[loader] Saved {vuln_type!r} -> {PROCESSED_DIR}/")


def _load_processed(vuln_type: str):
    #Load the three splits from data/processed/. Returns None if missing.
    if not _processed_exists(vuln_type):
        return None
    splits = {}
    for split_name in ("train", "val", "test"):
        path = _processed_path(vuln_type, split_name)
        with open(path, "r", encoding="utf-8") as f:
            splits[split_name] = [WindowRecord.from_dict(d) for d in json.load(f)]
    return splits["train"], splits["val"], splits["test"]



# Raw VUDENC processing (Phase 1 only)

def _load_raw_records(vuln_type: str) -> list[WindowRecord]:
    """
    Read the raw plain_<type> JSON file and turn it into labeled WindowRecords.

    The plain_* file structure is:
        {repo_url: {commit_sha: {keyword, diff, msg,
                                 files: {filename: {source, sourceWithComments,
                                                    changes: [{badparts, goodparts, ...}]}}}}}
    """
    plain_path = os.path.join(VUDENC_DATA_DIR, f"plain_{vuln_type}")
    if not os.path.exists(plain_path):
        print(f"[loader] Raw file not found: {plain_path!r}")
        return []

    print(f"[loader] Reading {plain_path!r}")
    try:
        with open(plain_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[loader] ERROR reading {plain_path!r}: {e}")
        return []

    return _parse_commits_to_records(data, vuln_type)


def _parse_commits_to_records(data: dict, vuln_type: str) -> list[WindowRecord]:
    """
    Walk the repo -> commit -> file structure and build context windows from
    each file's full source using the VUDENC block method.
    """
    records: list[WindowRecord] = []
    n_repos = len(data)
    n_commits = 0
    n_files = 0

    for repo_url, commits in data.items():
        for sha, entry in commits.items():
            n_commits += 1
            files = entry.get("files", {})
            if not files:
                continue

            for filename, file_info in files.items():
                # Prefer the comment-stripped 'source'; fall back to the raw one.
                source = file_info.get("source") or file_info.get("sourceWithComments")
                if not source or not source.strip():
                    continue

                changes = file_info.get("changes", [])
                file_records = _extract_records_from_file(
                    source, changes, vuln_type, filename or repo_url
                )
                if file_records:
                    n_files += 1
                    records.extend(file_records)

    print(
        f"[loader] {vuln_type!r}: {n_repos} repos, {n_commits} commits, "
        f"{n_files} files with source -> {len(records)} windows"
    )
    return records


def _extract_records_from_file(
    source: str, changes: list, vuln_type: str, filename: str
) -> list[WindowRecord]:
    """
    For one file: collect its vulnerable lines (badparts), locate them in the
    source, slide context windows across the whole file, and wrap each window
    as a WindowRecord (label 1 = vulnerable, 0 = safe).
    """
    # Gather the vulnerable lines across all change-hunks in this file.
    allbadparts: list[str] = []
    for change in changes:
        badparts = change.get("badparts", [])
        # Skip hunks with too many badparts (big refactors, not focused fixes).
        if len(badparts) < MAX_BADPARTS_PER_CHANGE:
            for bad in badparts:
                if bad and bad.strip():
                    allbadparts.append(bad)

    # Locate those lines inside the source (whitespace/comment tolerant).
    badpositions = findpositions(allbadparts, source)

    # Slide context windows across the file, labeling by overlap with badparts.
    blocks = getblocks(source, badpositions, VUDENC_BLOCK_STEP, VUDENC_BLOCK_LENGTH)

    records = []
    for snippet, label in blocks:
        if not snippet or not snippet.strip():
            continue
        records.append(WindowRecord(
            file=filename,
            vulnerability_type=vuln_type,
            code=snippet,
            label=label,   # ground truth: 1 = vulnerable, 0 = safe
        ))
    return records



# Splitting / stats


def _split(records: list) -> tuple[list, list, list]:
    #Apply 70/15/15  split
    labels = [r.label for r in records]

    train, temp = train_test_split(
        records,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=_safe_stratify(labels),
    )
    temp_labels = [r.label for r in temp]
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val, test = train_test_split(
        temp,
        test_size=relative_test,
        random_state=42,
        stratify=_safe_stratify(temp_labels),
    )
    return train, val, test


def _safe_stratify(labels: list) -> Optional[list]:
    #Return labels for stratification only if safe to do so.
    if len(set(labels)) < 2:
        return None
    if min(labels.count(l) for l in set(labels)) < 2:
        return None
    return labels


def _print_split_stats(vuln_type, train, val, test) -> None:
    total = len(train) + len(val) + len(test)
    vuln_count = sum(1 for r in (train + val + test) if r.label == 1)
    pct = (100 * vuln_count / total) if total else 0.0
    print(
        f"[loader] {vuln_type!r}: total={total} "
        f"({vuln_count} vulnerable, {pct:.1f}%)  "
        f"train={len(train)} val={len(val)} test={len(test)}"
    )
