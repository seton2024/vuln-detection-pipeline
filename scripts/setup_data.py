"""
ONE-TIME data preparation script.

What it does:
  1. Downloads the 7 plain_* files from Zenodo (~54 MB total)
  2. Processes each into WindowRecords and applies 70/15/15 split
  3. Saves results to data/processed/ (21 JSON files)
  4. Commits data/processed/ to git → partner just clones and skips this step

After this runs, VUDENC is never needed again.
Training and inference read only from data/processed/.
"""

import sys
import os
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import VUDENC_DATA_DIR, VULN_TYPES
from data.loader import prepare_all_types, PROCESSED_DIR

# Zenodo direct download URLs — one plain_* file per vuln type (~54 MB total)
ZENODO_URLS = {
    "sql":                  "https://zenodo.org/records/3559841/files/plain_sql?download=1",
    "xss":                  "https://zenodo.org/records/3559841/files/plain_xss?download=1",
    "xsrf":                 "https://zenodo.org/records/3559841/files/plain_xsrf?download=1",
    "command_injection":    "https://zenodo.org/records/3559841/files/plain_command_injection?download=1",
    "open_redirect":        "https://zenodo.org/records/3559841/files/plain_open_redirect?download=1",
    "path_disclosure":      "https://zenodo.org/records/3559841/files/plain_path_disclosure?download=1",
    "remote_code_execution":"https://zenodo.org/records/3559841/files/plain_remote_code_execution?download=1",
}


def download_plain_files() -> bool:
    """Download all plain_* files from Zenodo into data/vudenc/."""
    os.makedirs(VUDENC_DATA_DIR, exist_ok=True)

    for vuln_type, url in ZENODO_URLS.items():
        dest = os.path.join(VUDENC_DATA_DIR, f"plain_{vuln_type}")

        if os.path.exists(dest):
            print(f"  [skip] {dest} already exists")
            continue

        print(f"  Downloading {vuln_type}...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"  plain_{vuln_type}"
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

        except Exception as e:
            print(f"  [ERROR] Failed to download {vuln_type}: {e}")
            # Retry once
            try:
                print(f"  Retrying {vuln_type}...")
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(response.content)
                print(f"  [ok] {vuln_type} downloaded on retry")
            except Exception as e2:
                print(f"  [FAIL] {vuln_type} failed after retry: {e2}")
                return False

    return True


def main() -> int:
    print("=" * 60)
    print("VUDENC Data Preparation")
    print("=" * 60)

    # Step 1: Download raw files from Zenodo
    print("\nStep 1: Downloading plain_* files from Zenodo...")
    if not download_plain_files():
        print("\n[ERROR] Download failed. Check your internet connection.")
        return 1

    # Step 2: Process all 7 types → data/processed/
    print("\nStep 2: Processing into WindowRecords and splitting 70/15/15...")
    try:
        prepare_all_types()
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Step 3: Summary
    print(f"\n{'=' * 60}")
    print(f"Done. Processed data saved to: {PROCESSED_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
