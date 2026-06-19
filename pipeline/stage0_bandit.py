"""
Stage 0: Bandit static analysis runner and secrets scrubber.

  1. Run the Bandit security scanner on a code window and set bandit_flag.
  2. Scrub hardcoded secrets from code before it ever reaches Stage 3 (Claude API)
"""

import subprocess
import sys
import tempfile
import os
import re
import json
from typing import Optional

from pipeline.contract import WindowRecord



# Mapping: which Bandit issue codes are relevant for each vulnerability type.
# Bandit assigns each of its checks a short ID
# flag only for relevan voulnerability

BANDIT_RELEVANT_CODES = {
    "sql":                   ["B608"],                              # possible SQL injection
    "xss":                   ["B703", "B704"],                      # Django/Jinja template injection
    "command_injection":     ["B602", "B603", "B604", "B605", "B606", "B607"],  # subprocess/shell calls
    "xsrf":                  [],                                    # no XSRF checks in Bandit
    "path_disclosure":       ["B106", "B107"],                      # hardcoded file paths / passwords
    "open_redirect":         [],                                    # no open redirect checks in Bandit
    "remote_code_execution": ["B102", "B307"],                      # exec(), eval()
}

# Bandit  ID 4 hardcoded secrets.
SECRET_TEST_IDS = ["B105", "B106", "B107", "B108"]

# Patterns for: API keys, passwords, tokens
_SECRET_PATTERNS = [
    # Matches: password = "abc123", token = "xyz", SECRET_KEY = "..."
    re.compile(
        r"""(?ix)
        (password|passwd|secret|token|api[_\-]?key|auth[_\-]?key|access[_\-]?key
        |private[_\-]?key|client[_\-]?secret|credentials?)
        \s*=\s*
        (['"][^'"]{8,}['"])   # a quoted string of at least 8 chars
        """,
        re.VERBOSE | re.IGNORECASE,
    ),
    # Matches long alphanumeric strings that look like keys/tokens (20+ chars)
    re.compile(r"""(['"])[A-Za-z0-9+/=_\-]{20,}\1"""),
]

#Public function for Stage 0

def run_bandit(record: WindowRecord) -> bool:
    """
    Run Bandit on record.code and update record.bandit_flag in-place.

    Returns True if Bandit found a security issue relevant to
    record.vulnerability_type.
    """
    temp_path = _write_temp_py(record.code) #work with tem copy of record.code
    try:
        result = _run_bandit_on_file(temp_path)
        issues = _parse_bandit_json(result)
        flag = _has_relevant_issue(issues, record.vulnerability_type) #checks for relevant issue
        record.bandit_flag = flag #write true or false to record.bandit_flag
        return flag
    finally:
        # Always delete the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def scrub_secrets(code: str) -> str:
    #Remove hardcoded secrets from code before sending it to Claude (Stage 3).
    
    lines = code.splitlines(keepends=True)
    flagged_lines = _find_secret_lines_via_bandit(code)# Bandit with only B105-B108

    scrubbed_lines = []
    for i, line in enumerate(lines):
        lineno = i + 1  # Bandit reports 1-based line numbers
        if lineno in flagged_lines:
            line = _redact_string_literals(line)
        scrubbed_lines.append(line)

    scrubbed = "".join(scrubbed_lines)

    # Also apply regex-based scrubbing as a second pass
    scrubbed = _apply_regex_scrubbing(scrubbed)

    return scrubbed

#private functions for Stage 0

def _write_temp_py(code: str) -> str:

    #Write code to a temporary .py file and return its path.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        return f.name


def _run_bandit_on_file(file_path: str) -> str:
    #Run `bandit -f json -q <file_path>` and return its stdout as a string.
  
    try:
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-f", "json", "-q", file_path],
            capture_output=True,
            text=True,
            timeout=30,  # never hang forever
        )
        # Bandit exit codes: 0 = no issues, 1 = issues found, 2 = error
        if result.returncode == 2:
            raise RuntimeError(f"Bandit returned error: {result.stderr.strip()}")
        return result.stdout
    except FileNotFoundError:
        raise RuntimeError(
            "Bandit is not installed. Run: pip install bandit"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Bandit timed out after 30 seconds")


def _parse_bandit_json(output: str) -> list[dict]:
    """
    Parse Bandit's JSON output and return the list of issue dictionaries.
    Returns an empty list if the output is empty or not valid JSON.
    """
    if not output or not output.strip():
        return []
    try:
        data = json.loads(output)
        return data.get("results", [])
    except json.JSONDecodeError:
        # Bandit sometimes emits non-JSON warnings before the JSON block
        # Try to find the JSON portion
        start = output.find("{")
        if start == -1:
            return []
        try:
            data = json.loads(output[start:])
            return data.get("results", [])
        except json.JSONDecodeError:
            return []


def _has_relevant_issue(issues: list[dict], vuln_type: str) -> bool:
    """
    Check whether any Bandit issue is relevant to the given vulnerability type.

    For types with no mapping (xsrf, open_redirect), we fall back to:
    any HIGH severity issue = True.
    """
    relevant_codes = BANDIT_RELEVANT_CODES.get(vuln_type, [])

    if relevant_codes:
        # Check if any issue's test_id is in our relevant list
        return any(issue.get("test_id") in relevant_codes for issue in issues)
    else:
        # Fallback: flag if Bandit found anything HIGH severity
        return any(issue.get("issue_severity") == "HIGH" for issue in issues)


def _find_secret_lines_via_bandit(code: str) -> set[int]:

    #Run Bandit with only the secret-detection tests 
    flagged = set()
    temp_path = _write_temp_py(code)
    try:
        # -t B105,B106,... tells Bandit to run ONLY those specific tests
        test_ids = ",".join(SECRET_TEST_IDS)
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-f", "json", "-q", "-t", test_ids, temp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        issues = _parse_bandit_json(result.stdout)
        for issue in issues:
            lineno = issue.get("line_number")
            if lineno is not None:
                flagged.add(lineno)
    except Exception:
        pass  # If Bandit fails here, we still apply regex scrubbing below
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return flagged


def _redact_string_literals(line: str) -> str:

    #Replace all quoted string literals in a line with '<REDACTED>'.

    line = re.sub(r'"[^"]*"', '"<REDACTED>"', line)
    line = re.sub(r"'[^']*'", "'<REDACTED>'", line)
    return line


def _apply_regex_scrubbing(code: str) -> str:
    #Apply the regex-based secret patterns to the full code string.
    for pattern in _SECRET_PATTERNS:
        code = pattern.sub(
            lambda m: m.group(0).split("=")[0] + '= "<REDACTED>"'
            if "=" in m.group(0)
            else '"<REDACTED>"',
            code,
        )
    return code


# ---------------------------------------------------------------------------
# Demo — run this file directly to see both functions in action
# How to run: & "C:\Users\seton\anaconda3\python.exe" -m  pipeline.stage0_bandit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEMO_CODE = """
import sqlite3

API_KEY = "sk-abc123verylongsecretkey9876543210"

def get_user(username):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchall()
"""

    print("=== Stage 0 Demo ===\n")
    print("Input code:")
    print(DEMO_CODE)

    # Test run_bandit
    from pipeline.contract import WindowRecord
    record = WindowRecord(
        file="demo.py",
        vulnerability_type="sql",
        code=DEMO_CODE,
    )
    flag = run_bandit(record)
    print(f"\nBandit flag for SQL injection: {flag}")  # Should be True

    # Test scrub_secrets
    scrubbed = scrub_secrets(DEMO_CODE)
    print("\nScrubbed code (API key should be redacted):")
    print(scrubbed)
