"""
Stage 2: Local Llama via Ollama.

"""


import re
import hashlib

import config
from pipeline.contract import WindowRecord


def _snippet(code):
    """Take the middle STAGE2_WINDOW_CHARS characters of the code.

    The consolidation step upstream already trims each window to a small
    intersection, so the vulnerable line is usually near the middle. This is
    just the safety cap on top of that (partner spec "What Stage 2 receives").
    """
    mid = len(code) // 2
    half = config.STAGE2_WINDOW_CHARS // 2
    return code[max(0, mid - half): mid + half]


def _build_prompt(snippet, vuln_type):
    vuln_name = config.VULN_TYPE_NAMES.get(vuln_type, vuln_type)
    return (
        f"You are a security expert. Analyze this Python code for {vuln_name} "
        f"vulnerabilities.\n"
        f"Reply with ONLY a decimal between 0.0 and 1.0 representing the "
        f"probability this code is vulnerable.\n"
        f"0.0 = definitely safe. 1.0 = definitely vulnerable.\n\n"
        f"CODE:\n{snippet}\n\nPROBABILITY:"
    )


def _parse_score(text):
    """Grab the first 0-1 number from the model's reply and clamp it."""
    match = re.search(r"\d*\.?\d+", text)
    if not match:
        return 0.5                      # no number -> uncertain, let it escalate
    score = float(match.group())
    if score > 1.0:                     # model gave a percentage like 95
        score = score / 100.0 if score <= 100.0 else 1.0
    return round(min(max(score, 0.0), 1.0), 4)


def _mock_score(record):
    """Deterministic fake score for testing without Ollama (config.OLLAMA_MOCK)."""
    h = hashlib.md5((record.vulnerability_type + record.code).encode("utf-8")).hexdigest()
    return round(int(h[:4], 16) / 0xFFFF, 4)


def predict(record: WindowRecord) -> None:
    """Run Llama on record.code and set record.stage2_score in place (0.0-1.0).

    Writes directly to the record and returns None, matching Stage 0/1.
    On any failure we fall back to 0.5 (uncertain) so a window is never
    silently dropped.
    """
    if config.OLLAMA_MOCK:
        record.stage2_score = _mock_score(record)
        return

    snippet = _snippet(record.code)
    prompt = _build_prompt(snippet, record.vulnerability_type)

    # Import here so the rest of the pipeline runs without the ollama package.
    try:
        import ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        record.stage2_score = _parse_score(response["message"]["content"])
    except Exception as err:  # noqa: BLE001 -- student project: log and fall back
        print(f"[stage2] WARNING: Ollama call failed ({err}). "
              f"Is `ollama serve` running and `{config.OLLAMA_MODEL}` pulled?")
        record.stage2_score = 0.5
