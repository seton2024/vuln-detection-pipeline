"""
Stage 2: Local Llama via Ollama.

Returns a probability score AND, when the score exceeds STAGE2_SAFE_THRESHOLD,
a list of line-level findings (same schema as Stage 3).
"""

import hashlib
import json
import re

import config
from pipeline.contract import WindowRecord


def _snippet(code):
    """Take the middle STAGE2_WINDOW_CHARS characters of the code."""
    mid = len(code) // 2
    half = config.STAGE2_WINDOW_CHARS // 2
    return code[max(0, mid - half): mid + half]


def _build_prompt(snippet, vuln_type):
    vuln_name = config.VULN_TYPE_NAMES.get(vuln_type, vuln_type)
    return (
        f"You are a security expert. Analyze this Python code for {vuln_name} vulnerabilities.\n"
        f"Reply with ONLY valid JSON matching this schema exactly. "
        f"Do NOT wrap it in markdown fences and do NOT add any text before or after the JSON:\n"
        f"{{\n"
        f'  "probability": <float 0.0-1.0>,\n'
        f'  "findings": [\n'
        f'    {{\n'
        f'      "line_in_window": <int, 1-indexed>,\n'
        f'      "code": "<the exact bad line>",\n'
        f'      "reason": "<why this is a vulnerability>",\n'
        f'      "fix": "<concrete fix suggestion>"\n'
        f'    }}\n'
        f'  ]\n'
        f"}}\n"
        f"0.0 = definitely safe. 1.0 = definitely vulnerable. "
        f"If safe, return an empty findings array.\n\n"
        f"CODE:\n{snippet}"
    )


def _extract_json(text: str) -> dict:
    """Parse Llama reply into a dict, tolerating markdown fences / leading prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    start = text.find("{")
    if start == -1:
        return json.loads(text)
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def _parse_response(text: str):
    """Return (score, findings) from the model reply.

    Falls back to (0.5, []) on any parse error so a window is never silently dropped.
    """
    try:
        obj = _extract_json(text)
        raw_prob = obj.get("probability", 0.5)
        score = float(raw_prob)
        if score > 1.0:
            score = score / 100.0 if score <= 100.0 else 1.0
        score = round(min(max(score, 0.0), 1.0), 4)
        findings = obj.get("findings", []) or []
        return score, findings
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Old-style plain-number reply — extract the number directly
        match = re.search(r"\d*\.?\d+", text or "")
        if match:
            raw = float(match.group())
            score = raw / 100.0 if raw > 1.0 else raw
            score = round(min(max(score, 0.0), 1.0), 4)
        else:
            score = 0.5
        return score, []


def _mock_score(record) -> float:
    """Deterministic fake score for testing without Ollama."""
    h = hashlib.md5((record.vulnerability_type + record.code).encode("utf-8")).hexdigest()
    return round(int(h[:4], 16) / 0xFFFF, 4)


def _mock_findings(record, score) -> list:
    """Deterministic fake findings mirroring Stage 3 schema.

    Only produced when score > STAGE2_SAFE_THRESHOLD so the findings column
    is non-empty for windows the model thinks are suspicious.
    """
    if score <= config.STAGE2_SAFE_THRESHOLD:
        return []
    lines = (record.code or "").splitlines()
    vuln_name = config.VULN_TYPE_NAMES.get(record.vulnerability_type, record.vulnerability_type)
    # Pick the line most likely to be the offending one (heuristic: longest line)
    if lines:
        bad_idx = max(range(len(lines)), key=lambda i: len(lines[i]))
        bad_line = lines[bad_idx].strip()
        line_no = bad_idx + 1
    else:
        bad_line = "<empty>"
        line_no = 1
    return [{
        "line_in_window": line_no,
        "code": bad_line[:120],
        "reason": f"[mock] Potential {vuln_name} pattern detected",
        "fix": "Sanitize / parameterize this input before use",
    }]


def predict(record: WindowRecord) -> None:
    """Run Llama on record.code; set stage2_score and stage2_findings in place.

    Findings are only populated when score > STAGE2_SAFE_THRESHOLD.
    On any failure falls back to score=0.5, findings=[].
    """
    if config.OLLAMA_MOCK:
        score = _mock_score(record)
        record.stage2_score = score
        record.stage2_findings = _mock_findings(record, score)
        return

    snippet = _snippet(record.code)
    prompt = _build_prompt(snippet, record.vulnerability_type)

    try:
        import ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        score, findings = _parse_response(response["message"]["content"])
        record.stage2_score = score
        record.stage2_findings = findings if score > config.STAGE2_SAFE_THRESHOLD else []
    except Exception as err:  # noqa: BLE001
        print(f"[stage2] WARNING: Ollama call failed ({err}). "
              f"Is `ollama serve` running and `{config.OLLAMA_MODEL}` pulled?")
        record.stage2_score = 0.5
        record.stage2_findings = []
