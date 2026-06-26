"""
Stage 3: Claude Haiku via Batch API.

"""


import os
import re
import json
from pathlib import Path

import config

# Load .env from the project root so ANTHROPIC_API_KEY is available without
# manually exporting it in every shell session.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass
from pipeline.contract import WindowRecord

# Secret scrubbing lives in Person A's (done) Bandit module. Fall back to a
# no-op if it isn't present yet, so my file still imports during parallel work.
try:
    from pipeline.stage0_bandit import scrub_secrets
except ImportError:
    def scrub_secrets(code):
        print("[stage3] NOTE: pipeline.stage0_bandit.scrub_secrets not found yet "
              "-- sending code unscrubbed. Wire this up before any real run.")
        return code


# The fixed instruction part of the prompt. It never changes between calls, so
# we mark it for prompt caching to cut cost (only the code differs each call).
SYSTEM_PROMPT = """You are a security code reviewer specializing in Python vulnerabilities.
You will be given a code snippet and a vulnerability type to check for.
Respond ONLY with valid JSON matching this schema exactly. Do NOT wrap it in
markdown code fences and do NOT add any text before or after the JSON:
{
  "verdict": "vulnerable" | "not_vulnerable",
  "findings": [
    {
      "line_in_window": <int, 1-indexed>,
      "code": "<the exact bad line — escape any double-quote characters as \\\" >",
      "reason": "<why this is a vulnerability>",
      "fix": "<concrete fix suggestion>"
    }
  ]
}
If not vulnerable, return an empty findings array.
IMPORTANT: all string values must be valid JSON strings. Escape every double-quote
character that appears inside a string value as \\\" (backslash-quote)."""


def _extract_json(text: str) -> dict:
    """Parse the model reply into a dict, tolerating markdown fences / stray prose.

    Strategy:
      1. Strip leading markdown fence if present.
      2. Try standard JSON parsing (fast path).
      3. If that fails (typically unescaped " inside a "code" field), fall back
         to regex extraction of the verdict and per-finding reason/fix — skipping
         the malformed "code" string that caused the failure.
    """
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)

    start = text.find("{")
    json_text = text[start:] if start != -1 else text

    # Fast path: well-formed JSON
    try:
        obj, _ = json.JSONDecoder().raw_decode(json_text)
        return obj
    except json.JSONDecodeError:
        pass

    # Fallback: extract verdict + safe fields via regex when the "code" field
    # contains unescaped double-quotes that break the JSON parser.
    verdict_match = re.search(r'"verdict"\s*:\s*"(vulnerable|not_vulnerable)"', json_text)
    if not verdict_match:
        raise json.JSONDecodeError("no verdict found", json_text, 0)

    verdict = verdict_match.group(1)
    findings = []
    for m in re.finditer(
        r'"line_in_window"\s*:\s*(\d+)'
        r'.*?"reason"\s*:\s*"([^"]*)"'
        r'.*?"fix"\s*:\s*"([^"]*)"',
        json_text, re.DOTALL
    ):
        findings.append({
            "line_in_window": int(m.group(1)),
            "code": "",   # omitted — unescaped quotes made this field unparseable
            "reason": m.group(2),
            "fix": m.group(3),
        })

    return {"verdict": verdict, "findings": findings}


def predict(record: WindowRecord) -> None:
    """Adjudicate record.code with Claude Haiku, writing results in place.

    Sets record.stage3_verdict ("vulnerable" / "not_vulnerable") and
    record.stage3_findings (list of dicts; [] when not vulnerable).
    Returns None. On a disabled toggle or any error it leaves the verdict as
    None ("not run").
    """
    # Safety guard: never call the API when the toggle is off (the runner also
    # checks this, but double-checking here protects against accidental charges).
    if not config.STAGE3_ENABLED:
        print("[stage3] STAGE3_ENABLED is off -- skipping Claude call. "
              "Set env STAGE3_ENABLED=1 to enable.")
        return

    safe_code = scrub_secrets(record.code)   # ALWAYS scrub before sending out

    api_key = os.environ.get(config.CLAUDE_API_KEY_ENV)
    if not api_key:
        print(f"[stage3] ERROR: env var {config.CLAUDE_API_KEY_ENV} is not set.")
        return

    try:
        import anthropic
    except ImportError:
        print("[stage3] ERROR: `anthropic` not installed (pip install anthropic).")
        return

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = (
        f"Vulnerability type to check: {record.vulnerability_type}\n\n"
        f"Code snippet:\n{safe_code}"
    )

    # NOTE on cost: for the final full run, switch this to the Batch API for a
    # 50% discount (no live-demo deadline). Kept synchronous here for clarity.
    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # cache the fixed part
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        result = _extract_json(raw)
        record.stage3_verdict = result["verdict"]
        record.stage3_findings = result.get("findings", [])
    except (json.JSONDecodeError, KeyError, IndexError) as err:
        # Claude returned something we couldn't parse -> mark uncertain.
        snippet = (raw[:200] if "raw" in dir() and isinstance(raw, str) else "<no text>")
        print(f"[stage3] WARNING: could not parse Claude reply ({err}). "
              f"First 200 chars: {snippet!r}")
        record.stage3_verdict = None
        record.stage3_findings = None
    except Exception as err:  # noqa: BLE001 -- network/API error: log and bail
        print(f"[stage3] WARNING: Claude call failed ({err}).")
        record.stage3_verdict = None
        record.stage3_findings = None
