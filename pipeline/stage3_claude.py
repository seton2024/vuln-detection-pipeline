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
      "code": "<the exact bad line>",
      "reason": "<why this is a vulnerability>",
      "fix": "<concrete fix suggestion>"
    }
  ]
}
If not vulnerable, return an empty findings array."""


def _extract_json(text: str) -> dict:
    """Parse the model reply into a dict, tolerating markdown fences / stray prose.

    Claude often wraps the JSON in ```json ... ``` fences and sometimes adds an
    explanatory sentence AFTER the closing fence. We strip a leading fence, then
    decode just the first balanced {...} object and ignore any trailing text.
    """
    text = (text or "").strip()

    # Strip a leading markdown fence line (``` or ```json) if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)

    # Decode the first JSON object starting at the first "{"; raw_decode stops
    # at the end of that object and ignores trailing fences/prose.
    start = text.find("{")
    if start == -1:
        return json.loads(text)  # no object -> raise a clear JSONDecodeError
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


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
