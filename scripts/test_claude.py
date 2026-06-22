"""
Isolated smoke test for Stage 3 (Claude Haiku).

Bypasses the cascade gating and calls Claude directly on one obviously-vulnerable
snippet, so you can confirm your API key, model, and JSON parsing all work before
running the full pipeline.

USAGE (PowerShell)
    $env:ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/test_claude.py
"""

import os
import sys
import json

# Enable Stage 3 just for this process (config reads this at import time).
os.environ["STAGE3_ENABLED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from pipeline.contract import WindowRecord
from pipeline import stage3_claude

VULN_CODE = '''@app.route("/login", methods=["POST"])
def login():
    user = request.form["username"]
    pw = request.form["password"]
    query = "SELECT * FROM users WHERE name='" + user + "' AND pw='" + pw + "'"
    cur = get_db().execute(query)
    return "ok" if cur.fetchone() else "denied"'''


def main() -> int:
    key_var = config.CLAUDE_API_KEY_ENV  # "ANTHROPIC_API_KEY"
    if not os.environ.get(key_var):
        print(f"ERROR: no API key found. Set it first:\n"
              f'    PowerShell:  $env:{key_var}="sk-ant-..."\n'
              f"  ...or put it in a .env file (copy env.example .env).")
        return 1

    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("ERROR: anthropic not installed. Run: pip install -r requirements.txt")
        return 1

    rec = WindowRecord(file="test", vulnerability_type="sql", code=VULN_CODE)
    print(f"Calling {config.CLAUDE_MODEL} on a known SQL-injection snippet ...\n")
    stage3_claude.predict(rec)

    print("verdict :", rec.stage3_verdict)
    print("findings:", json.dumps(rec.stage3_findings, indent=2))

    if rec.stage3_verdict == "vulnerable":
        print("\nPASS — Claude is wired up and flagged the vulnerability.")
        return 0
    if rec.stage3_verdict is None:
        print("\nFAIL — call did not complete (see warning above: key/network/model?).")
        return 1
    print("\nDONE — call worked, but verdict was not 'vulnerable'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
