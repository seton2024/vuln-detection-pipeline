"""
demo_flask_app.py  —  A realistic (but intentionally vulnerable) Flask web application.

This file is DEMO DATA for the vulnerability detection pipeline.
It contains deliberate examples of all 7 vulnerability types so the pipeline
has something meaningful to find.

Vulnerability map:
  sql               → /login, /search (string-concatenated SQL queries)
  xss               → /profile, /comment (unescaped user input rendered in HTML)
  command_injection  → /ping, /convert (shell=True + user input)
  xsrf              → /transfer, /delete_account (no CSRF token check)
  path_disclosure    → /download, /read_log (unsanitised path from user input)
  open_redirect      → /logout, /oauth_callback (redirect target from query string)
  remote_code_execution → /debug, /template_render (eval / exec on user input)
"""

import os
import subprocess
import sqlite3

from flask import (
    Flask, request, redirect, render_template_string,
    session, g, send_file
)

app = Flask(__name__)
app.secret_key = "hardcoded_secret_key_do_not_ship"   # B105 — hardcoded password

DATABASE = "users.db"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


# ---------------------------------------------------------------------------
# SQL INJECTION  (vuln type: sql)
# ---------------------------------------------------------------------------

@app.route("/login", methods=["POST"])
def login():
    """Classic string-concatenated SQL — allows ' OR '1'='1 bypass."""
    username = request.form.get("username", "")
    password = request.form.get("password", "")

    # VULNERABLE: user-controlled values concatenated directly into query string
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    db = get_db()
    cursor = db.execute(query)          # SQL injection here
    user = cursor.fetchone()

    if user:
        session["user"] = username
        return "Logged in"
    return "Invalid credentials", 401


@app.route("/search")
def search():
    """Second SQL injection — search endpoint with %LIKE% injection."""
    term = request.args.get("q", "")

    # VULNERABLE: term injected into LIKE clause without parameterisation
    sql = "SELECT id, title FROM posts WHERE title LIKE '%" + term + "%'"
    results = get_db().execute(sql).fetchall()
    return str(results)


# ---------------------------------------------------------------------------
# CROSS-SITE SCRIPTING  (vuln type: xss)
# ---------------------------------------------------------------------------

@app.route("/profile")
def profile():
    """Reflects the 'name' query param directly into HTML — stored XSS via template."""
    name = request.args.get("name", "Anonymous")

    # VULNERABLE: name is not escaped; attacker can inject <script>...</script>
    html = "<html><body><h1>Hello, " + name + "!</h1></body></html>"
    return render_template_string(html)


@app.route("/comment", methods=["POST"])
def post_comment():
    """Stores a comment and renders it back without escaping."""
    comment = request.form.get("comment", "")

    # VULNERABLE: comment rendered unsanitised — DOM XSS when retrieved
    template = "<html><body><p>Your comment: {{ comment|safe }}</p></body></html>"
    return render_template_string(template, comment=comment)


# ---------------------------------------------------------------------------
# COMMAND INJECTION  (vuln type: command_injection)
# ---------------------------------------------------------------------------

@app.route("/ping")
def ping():
    """Network diagnostic endpoint — passes user host directly to shell."""
    host = request.args.get("host", "localhost")

    # VULNERABLE: shell=True + unsanitised host → OS command injection
    result = subprocess.run(
        "ping -c 1 " + host,
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@app.route("/convert")
def convert_file():
    """File format conversion — passes filename to ImageMagick via shell."""
    filename = request.args.get("file", "")
    output   = request.args.get("out",  "output.png")

    # VULNERABLE: both filename and output are user-controlled in a shell call
    os.system("convert " + filename + " " + output)
    return f"Converted {filename} → {output}"


# ---------------------------------------------------------------------------
# CROSS-SITE REQUEST FORGERY  (vuln type: xsrf)
# ---------------------------------------------------------------------------

@app.route("/transfer", methods=["POST"])
def transfer_funds():
    """Bank-style transfer — no CSRF token required, so any page can POST here."""
    if "user" not in session:
        return "Not logged in", 403

    amount    = request.form.get("amount", "0")
    recipient = request.form.get("to",     "")

    # VULNERABLE: no CSRF token check — any authenticated cross-origin POST succeeds
    db = get_db()
    db.execute(
        "UPDATE accounts SET balance = balance - ? WHERE user = ?",
        (amount, session["user"])
    )
    db.execute(
        "UPDATE accounts SET balance = balance + ? WHERE user = ?",
        (amount, recipient)
    )
    db.commit()
    return f"Transferred {amount} to {recipient}"


@app.route("/delete_account", methods=["POST"])
def delete_account():
    """Account deletion — no CSRF protection, destructive action."""
    if "user" not in session:
        return "Not logged in", 403

    # VULNERABLE: no origin check, no CSRF token — attacker can delete accounts
    username = session["user"]
    get_db().execute("DELETE FROM users WHERE username = ?", (username,))
    get_db().commit()
    session.clear()
    return "Account deleted"


# ---------------------------------------------------------------------------
# PATH DISCLOSURE  (vuln type: path_disclosure)
# ---------------------------------------------------------------------------

@app.route("/download")
def download_file():
    """File download endpoint — path traversal allows reading arbitrary files."""
    filename = request.args.get("file", "")

    # VULNERABLE: no sanitisation → ../../etc/passwd works
    file_path = os.path.join("/var/app/uploads", filename)
    return send_file(file_path)


@app.route("/read_log")
def read_log():
    """Admin log viewer — reads server-side log files, path from user input."""
    log_name = request.args.get("log", "app.log")

    # VULNERABLE: directory traversal; also discloses server-side path on error
    full_path = "/var/log/app/" + log_name
    try:
        with open(full_path) as f:
            return f.read()
    except FileNotFoundError:
        # VULNERABLE: returns the full server path in the error message
        return f"File not found: {full_path}", 404


# ---------------------------------------------------------------------------
# OPEN REDIRECT  (vuln type: open_redirect)
# ---------------------------------------------------------------------------

@app.route("/logout")
def logout():
    """Logout — redirects to a 'next' URL without validation."""
    session.clear()
    next_url = request.args.get("next", "/")

    # VULNERABLE: attacker sets ?next=https://evil.com and user is redirected there
    return redirect(next_url)


@app.route("/oauth_callback")
def oauth_callback():
    """OAuth return handler — passes redirect_uri from query string directly."""
    code         = request.args.get("code", "")
    redirect_uri = request.args.get("redirect_uri", "/dashboard")

    # VULNERABLE: redirect target comes from attacker-controlled query param
    # (should validate against a whitelist of known URIs)
    if code:
        session["authenticated"] = True
        return redirect(redirect_uri)
    return "Auth failed", 400


# ---------------------------------------------------------------------------
# REMOTE CODE EXECUTION  (vuln type: remote_code_execution)
# ---------------------------------------------------------------------------

@app.route("/debug")
def debug_eval():
    """Debug endpoint (left open by accident) — executes arbitrary Python."""
    expr = request.args.get("expr", "")

    # VULNERABLE: eval() on user-controlled input = arbitrary code execution
    try:
        result = eval(expr)             # B307 — eval of untrusted input
        return str(result)
    except Exception as e:
        return str(e), 400


@app.route("/template_render")
def custom_template():
    """'Custom template' feature — compiles and runs user-supplied Python code."""
    user_code = request.form.get("code", "")

    # VULNERABLE: exec() on POST body = full server takeover
    local_vars = {}
    exec(user_code, {}, local_vars)     # B102 — exec of untrusted input
    output = local_vars.get("output", "")
    return str(output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # VULNERABLE: debug=True in production exposes Werkzeug debugger (RCE)
    app.run(debug=True, host="0.0.0.0", port=5000)
