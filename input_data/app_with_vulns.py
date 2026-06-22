"""Sample input for the pipeline demo — contains deliberate vulnerabilities.

Lines that are TRULY vulnerable are tagged with a trailing `# VULN` marker.
run_pipeline_demo.py strips that marker before scanning (so the model never
sees it) and uses it only to draw the red ground-truth dots in the PNGs.
"""

import os
import sqlite3


def get_user(conn, username):
    cur = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"  # VULN
    cur.execute(query)  # VULN
    return cur.fetchall()


def ping_host(user_input):
    os.system("ping " + user_input)  # VULN


def render_greeting(name):
    return "<h1>Hello " + name + "</h1>"  # VULN


def run_expr(expr):
    return eval(expr)  # VULN


def safe_add(a, b):
    return a + b
