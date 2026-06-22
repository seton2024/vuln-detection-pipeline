"""Sample input for the pipeline demo — contains deliberate vulnerabilities."""

import os
import sqlite3


def get_user(conn, username):
    cur = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cur.execute(query)
    return cur.fetchall()


def ping_host(user_input):
    os.system("ping " + user_input)


def render_greeting(name):
    return "<h1>Hello " + name + "</h1>"


def run_expr(expr):
    return eval(expr)


def safe_add(a, b):
    return a + b
