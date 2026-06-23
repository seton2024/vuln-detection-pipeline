"""Sample input for the pipeline demo — clean, no vulnerabilities."""


def add(a, b):
    return a + b


def greet(name):
    return f"Hello, {name.strip()}!"


def total(items):
    return sum(int(x) for x in items)


def clamp(value, low, high):
    return max(low, min(value, high))
