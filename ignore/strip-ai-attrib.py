#!/usr/bin/env python3
"""Rewrite a commit message file in place: drop Co-Authored-By trailer lines,
re-insert exactly one blank line after the subject, trim trailing blanks."""
import sys, os
path = sys.argv[1]
with open(path) as f:
    lines = f.read().split("\n")
lines = [l for l in lines if not l.strip().startswith("Co-Authored-By:")]
while lines and lines[-1] == "":
    lines.pop()
if len(lines) > 1:
    while lines[1] == "":
        lines.pop(1)
    lines.insert(1, "")
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
