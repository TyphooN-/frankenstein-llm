#!/usr/bin/env python3
import os
from pathlib import Path
import time

PGID = 37312
LOG = Path("/home/typhoon/git/frankenstein-llm/verification/glm53flash-local/download.log")


def members():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            if int(fields[4]) == PGID and fields[2] != "Z":
                found.append((int(entry.name), fields[2]))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            pass
    return found


while current := members():
    time.sleep(30)

final_line = ""
try:
    lines = LOG.read_text(errors="replace").splitlines()
    final_line = lines[-1] if lines else "download log empty"
except OSError as error:
    final_line = f"cannot read download log: {error}"
print(f"GLM download PGID {PGID} exited; {final_line}", flush=True)
