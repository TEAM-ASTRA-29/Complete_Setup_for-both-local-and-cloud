"""
backend/session.py
==================
Session save / load / delete helpers.
"""

import os, json, hashlib
from datetime import datetime


PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "saved_sessions")
os.makedirs(PERSIST_DIR, exist_ok=True)


def make_sid(name: str) -> str:
    return hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]


def save_session(s: dict):
    path = os.path.join(PERSIST_DIR, f"{s['id']}.json")
    with open(path, "w") as f:
        json.dump(s, f, default=str, indent=2)


def load_sessions() -> list:
    out = []
    for fn in sorted(os.listdir(PERSIST_DIR)):
        if fn.endswith(".json") and not fn.startswith("_"):
            try:
                with open(os.path.join(PERSIST_DIR, fn)) as f:
                    out.append(json.load(f))
            except Exception:
                pass
    return out


def delete_session(sid: str):
    path = os.path.join(PERSIST_DIR, f"{sid}.json")
    if os.path.exists(path):
        os.remove(path)
