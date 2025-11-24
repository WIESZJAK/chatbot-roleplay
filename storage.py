"""Utilities for working with chat storage on disk."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import DEFAULT_CHAT_ID, DEFAULT_PERSONA, CHATS_DIR


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def get_chat_dir(chat_id: str) -> str:
    safe_chat_id = re.sub(r"[^a-zA-Z0-9_-]", "", chat_id)
    if not safe_chat_id:
        safe_chat_id = DEFAULT_CHAT_ID
    chat_path = os.path.join(CHATS_DIR, safe_chat_id)
    os.makedirs(chat_path, exist_ok=True)
    return chat_path


def get_chat_file_path(chat_id: str, filename: str) -> str:
    return os.path.join(get_chat_dir(chat_id), filename)


def append_message_to_disk(
    chat_id: str,
    role: str,
    content: str,
    ts: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    thoughts: Optional[str] = None,
    stats: Optional[str] = None,
    final_thoughts: Optional[str] = None,
):
    rec = {"ts": ts or now_iso(), "role": role, "content": content}
    if meta:
        rec["meta"] = meta
    if thoughts:
        rec["thoughts"] = thoughts
    if stats:
        rec["stats"] = stats
    if final_thoughts:
        rec["final_thoughts"] = final_thoughts
    with open(get_chat_file_path(chat_id, "messages.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec["ts"]


def read_last_messages(chat_id: str, k: int) -> List[Dict[str, Any]]:
    messages_file = get_chat_file_path(chat_id, "messages.jsonl")
    if not os.path.exists(messages_file):
        return []
    try:
        with open(messages_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        last = lines[-k:] if len(lines) > k else lines
        return [json.loads(ln) for ln in last if ln.strip()]
    except (IOError, json.JSONDecodeError):
        return []


def read_all_messages(chat_id: str) -> List[Dict[str, Any]]:
    messages_file = get_chat_file_path(chat_id, "messages.jsonl")
    if not os.path.exists(messages_file):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(messages_file, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    out.append(json.loads(ln))
        return out
    except (IOError, json.JSONDecodeError):
        return []


def save_all_messages(chat_id: str, messages: List[Dict[str, Any]]):
    with open(get_chat_file_path(chat_id, "messages.jsonl"), "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def load_persona(chat_id: str) -> Dict[str, Any]:
    persona_file = get_chat_file_path(chat_id, "persona.json")
    if os.path.exists(persona_file):
        try:
            with open(persona_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    with open(persona_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_PERSONA, f, ensure_ascii=False, indent=2)
    return DEFAULT_PERSONA


def save_persona(chat_id: str, persona: Dict[str, Any]):
    with open(get_chat_file_path(chat_id, "persona.json"), "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)


def load_summary(chat_id: str) -> str:
    summary_file = get_chat_file_path(chat_id, "summary.txt")
    if not os.path.exists(summary_file):
        return ""
    with open(summary_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def append_summary(chat_id: str, text: str, date: str, additional: str = ""):
    summary_entry = f"Summary for {date}:\n{text}{additional}\n\n"
    with open(get_chat_file_path(chat_id, "summary.txt"), "a", encoding="utf-8") as f:
        f.write(summary_entry)


def load_emotional_state(chat_id: str) -> Dict[str, str]:
    state_file = get_chat_file_path(chat_id, "emotional_state.json")
    if not os.path.exists(state_file):
        return {}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_emotional_state(chat_id: str, stats: Dict[str, str]):
    with open(get_chat_file_path(chat_id, "emotional_state.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

# Dodaj do storage.py jeśli brakuje
def delete_message_by_ts(chat_id: str, ts: str):
    messages = read_all_messages(chat_id)
    messages = [m for m in messages if m["ts"] != ts]
    save_all_messages(chat_id, messages)