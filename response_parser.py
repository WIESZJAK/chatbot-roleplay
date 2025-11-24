"""Utilities for parsing structured model responses."""
from __future__ import annotations

import re
from typing import Dict, Tuple


_SECTION_KEYS = ("thoughts", "content", "stats", "final_thoughts")


def _extract_prefixed_section(text: str, label: str) -> Tuple[str, str]:
    """Return a tuple of (section_body, remaining_text).

    Acts as a safety net for responses that prefix sections with ``label:`` but omit
    the expected markers. Pulling these stray sections out ensures the UI renders
    them in the right place instead of mixing them with the main answer.
    """

    if not text:
        return "", text

    pattern = rf"^\s*{re.escape(label)}\s*:\s*(.+?)(?:\n{{2,}}|\r?\n\r?\n|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", text

    section = match.group(1).strip()
    remaining = (text[: match.start()] + text[match.end() :]).strip()
    return section, remaining


def parse_full_response(full_response: str) -> Dict[str, str]:
    response_data = {"thoughts": "", "content": "", "stats": "", "final_thoughts": ""}
    text = full_response or ""
    
    # 1. Wyciągnij myśli <think> (Standardowo)
    think_match = re.search(r"<think\b[^>]*>(.*?)</think\s*>", text, re.IGNORECASE | re.DOTALL)
    if think_match:
        response_data["thoughts"] = think_match.group(1).strip()
        text = text.replace(think_match.group(0), "") # Usuwamy myśli z tekstu
    
    # 2. Wyciągnij Final Thoughts i WYCZYŚĆ ŚMIECI
    ft_match = re.search(r"(\*\*\[\[Final Thoughts\]\]\*\*|\[\[Final Thoughts\]\])([\s\S]*)$", text, re.IGNORECASE)
    if ft_match:
        raw_ft = ft_match.group(2).strip()
        
        # -- FIX: Usuwanie "Relevant memories" i powtórzonych wiadomości usera --
        # Jeśli model wypluwa historię czatu na końcu, ucinamy to.
        memory_leak_match = re.search(r"(Relevant memories|\[user\]|User:|Relevant context):", raw_ft, re.IGNORECASE)
        if memory_leak_match:
            raw_ft = raw_ft[:memory_leak_match.start()].strip()
            
        response_data["final_thoughts"] = raw_ft
        text = text[:ft_match.start()].strip()

    # 3. Wyciągnij Stats
    st_match = re.search(r"(\*\*\[\[Stats\]\]\*\*|\[\[Stats\]\])([\s\S]*)$", text, re.IGNORECASE)
    if st_match:
        response_data["stats"] = st_match.group(2).strip()
        text = text[:st_match.start()].strip()

    # 4. Wyciągnij RESPONSE (Nowy znacznik!)
    # Szukamy znacznika **[[Response]]**
    resp_match = re.search(r"(\*\*\[\[Response\]\]\*\*|\[\[Response\]\])([\s\S]*)", text, re.IGNORECASE)
    if resp_match:
        # Jeśli znaleziono znacznik, to co po nim jest treścią
        response_data["content"] = resp_match.group(2).strip()
        
        # A to co było PRZED znacznikiem, jeśli nie zostało wyłapane jako <think>,
        # może być niedomkniętą myślą.
        pre_response_text = text[:resp_match.start()].strip()
        if pre_response_text and not response_data["thoughts"]:
             # Czyścimy ewentualne otwarte tagi <think>
             response_data["thoughts"] = pre_response_text.replace("<think>", "").strip()
             
    else:
        # Fallback: Jeśli bot zapomniał znacznika Response, bierzemy to co zostało
        # Ale jeśli wcześniej wycięliśmy <think>, to jest bezpieczne.
        # Jeśli nie wycięliśmy <think> (bo brakowało zamknięcia), to tutaj mamy problem.
        # W takim wypadku zakładamy, że jeśli tekst zaczyna się od <think>, to całość to myśli (błąd modelu),
        # ale skoro nie ma znacznika Response, ciężko zgadnąć.
        # Przyjmijmy prostą strategię:
        if "<think>" in text and not response_data["thoughts"]:
             # Całość to myśli (urwana odpowiedź)
             response_data["thoughts"] = text.replace("<think>", "").strip()
        else:
             response_data["content"] = text.strip()

    return response_data


def parse_stats_from_text(text: str) -> Dict[str, str]:
    """Extract key/value stat pairs from a ``[[Stats]]`` block."""
    stats: Dict[str, str] = {}
    lines = text.strip().split("\n")
    try:
        start_index = next(i for i, line in enumerate(lines) if "[[Stats]]" in line)
        lines = lines[start_index + 1 :]
    except StopIteration:
        return stats

    for line in lines:
        line = line.strip()
        if ":" not in line:
            continue
        key_part, value_part = line.split(":", 1)
        key = key_part.replace("*", "").strip()
        value = value_part.strip()
        if key and value:
            stats[key] = value
    return stats