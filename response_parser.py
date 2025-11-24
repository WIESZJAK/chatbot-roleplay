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
    # Upewnij się, że funkcja _normalize_labeled_block jest zdefiniowana w tym pliku
    # lub jest poprawnie importowana, jeśli jej używasz. 
    # W nowej wersji z niej rezygnujemy, aby dane w JSON były czyste.

    response_data = {"thoughts": "", "content": "", "stats": "", "final_thoughts": ""}
    
    # Usuwamy ewentualne śmieciowe tagi XML, jeśli model je generuje
    text = full_response or ""
    text = re.sub(r"</?END_ANSWER>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?END_FINAL>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?BEGIN_ANSWER>", "", text, flags=re.IGNORECASE)
    
    # 1. Wyciągnij myśli <think>
    think_match = re.search(r"<think\b[^>]*>(.*?)</think\s*>", text, re.IGNORECASE | re.DOTALL)
    if think_match:
        response_data["thoughts"] = think_match.group(1).strip()
        text = text.replace(think_match.group(0), "")
    
    # 2. Wyciągnij Final Thoughts i WYCZYŚĆ nagłówek
    # Szukamy nagłówka (z gwiazdkami lub bez) i łapiemy całą resztę
    ft_match = re.search(r"(\*\*\[\[Final Thoughts\]\]\*\*|\[\[Final Thoughts\]\])([\s\S]*)$", text, re.IGNORECASE)
    if ft_match:
        # Treść sekcji (z nagłówkiem)
        raw_ft = ft_match.group(0).strip() 
        # Usuwamy tylko nagłówek, zostawiając treść
        cleaned_ft = re.sub(r"^(\*\*\[\[Final Thoughts\]\]\*\*|\[\[Final Thoughts\]\])", "", raw_ft, flags=re.IGNORECASE).strip()
        response_data["final_thoughts"] = cleaned_ft
        text = text[:ft_match.start()].strip() # Skracamy główny tekst

    # 3. Wyciągnij Stats i WYCZYŚĆ nagłówek
    st_match = re.search(r"(\*\*\[\[Stats\]\]\*\*|\[\[Stats\]\])([\s\S]*)$", text, re.IGNORECASE)
    if st_match:
        raw_st = st_match.group(0).strip()
        # Usuwamy tylko nagłówek, zostawiając treść
        cleaned_st = re.sub(r"^(\*\*\[\[Stats\]\]\*\*|\[\[Stats\]\])", "", raw_st, flags=re.IGNORECASE).strip()
        response_data["stats"] = cleaned_st
        text = text[:st_match.start()].strip() # Skracamy główny tekst

    # 4. Reszta to content
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