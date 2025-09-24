"""Utilities for parsing structured model responses."""
from __future__ import annotations

import re
from typing import Dict


_SECTION_KEYS = ("thoughts", "content", "stats", "final_thoughts")


def parse_full_response(full_response: str) -> Dict[str, str]:
    """Parse a structured assistant response into its individual sections."""
    response_data = {key: "" for key in _SECTION_KEYS}
    if not isinstance(full_response, str):
        return response_data

    remaining_text = full_response

    think_open_match = re.search(r"<think\\b[^>]*>", remaining_text, re.IGNORECASE)
    if think_open_match:
        think_start = think_open_match.start()
        think_content_start = think_open_match.end()
        think_close_match = re.search(r"</think\\s*>", remaining_text[think_content_start:], re.IGNORECASE)
        if think_close_match:
            think_content_end = think_content_start + think_close_match.start()
            response_data["thoughts"] = remaining_text[think_content_start:think_content_end].strip()
            remaining_text = (
                remaining_text[:think_start]
                + remaining_text[think_content_end + len(think_close_match.group(0)) :]
            )
        else:
            response_data["thoughts"] = remaining_text[think_content_start:].strip()
            remaining_text = remaining_text[:think_start]

    final_thoughts_match = re.search(r"(\*\*\[\[Final Thoughts\]\]\*\*[\s\S]*)", remaining_text, re.IGNORECASE)
    if final_thoughts_match:
        response_data["final_thoughts"] = final_thoughts_match.group(0).strip()
        remaining_text = remaining_text[: final_thoughts_match.start()]

    stats_match = re.search(r"(\*\*\[\[Stats\]\]\*\*[\s\S]*)", remaining_text, re.IGNORECASE)
    if stats_match:
        response_data["stats"] = stats_match.group(0).strip()
        remaining_text = remaining_text[: stats_match.start()]

    response_data["content"] = remaining_text.strip()
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