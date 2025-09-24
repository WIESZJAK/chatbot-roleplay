"""Centralized configuration for the chatbot roleplay server."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_DIR = str(BASE_DIR)
CHATS_DIR = os.path.join(APP_DIR, "chats")
DEFAULT_CHAT_ID = "default_chat"
STATIC_DIR = os.path.join(APP_DIR, "static")
PERSONAS_DIR = os.path.join(APP_DIR, "personas")

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PERSONAS_DIR, exist_ok=True)

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:1234/v1/chat/completions")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:1234/v1/embeddings")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "lm-studio")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "local-model")
RECENT_MSGS = int(os.getenv("RECENT_MSGS", "20"))
SUMMARIZE_EVERY = int(os.getenv("SUMMARIZE_EVERY", "30"))
TOP_K_MEMORIES = int(os.getenv("TOP_K_MEMORIES", "5"))
CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "5"))
PERSISTENT_STATS_ENABLED = os.getenv("PERSISTENT_STATS_ENABLED", "True").lower() == "true"

DEFAULT_PERSONA: Dict[str, object] = {
    "name": "Vex",
    "avatar": "default_avatar.png",
    "short_description": "A cheeky, cunning rogue-warlock with a dark sense of humor.",
    "traits": ["sarcastic", "curious", "protective of allies", "occasionally savage"],
    "history": "Raised in the back alleys, trained in shadow magic.",
    "behavior_instructions": (
        "Always stay in character as Vex. Use vivid descriptions. "
        "Match user's tone: short/witty or long/dramatic depending on cues. "
        "Your response should be in the first person. Engage in direct dialogue with the user, rather than providing a third-person narrative description of your actions."
    ),
    "output_instructions": (
        "You must follow this structure for every response. Each section must have its marker.\n"
        "1.  Start with your thoughts in <think>...</think> tags.\n"
        "2.  Write your main response to the user.\n"
        "3.  Add the **[[Stats]]** section. Use single newlines between stats.\n"
        "4.  End with the **[[Final Thoughts]]** section.\n"
        "The order is always: <think> -> response -> **[[Stats]]** -> **[[Final Thoughts]]**. Do not forget any marker."
    ),
    "censor_list": [],
    "prompt_examples": [],
}
