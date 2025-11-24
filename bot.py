# gem_fixed.py
"""
Advanced roleplay server with proper streaming and modern UI
Usage:
pip install fastapi uvicorn requests python-dotenv numpy Pillow faiss-cpu openai
uvicorn gem:app --reload --port 7860
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, json, threading, asyncio, requests, re, shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np
import uvicorn

# optional faiss
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# --- OpenAI client helper ---
try:
    # ZMIANA: Dodano import AsyncOpenAI
    from openai import OpenAI, AsyncOpenAI, APIError
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    APIError = None
    HAS_OPENAI = False

load_dotenv()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHATS_DIR = os.path.join(APP_DIR, "chats")
DEFAULT_CHAT_ID = "default_chat"
STATIC_DIR = os.path.join(APP_DIR, "static")
PERSONAS_DIR = os.path.join(APP_DIR, "personas")
FRONTEND_HTML_PATH = os.path.join(STATIC_DIR, "index.html")
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PERSONAS_DIR, exist_ok=True)

# --- Dynamic File Path Management ---
def get_chat_dir(chat_id: str) -> str:
    safe_chat_id = re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)
    if not safe_chat_id:
        safe_chat_id = DEFAULT_CHAT_ID
    chat_path = os.path.join(CHATS_DIR, safe_chat_id)
    os.makedirs(chat_path, exist_ok=True)
    return chat_path

def get_chat_file_path(chat_id: str, filename: str) -> str:
    return os.path.join(get_chat_dir(chat_id), filename)

# config
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

# --- LMStudio / OpenAI client helpers ---

# Klient synchroniczny (do zadań blokujących jak podsumowania)
_openai_client = None
def get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if not HAS_OPENAI:
        return None
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key = os.getenv("MODEL_API_KEY", "lm-studio")
    try:
        _openai_client = OpenAI(base_url=base_url, api_key=api_key)
        print(f"Synchronous OpenAI client initialized for base_url: {base_url}")
    except Exception as e:
        print(f"get_openai_client init failed: {e}")
        _openai_client = None
    return _openai_client

# ZMIANA: Dodano klienta asynchronicznego (do streamingu w WebSocket)
_async_openai_client = None
def get_async_openai_client():
    global _async_openai_client
    if _async_openai_client is not None:
        return _async_openai_client
    if not HAS_OPENAI:
        return None
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key = os.getenv("MODEL_API_KEY", "lm-studio")
    try:
        _async_openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        print(f"Asynchronous OpenAI client initialized for base_url: {base_url}")
    except Exception as e:
        print(f"get_async_openai_client init failed: {e}")
        _async_openai_client = None
    return _async_openai_client

# Global stop flag
stop_generation = threading.Event()
FAISS_INDEX_CACHE = {} # key: chat_id, value: (index, count)
EMBEDDINGS_ENABLED = True # Enabled by default, compute_embedding will handle errors

_frontend_cache: Optional[str] = None

DEFAULT_PERSONA = {
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
    "prompt_examples": []
}

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------- Utilities ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def _normalize_labeled_block(text: str, header: str) -> str:
    if not text:
        return ""
    header_token = f"**[[{header}]]**"
    if header_token.lower() not in text.lower():
        return header_token + "\n" + text.strip()
    return text.strip()


def _extract_prefixed_section(text: str, label: str) -> (str, str):
    """Extract a section that begins with ``label:`` and return ``(body, remaining)``.

    This is a safety net for model outputs that forget the required markers but still
    prefix their thoughts or other sections with an explicit label. The function pulls
    the labeled paragraph out of ``text`` so that it can be rendered in the correct
    UI container instead of leaking into the main answer.
    """

    if not text:
        return "", text

    pattern = rf"^\s*{re.escape(label)}\s*:\s*(.+?)(?:\n{{2,}}|\r?\n\r?\n|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", text

    section_body = match.group(1).strip()
    remaining = (text[: match.start()] + text[match.end() :]).strip()
    return section_body, remaining


def parse_full_response(full_response: str) -> Dict[str, str]:
    response_data = {"thoughts": "", "content": "", "stats": "", "final_thoughts": ""}
    remaining_text = full_response

    think_open_match = re.search(r"<think\b[^>]*>", remaining_text, re.IGNORECASE)
    if think_open_match:
        think_start = think_open_match.start()
        think_content_start = think_open_match.end()
        think_close_match = re.search(r"</think\s*>", remaining_text[think_content_start:], re.IGNORECASE)

        if think_close_match:
            think_content_end = think_content_start + think_close_match.start()
            removal_end = think_content_end + len(think_close_match.group(0))
        else:
            remainder_after_think = remaining_text[think_content_start:]
            fallback_end = len(remaining_text)
            for pattern in (r"\*\*\[\[Stats\]\]\*\*", r"\*\*\[\[Final Thoughts\]\]\*\*"):
                marker_match = re.search(pattern, remainder_after_think, re.IGNORECASE)
                if marker_match:
                    fallback_end = think_content_start + marker_match.start()
                    break
            think_content_end = fallback_end
            removal_end = think_content_end

        response_data["thoughts"] = remaining_text[think_content_start:think_content_end].strip()
        remaining_text = remaining_text[:think_start] + remaining_text[removal_end:]

    # Fallback: capture a **[[Thoughts]]** block if <think> tags are missing
    if not response_data["thoughts"]:
        thoughts_match = re.search(
            r"(\*\*\[\[Thoughts\]\]\*\*[\s\S]*?)(?=\*\*\[\[(Stats|Final Thoughts)\]\]\*\*|$)",
            remaining_text,
            re.IGNORECASE,
        )
        if thoughts_match:
            raw_thoughts = thoughts_match.group(1)
            response_data["thoughts"] = re.sub(r"^\*\*\[\[Thoughts\]\]\*\*\s*", "", raw_thoughts, flags=re.IGNORECASE).strip()
            remaining_text = (remaining_text[: thoughts_match.start()] + remaining_text[thoughts_match.end() :]).strip()

    # Final fallback: pull out an inline "Thoughts: " prefix if present
    if not response_data["thoughts"]:
        extracted_thoughts, remaining_text = _extract_prefixed_section(remaining_text, "thoughts")
        response_data["thoughts"] = extracted_thoughts

    final_thoughts_match = re.search(r'(\*\*\[\[Final Thoughts\]\]\*\*[\s\S]*)', remaining_text, re.IGNORECASE)
    if final_thoughts_match:
        response_data["final_thoughts"] = final_thoughts_match.group(0).strip()
        remaining_text = remaining_text[:final_thoughts_match.start()]
    stats_match = re.search(r'(\*\*\[\[Stats\]\]\*\*[\s\S]*)', remaining_text, re.IGNORECASE)
    if stats_match:
        response_data["stats"] = stats_match.group(0).strip()
        remaining_text = remaining_text[:stats_match.start()]

    response_data["content"] = remaining_text.strip()
    if not response_data["thoughts"]:
        response_data["thoughts"] = ""
    response_data["stats"] = _normalize_labeled_block(response_data["stats"], "Stats")
    response_data["final_thoughts"] = _normalize_labeled_block(response_data["final_thoughts"], "Final Thoughts")
    return response_data

def append_message_to_disk(chat_id: str, role: str, content: str, ts: Optional[str] = None, meta: Optional[Dict[str,Any]] = None, thoughts: Optional[str] = None, stats: Optional[str] = None, final_thoughts: Optional[str] = None):
    rec = {"ts": ts or now_iso(), "role": role, "content": content}
    if meta: rec["meta"] = meta
    if thoughts: rec["thoughts"] = thoughts
    if stats: rec["stats"] = stats
    if final_thoughts: rec["final_thoughts"] = final_thoughts
    with open(get_chat_file_path(chat_id, "messages.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec["ts"]

def read_last_messages(chat_id: str, k: int) -> List[Dict[str, Any]]:
    messages_file = get_chat_file_path(chat_id, "messages.jsonl")
    if not os.path.exists(messages_file): return []
    try:
        with open(messages_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        last = lines[-k:] if len(lines) > k else lines
        return [json.loads(ln) for ln in last if ln.strip()]
    except (IOError, json.JSONDecodeError):
        return []

def read_all_messages(chat_id: str) -> List[Dict[str, Any]]:
    messages_file = get_chat_file_path(chat_id, "messages.jsonl")
    if not os.path.exists(messages_file): return []
    out = []
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

def delete_message_by_ts(chat_id: str, ts: str):
    messages = read_all_messages(chat_id)
    messages_to_keep = [msg for msg in messages if msg.get("ts") != ts]
    save_all_messages(chat_id, messages_to_keep)
    delete_embedding_by_ts(chat_id, ts)

def update_message_by_ts(chat_id: str, ts: str, raw_content: str):
    messages = read_all_messages(chat_id)
    updated = False
    parsed_data = parse_full_response(raw_content)
    for msg in messages:
        if msg.get("ts") == ts:
            msg["content"] = parsed_data["content"]
            msg["thoughts"] = parsed_data["thoughts"]
            msg["stats"] = parsed_data["stats"]
            msg["final_thoughts"] = parsed_data["final_thoughts"]
            updated = True
            break
    if updated:
        save_all_messages(chat_id, messages)
        delete_embedding_by_ts(chat_id, ts)
        vec = compute_embedding(parsed_data["content"], model_name=None)
        if vec is not None:
            append_embedding(chat_id, vec, {"ts": ts, "role": "assistant", "content": parsed_data["content"]})
    return updated

def load_persona(chat_id: str) -> Dict[str, Any]:
    persona_file = get_chat_file_path(chat_id, "persona.json")
    if os.path.exists(persona_file):
        try:
            with open(persona_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # If file is invalid or doesn't exist, create a new one with the default
    with open(persona_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_PERSONA, f, ensure_ascii=False, indent=2)
    return DEFAULT_PERSONA

def save_persona(chat_id: str, persona: Dict[str,Any]):
    with open(get_chat_file_path(chat_id, "persona.json"), "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)

def load_summary(chat_id: str) -> str:
    summary_file = get_chat_file_path(chat_id, "summary.txt")
    if not os.path.exists(summary_file): return ""
    with open(summary_file, "r", encoding="utf-8") as f:
        return f.read().strip()

def append_summary(chat_id: str, text: str, date: str, additional: str = ""):
    summary_entry = f"Summary for {date}:\n{text}{additional}\n\n"
    with open(get_chat_file_path(chat_id, "summary.txt"), "a", encoding="utf-8") as f:
        f.write(summary_entry)

def mark_new_day(chat_id: str):
    summary_text = summarize_older_messages_once(chat_id, force=True)
    new_day_file = get_chat_file_path(chat_id, "new_day.txt")
    session_count_file = get_chat_file_path(chat_id, "session_count.txt")
    prev_date_str = ""
    if os.path.exists(new_day_file):
        with open(new_day_file, "r", encoding="utf-8") as f:
            prev_date_str = f.read().strip()
    session_count = 1
    if os.path.exists(session_count_file):
        try:
            with open(session_count_file, "r") as f:
                session_count = int(f.read().strip()) + 1
        except (ValueError, FileNotFoundError):
            session_count = 1
    with open(session_count_file, "w") as f:
        f.write(str(session_count))
    now = datetime.now()
    d = now.strftime("%Y-%m-%d %H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    marker = f"-- NEW DAY: {d} --"
    days_passed = 0
    if prev_date_str:
        try:
            prev_date = datetime.strptime(prev_date_str, "%Y-%m-%d %H:%M:%S")
            days_passed = (now - prev_date).days
        except ValueError:
            pass
    additional_info = f"\nINFO: Starting new session on {current_date}. {days_passed} days have passed. Current session: {session_count}"
    append_message_to_disk(chat_id, "system", marker + additional_info)
    with open(new_day_file, "w", encoding="utf-8") as f:
        f.write(d)
    return marker, summary_text

def clear_chat_memory(chat_id: str):
    chat_dir = get_chat_dir(chat_id)
    if os.path.exists(chat_dir):
        shutil.rmtree(chat_dir)
    os.makedirs(chat_dir, exist_ok=True)
    # Re-initialize essential files
    save_persona(chat_id, DEFAULT_PERSONA)
    with open(get_chat_file_path(chat_id, "session_count.txt"), "w") as f: f.write("0")
    with open(get_chat_file_path(chat_id, "emotional_state.json"), "w") as f: json.dump({}, f)
    with open(get_chat_file_path(chat_id, "world_events.jsonl"), "w") as f: pass
    if chat_id in FAISS_INDEX_CACHE:
        del FAISS_INDEX_CACHE[chat_id]
    print(f"Memory cleared for chat: {chat_id}")

# ---------------- Dynamic Emotional State ----------------
def load_emotional_state(chat_id: str) -> Dict[str, str]:
    state_file = get_chat_file_path(chat_id, "emotional_state.json")
    if not os.path.exists(state_file): return {}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_emotional_state(chat_id: str, stats: Dict[str, str]):
    with open(get_chat_file_path(chat_id, "emotional_state.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def parse_stats_from_text(text: str) -> Dict[str, str]:
    stats = {}
    lines = text.strip().split('\n')
    try:
        start_index = next(i for i, line in enumerate(lines) if '[[Stats]]' in line)
        lines = lines[start_index + 1:]
    except StopIteration:
        return {} # No stats block found
    for line in lines:
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].replace('*', '').strip()
            value = parts[1].strip()
            if key and value:
                stats[key] = value
    return stats

# ---------------- Embeddings / Memory ----------------
def get_all_embeddings(chat_id: str):
    embeddings_npy = get_chat_file_path(chat_id, "embeddings.npy")
    embeddings_meta = get_chat_file_path(chat_id, "embeddings_meta.jsonl")
    if not os.path.exists(embeddings_npy) or not os.path.exists(embeddings_meta):
        return np.zeros((0,)), []
    try:
        embs = np.load(embeddings_npy)
    except Exception: # Handle empty or corrupted file
        embs = np.zeros((0,))
    meta = []
    with open(embeddings_meta, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try: meta.append(json.loads(ln))
                except: continue
    return embs, meta

def append_embedding(chat_id: str, vec: np.ndarray, meta: Dict[str,Any]):
    if chat_id in FAISS_INDEX_CACHE: del FAISS_INDEX_CACHE[chat_id]
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    embeddings_npy = get_chat_file_path(chat_id, "embeddings.npy")
    if not os.path.exists(embeddings_npy):
        np.save(embeddings_npy, vec)
    else:
        try:
            old = np.load(embeddings_npy)
            combined = np.vstack([old, vec]) if old.size > 0 else vec
            np.save(embeddings_npy, combined)
        except Exception: # If file is corrupt, overwrite it
            np.save(embeddings_npy, vec)
    with open(get_chat_file_path(chat_id, "embeddings_meta.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def delete_embedding_by_ts(chat_id: str, ts: str):
    if not EMBEDDINGS_ENABLED: return
    if chat_id in FAISS_INDEX_CACHE: del FAISS_INDEX_CACHE[chat_id]
    embs, meta = get_all_embeddings(chat_id)
    if embs.size > 0:
        keep_indices = [i for i, m in enumerate(meta) if m.get("ts") != ts]
        if len(keep_indices) < len(meta):
            embs = embs[keep_indices]
            meta = [meta[i] for i in keep_indices]
            np.save(get_chat_file_path(chat_id, "embeddings.npy"), embs)
            with open(get_chat_file_path(chat_id, "embeddings_meta.jsonl"), "w", encoding="utf-8") as f:
                for m in meta: f.write(json.dumps(m, ensure_ascii=False) + '\n')

def compute_embedding(text: str, model_name: Optional[str] = None) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    
    clean_text = re.sub(r'<br\s*/?>', '\n', text)
    clean_text = re.sub(r'<[^>]+>', '', clean_text).strip()
    if not clean_text:
        return None

    model_to_use = model_name or EMBEDDING_MODEL
    
    # Prefer OpenAI client if available
    client = get_openai_client()
    if client:
        try:
            resp = client.embeddings.create(model=model_to_use, input=[clean_text])
            vec = resp.data[0].embedding
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            print(f"OpenAI client embedding call failed for model {model_to_use}: {e}. Falling back to requests.")

    # Fallback to requests
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY: headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    payload = {"model": model_to_use, "input": [clean_text]}
    try:
        r = requests.post(EMBEDDING_API_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        vec = data['data'][0]['embedding']
        return np.array(vec, dtype=np.float32)
    except requests.RequestException as e:
        print(f"Embedding HTTP error for model {model_to_use}: {e}.")
        raise
    except (KeyError, IndexError) as e:
        print(f"Failed to parse embedding response for model {model_to_use}: {e}.")
        raise

def _start_embedding_thread(chat_id: str, text_to_embed: str, meta: dict, settings: dict):
    def _job():
        try:
            role = meta.get("role", "unknown")
            text_with_context = f"{role}: {text_to_embed}"
            embedding_model_name = settings.get("embedding_model")
            
            vec = compute_embedding(text_with_context, model_name=embedding_model_name)
            if vec is not None and vec.size > 0:
                append_embedding(chat_id, vec, meta)
        except Exception as e:
            print(f"Error in background embedding thread: {e}")
    t = threading.Thread(target=_job, daemon=True)
    t.start()

def build_faiss_index(chat_id: str, embs: np.ndarray):
    if not HAS_FAISS or embs.ndim != 2 or embs.shape[0] == 0: return None
    cached_index, cached_count = FAISS_INDEX_CACHE.get(chat_id, (None, 0))
    if cached_index is not None and cached_count == embs.shape[0]:
        return cached_index
    try:
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        normalized_embs = embs.copy()
        faiss.normalize_L2(normalized_embs)
        idx.add(normalized_embs)
        FAISS_INDEX_CACHE[chat_id] = (idx, embs.shape[0])
        return idx
    except Exception as e:
        print(f"Failed to build Faiss index: {e}")
        return None

def semantic_search(chat_id: str, query: str, top_k: int = TOP_K_MEMORIES, settings: Dict = {}) -> List[Dict[str,Any]]:
    if not settings.get("enable_memory", True):
        return []
    
    embedding_model_name = settings.get("embedding_model")
    try:
        query_with_context = f"user: {query}"
        qv = compute_embedding(query_with_context, model_name=embedding_model_name)
    except Exception as e:
        print(f"Semantic search failed because embedding could not be computed: {e}")
        return []

    if qv is None: return []
    embs, meta = get_all_embeddings(chat_id)
    if embs.size == 0: return []
    qv_norm = qv / (np.linalg.norm(qv) + 1e-12)
    if HAS_FAISS:
        idx = build_faiss_index(chat_id, embs)
        if idx is not None:
            try:
                _, I = idx.search(qv_norm.reshape(1, -1), top_k)
                return [meta[int(i)] for i in I[0] if 0 <= i < len(meta)]
            except Exception as e:
                print(f"Faiss search failed: {e}")
    # Fallback to numpy if Faiss fails or is unavailable
    try:
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = (embs_norm @ qv_norm).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [meta[int(i)] for i in top_idx]
    except Exception as e:
        print(f"Numpy search failed: {e}")
        return []

# ---------------- Chat model & parsing ----------------
def extract_content_from_json_chunk(part: Any) -> Optional[str]:
    try:
        if isinstance(part, dict) and "choices" in part and part["choices"]:
            delta = part["choices"][0].get("delta", {})
            if "content" in delta and delta["content"] is not None:
                return delta["content"]
    except (KeyError, IndexError, TypeError):
        pass
    return None

def _parse_chat_response_json(data: Any) -> str:
    if isinstance(data, dict):
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            if "content" in msg:
                return msg["content"]
    # Fallback for unexpected structures
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return ""

# Pozostawiamy tę funkcję dla operacji synchronicznych
def call_chat_model_raw(messages: List[Dict[str,str]], stream: bool=False, timeout:int=300, settings: Optional[Dict[str, Any]] = None):
    settings = settings or {}
    model_to_use = settings.get("model", TEXT_MODEL_NAME)
    
    payload = {"model": model_to_use, "messages": messages}
    if settings.get("temperature") is not None: payload["temperature"] = float(settings["temperature"])
    if settings.get("max_tokens") is not None: payload["max_tokens"] = int(settings["max_tokens"])
    
    client = get_openai_client()
    if client and not stream: # Użyj klienta synchronicznego tylko dla zapytań nie-strumieniowych
        try:
            completion = client.chat.completions.create(**payload, timeout=timeout)
            return json.loads(completion.model_dump_json())
        except Exception as e:
            print(f"Error calling model via sync OpenAI client: {e}")
            return {"_error": str(e)}

    # Fallback do 'requests' dla zadań synchronicznych, jeśli klient openai się nie powiedzie
    print(f"--- Falling back to 'requests' for sync call ---")
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY: headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    try:
        r = requests.post(MODEL_API_URL, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"Error calling model with requests: {e}")
        return {"_error": str(e)}

# ---------------- Summarization background ----------------
def summarize_older_messages_once(chat_id: str, force: bool = False) -> str:
    messages_file = get_chat_file_path(chat_id, "messages.jsonl")
    if not os.path.exists(messages_file): return "No messages to summarize."
    with open(messages_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not force and len(lines) <= RECENT_MSGS: return "Not enough messages to summarize."
    selected_lines = lines if force else lines[:-RECENT_MSGS]
    if not selected_lines: return "No older messages to summarize."
    text_to_summarize = []
    for ln in selected_lines:
        try:
            rec = json.loads(ln)
            if rec.get('role') in ('user', 'assistant'):
                text_to_summarize.append(f"[{rec.get('role')}]: {rec.get('content','')}")
        except json.JSONDecodeError:
            continue
    if not text_to_summarize: return "No user or assistant messages found to summarize."
    conversation_log = "\n".join(text_to_summarize)
    persona = load_persona(chat_id)
    system_prompt = (
        "You are a roleplay memory summarizer. Your task is to produce a structured, detailed summary of the provided conversation log. "
        "Focus on key events, character development, emotional shifts, and crucial decisions or promises made by either character. Be factual and objective. Extract specific, important details, not just a general overview.\n"
        "**Key Events & Decisions:**\n- [List the most important plot points, actions, and decisions.]\n\n"
        "**Key Facts & Relationships:**\n- [List specific facts established, e.g., 'User's name is John', 'Vex is afraid of spiders'.]\n- [Describe the current state of the relationship.]\n\n"
        "**Character States:**\n- User: [Describe the user's apparent role, personality, and key actions.]\n- Assistant: [Describe the assistant's emotional state, key decisions, or personality traits displayed.]"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please summarize the following conversation log:\n\n" + conversation_log}]
    resp = call_chat_model_raw(messages, stream=False)
    if isinstance(resp, dict) and "_error" in resp:
        return f"Summarization error: {resp.get('_error')}"
    summary_text = _parse_chat_response_json(resp)
    censor = persona.get("censor_list", []) or []
    for w in censor:
        if w: summary_text = summary_text.replace(w, "xxx")
    if force and summary_text:
      current_date = datetime.now().strftime("%Y-%m-%d")
      append_summary(chat_id, summary_text, current_date, "")
    return summary_text

# ---------------- Prompt builder ----------------
def build_prompt_context(chat_id: str, user_msg: str, settings: Dict[str, Any]) -> List[Dict[str,str]]:
    persona = load_persona(chat_id)
    summary = load_summary(chat_id)
    system_parts = []
    
    system_parts.append(f"You are: {persona.get('name')}. {persona.get('short_description')}")
    if persona.get("traits"): system_parts.append("Traits: " + ", ".join(persona.get("traits", [])))
    if persona.get("history"): system_parts.append("History: " + persona.get("history"))
    behavior_instructions = persona.get("behavior_instructions")
    if behavior_instructions:
        if isinstance(behavior_instructions, list):
            behavior_text = "\n".join([str(item) for item in behavior_instructions if item])
        else:
            behavior_text = str(behavior_instructions)
        system_parts.append("Behavior rules: " + behavior_text)
    
    system_parts.append(f"Current time: {datetime.utcnow().isoformat() + 'Z'}. Use this to track time and day/night in your responses.")
    if settings.get("persistent_stats", False):
        current_stats = load_emotional_state(chat_id)
        if current_stats:
            stats_str = "\n".join([f"**{k}**: {v}" for k, v in current_stats.items()])
            system_parts.append(f"Your current emotional state is:\n{stats_str}\nBased on the user's message, you must UPDATE this state in your response's [[Stats]] block. Do not invent a new state from scratch; evolve the existing one.")
        else:
            system_parts.append("This is your first interaction. Establish your initial emotional state in the [[Stats]] block of your response.")

    if summary: system_parts.append("Memory summary:\n" + (summary if len(summary) < 2000 else summary[-2000:]))
    
    if float(settings.get("thought_ratio", 0.5)) < 0.2: system_parts.append("Thought Ratio: LOW. Your <think> block MUST be very brief and concise.")
    elif float(settings.get("thought_ratio", 0.5)) > 0.8: system_parts.append("Thought Ratio: HIGH. Your <think> block MUST be extremely detailed.")
    if float(settings.get("talkativeness", 0.5)) < 0.2: system_parts.append("Talkativeness: VERY LOW. Your response (excluding thoughts/stats) MUST be extremely concise (1-2 sentences).")
    elif float(settings.get("talkativeness", 0.5)) > 0.8: system_parts.append("Talkativeness: VERY HIGH. Your response (excluding thoughts/stats) MUST be very long and descriptive.")

    system_parts.append(
        "Stay strictly in the first person. Describe any small actions inline using *action* markup within your dialogue, and avoid third-person narration or stage directions."
    )

    recent_messages = read_last_messages(chat_id, 5)
    greeting_delivered = any("New day greeting acknowledged" in m.get("content", "") for m in recent_messages if m.get("role") == "system")
    last_sys_msg = next((msg['content'] for msg in reversed(recent_messages) if msg['role'] == 'system'), "")
    if "-- NEW DAY:" in last_sys_msg and not greeting_delivered:
        days_passed_match = re.search(r'(\d+) days have passed', last_sys_msg)
        days_passed = int(days_passed_match.group(1)) if days_passed_match else 0
        greeting_instruction = (f"A new day has begun ({days_passed} days passed). Start with a morning greeting.") if days_passed > 0 else "This is a new session on the same day. Start with a greeting as if returning after a short break."
        system_parts.append(f"**CRITICAL CONTEXT: A NEW DAY/SESSION HAS BEGUN!**\n- **Instructions:** {greeting_instruction}\n- After the greeting, respond to the user's message as usual.\n")

    last_assistant_msg = next((m.get("content", "") for m in reversed(recent_messages) if m.get("role") == "assistant" and m.get("content")), "")
    if last_assistant_msg:
        preview = (last_assistant_msg[:300] + "...") if len(last_assistant_msg) > 300 else last_assistant_msg
        system_parts.append(
            "Avoid repeating or paraphrasing your last reply. Offer new details, emotions, or angles compared to this prior response: "
            + preview
        )

    output_instructions = persona.get("output_instructions", DEFAULT_PERSONA["output_instructions"])
    if isinstance(output_instructions, list):
        output_instructions = "\n".join([str(item) for item in output_instructions if item])
    reinforced_output_instructions = (
        "**MANDATORY OUTPUT FORMATTING RULES:**\n"
        "You MUST follow this structure for every single response. No exceptions. Each section requires its marker.\n"
        + output_instructions
    )
    system_parts.append(reinforced_output_instructions)
    system_parts.append("STRICT: The **[[Final Thoughts]]** section must NEVER be empty. Always include 1–2 concise reflective sentences. Keep <think> strictly for internal reasoning only; do not merge it with the answer.");

    system_text = "\n\n".join(system_parts)
    messages = [{"role": "system", "content": system_text}]

    relevant_memories = semantic_search(chat_id, user_msg, TOP_K_MEMORIES, settings)
    if relevant_memories:
        mem_texts = [f"[{m.get('role', '?')} @ {m.get('ts', '?')}] {m.get('content', '')}" for m in relevant_memories]
        messages.append({"role": "system", "content": "Relevant memories:\n" + "\n".join(mem_texts)})

    history = read_last_messages(chat_id, RECENT_MSGS)
    for r in history:
        if r.get("role") in ("user", "assistant") and r.get("content"):
            messages.append({"role": r["role"], "content": r.get("content")})

    # Avoid duplicating the latest user message if it was already persisted
    if not history or history[-1].get("role") != "user" or history[-1].get("content", "") != user_msg:
        messages.append({"role": "user", "content": user_msg})

    messages.append({"role": "assistant", "content": "<think>"})

    return messages

# ---------------- Modern UI ----------------

def load_frontend_html() -> str:
    """Load the prebuilt frontend file.

    Separating the UI into ``static/index.html`` keeps backend edits focused on
    Python while allowing HTML/CSS/JS changes without touching server code.
    """

    global _frontend_cache
    if _frontend_cache is None:
        try:
            with open(FRONTEND_HTML_PATH, "r", encoding="utf-8") as f:
                _frontend_cache = f.read()
        except FileNotFoundError:
            return "<h1>Frontend not found. Please ensure static/index.html exists.</h1>"
    return _frontend_cache

# ---------------- HTTP endpoints ----------------
def get_safe_chat_id(chat_id: str):
    safe_chat_id = re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)
    return safe_chat_id or DEFAULT_CHAT_ID

@app.get("/", response_class=HTMLResponse)
async def index():
    raw_html = load_frontend_html()
    final_html = raw_html.replace('__PERSISTENT_STATS_ENABLED__', "true" if PERSISTENT_STATS_ENABLED else "false")
    return HTMLResponse(content=final_html)

@app.get("/system_info")
async def system_info():
    return {"version": "0.45", "model_name": TEXT_MODEL_NAME}

@app.get("/models")
async def get_available_models():
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    api_key = os.getenv("MODEL_API_KEY", "lm-studio")
    models_url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"Attempting to fetch models from: {models_url}")
    try:
        response = requests.get(models_url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Received from /models endpoint: {data}")
        model_ids = [model.get("id") for model in data.get("data", []) if model.get("id")]
        if not model_ids and isinstance(data, list): # Handle alternate response format
             model_ids = [m.get('id') for m in data if m.get('id')]
        return {"models": sorted(list(set(model_ids)))}
    except requests.RequestException as e:
        print(f"ERROR: Failed to fetch models from {models_url}. Reason: {e}")
        return JSONResponse(status_code=500, content={"error": f"Connection to model server failed. Check if the server is running and if LMSTUDIO_BASE_URL in your .env file is correct. Details: {e}", "models": []})
    except Exception as e:
        print(f"An unexpected error occurred while fetching models: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "models": []})

# Chat Management
@app.get("/chats")
async def list_chats():
    if not os.path.exists(CHATS_DIR): return []
    return sorted([d for d in os.listdir(CHATS_DIR) if os.path.isdir(os.path.join(CHATS_DIR, d))])

@app.post("/chats/create")
async def create_chat(req: Request):
    body = await req.json()
    chat_name = body.get("name", "").strip()
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', chat_name)
    if not safe_name:
        return JSONResponse(status_code=400, content={"error": "Invalid chat name."})
    get_chat_dir(safe_name)
    save_persona(safe_name, DEFAULT_PERSONA)
    return {"status": "ok", "chat_id": safe_name}

@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(chat_id: str):
    safe_chat_id = get_safe_chat_id(chat_id)
    chat_dir = get_chat_dir(safe_chat_id)
    if os.path.exists(chat_dir):
        try:
            shutil.rmtree(chat_dir)
            return {"status": "ok", "message": f"Chat '{safe_chat_id}' deleted."}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to delete chat: {e}"})
    return JSONResponse(status_code=404, content={"error": "Chat not found."})

# Chat-specific endpoints
@app.get("/{chat_id}/messages")
async def get_messages(chat_id: str):
    return read_all_messages(get_safe_chat_id(chat_id))

@app.get("/{chat_id}/persona")
async def get_persona(chat_id: str):
    return load_persona(get_safe_chat_id(chat_id))

@app.post("/{chat_id}/persona")
async def post_persona(chat_id: str, req: Request):
    body = await req.json()
    save_persona(get_safe_chat_id(chat_id), body)
    return {"status": "ok"}

@app.post("/{chat_id}/clear_memory")
async def clear_memory_endpoint(chat_id: str):
    clear_chat_memory(get_safe_chat_id(chat_id))
    return {"status": "ok", "message": "Memory cleared."}

@app.post("/{chat_id}/new_day")
async def new_day_endpoint(chat_id: str):
    marker, summary = mark_new_day(get_safe_chat_id(chat_id))
    return {"marker": marker, "summary": summary}

@app.get("/{chat_id}/last_summary")
async def get_last_summary(chat_id: str):
    summary_content = load_summary(get_safe_chat_id(chat_id))
    if not summary_content:
        return {"summary": None}
    parts = re.split(r'Summary for .*:', summary_content)
    last_summary = parts[-1].strip() if len(parts) > 1 else summary_content.strip()
    return {"summary": last_summary}

@app.post("/{chat_id}/force_summarize")
async def force_summarize_endpoint(chat_id: str):
    safe_chat_id = get_safe_chat_id(chat_id)
    summary = summarize_older_messages_once(safe_chat_id, force=True)
    return {"summary": summary}

@app.post("/{chat_id}/delete_message")
async def delete_message_endpoint(chat_id: str, req: Request):
    body = await req.json()
    ts = body.get("ts")
    if ts:
        delete_message_by_ts(get_safe_chat_id(chat_id), ts)
        return {"ok": True}
    return JSONResponse({"error": "No ts provided"}, status_code=400)

@app.post("/{chat_id}/edit_message")
async def edit_message_endpoint(chat_id: str, req: Request):
    body = await req.json()
    ts, raw_content = body.get("ts"), body.get("raw_content")
    if ts is None or raw_content is None:
        return JSONResponse(status_code=400, content={"error": "ts and raw_content are required"})
    safe_chat_id = get_safe_chat_id(chat_id)
    if update_message_by_ts(safe_chat_id, ts, raw_content):
        messages = read_all_messages(safe_chat_id)
        updated_msg = next((msg for msg in messages if msg.get("ts") == ts), None)
        return {"ok": True, "updated_message": updated_msg or {}}
    return JSONResponse(status_code=404, content={"error": "Message not found"})

@app.post("/{chat_id}/inject_event")
async def inject_event_endpoint(chat_id: str, req: Request):
    safe_chat_id, body = get_safe_chat_id(chat_id), await req.json()
    event_text, p_type = body.get("event", "").strip(), body.get("type", "messages")
    try: value = int(body.get("value", 3))
    except (ValueError, TypeError): value = 3
    if not event_text: return JSONResponse(status_code=400, content={"error": "Event text cannot be empty."})
    event_entry = {"text": event_text, "persistence_type": p_type, "value": value, "start_ts": now_iso(), "start_msg_count": len(read_all_messages(safe_chat_id))}
    with open(get_chat_file_path(safe_chat_id, "world_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(event_entry) + "\n")
    append_message_to_disk(safe_chat_id, "system", f"[WORLD EVENT INJECTED] {event_text}")
    return {"status": "ok", "event_injected": event_text}

# Global endpoints
@app.post("/test_text_model")
async def test_text_model(req: Request):
    body = await req.json()
    model_name = body.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"success": False, "error": "No model name provided."})
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
        settings = {"model": model_name, "max_tokens": 20}
        resp = call_chat_model_raw(messages, stream=False, timeout=20, settings=settings)

        if isinstance(resp, dict) and "_error" in resp:
            raise Exception(resp["_error"])
        
        content = _parse_chat_response_json(resp)
        if content:
            return {"success": True, "response": content}
        else:
            raise Exception("Received an empty or invalid response from the model.")
    except Exception as e:
        error_message = f"An API error occurred: {str(e)}. Check if the model is a chat model and is fully loaded."
        return {"success": False, "error": error_message}

@app.get("/personas")
async def list_personas():
    if not os.path.exists(PERSONAS_DIR): return []
    return sorted([f.replace('.json', '') for f in os.listdir(PERSONAS_DIR) if f.endswith('.json')])

@app.get("/personas/{name}")
async def get_saved_persona(name: str):
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    file_path = os.path.join(PERSONAS_DIR, f"{safe_name}.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Persona not found"})
    except json.JSONDecodeError as e:
        print(f"ERROR: Corrupted persona file at {file_path}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Persona file '{safe_name}.json' is corrupted."})


@app.post("/personas/{name}")
async def save_named_persona(name: str, req: Request):
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    if not safe_name: return JSONResponse(status_code=400, content={"error": "Invalid persona name"})
    body = await req.json()
    with open(os.path.join(PERSONAS_DIR, f"{safe_name}.json"), "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
    return {"status": "saved", "name": safe_name}

@app.post("/generate_persona")
async def generate_persona(req: Request):
    description = (await req.json()).get("description")
    if not description:
        return JSONResponse(status_code=400, content={"error": "Description is required"})
    generation_prompt = f"""Based on the user's simple description, create a detailed persona JSON object for a roleplaying AI. User wants: "{description}". Expand on this. The JSON should include 'name', 'avatar' (a filename like 'ninja_avatar.png'), 'short_description', 'traits' (a list of strings), 'history', 'behavior_instructions' and 'output_instructions'. Make the persona rich and interesting. The output_instructions MUST be the detailed template including <think>, **[[Stats]]**, and **[[Final Thoughts]]**. Your output MUST be ONLY the raw JSON object, with no other text or markdown fences before or after it."""
    messages = [{"role": "system", "content": "You are a creative assistant that generates JSON objects for AI personas."}, {"role": "user", "content": generation_prompt}]
    response = call_chat_model_raw(messages, stream=False)
    if isinstance(response, dict) and "_error" in response:
        return JSONResponse(status_code=500, content={"error": response["_error"]})
    content = _parse_chat_response_json(response).strip()
    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match: raise ValueError("No JSON object found in the LLM response.")
        persona_json = json.loads(json_match.group(0))
        if "output_instructions" not in persona_json:
            persona_json["output_instructions"] = DEFAULT_PERSONA["output_instructions"]
        return {"persona": persona_json}
    except Exception as e:
        print(f"Failed to parse persona JSON from LLM. Response was:\n{content}")
        return JSONResponse(status_code=500, content={"error": f"Failed to parse LLM response as JSON: {e}"})

@app.post("/test_embeddings")
async def test_embeddings(req: Request):
    body = await req.json()
    model_name = body.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"success": False, "error": "No model name provided."})
    try:
        test_vec = compute_embedding("test", model_name=model_name)
        success = test_vec is not None and test_vec.shape[0] > 0
        if not success:
            return {"success": False, "error": "Failed to get a valid embedding vector. Check if the model is an embedding model and is fully loaded in LM Studio."}
        return {"success": True, "error": ""}
    except Exception as e:
        return {"success": False, "error": f"An API error occurred: {str(e)}"}

# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("INFO:     connection open")
    global stop_generation
    chat_id = DEFAULT_CHAT_ID

    # ZMIANA: Cała funkcja została przeniesiona i zmodyfikowana, aby używać klienta asynchronicznego
    async def process_and_stream_response(messages: List[Dict], settings: Dict, chat_id: str):
        stop_generation.clear()

        # Notify the client that a fresh generation cycle is starting so it can
        # reset any buffered partial text from previous replies. Previously this
        # signal was only sent during regenerations, which could leave the
        # frontend reusing stale `fullResponseText` and showing repeated
        # content. Broadcasting the start event for every turn keeps the UI in
        # sync with the server-side stream.
        await ws.send_text(json.dumps({"type": "start"}))
        
        memory_message = next((m['content'] for m in messages if m['role'] == 'system' and 'Relevant memories:' in m['content']), None)
        if memory_message:
            await ws.send_text(json.dumps({"type": "memory_info", "content": memory_message}))

        full_response = ""
        if messages[-1]["role"] == "assistant" and messages[-1]["content"]:
             full_response += messages[-1]["content"]

        client = get_async_openai_client()
        if not client:
            await ws.send_text(json.dumps({"type": "error", "message": "Asynchronous OpenAI client not initialized."}))
            return

        try:
            model_to_use = settings.get("model", TEXT_MODEL_NAME)
            payload = {"model": model_to_use, "messages": messages, "stream": True}
            if settings.get("temperature") is not None: payload["temperature"] = float(settings["temperature"])
            if settings.get("max_tokens") is not None: payload["max_tokens"] = int(settings["max_tokens"])

            # Użycie klienta asynchronicznego i pętli 'async for'
            stream = await client.chat.completions.create(**payload)
            async for chunk in stream:
                if stop_generation.is_set():
                    print("Streaming stopped by client.")
                    break
                
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    full_response += chunk_text
                    await ws.send_text(json.dumps({"type": "partial", "chunk": chunk_text}))

        except Exception as e:
            error_message = f"Stream processing error: {str(e)}"
            print(f"Error during OpenAI stream processing: {e}")
            await ws.send_text(json.dumps({"type": "error", "message": error_message}))

        # Finalizacja po zakończeniu strumienia
        if stop_generation.is_set():
            await ws.send_text(json.dumps({"type": "stopped"}))
        else:
            await ws.send_text(json.dumps({"type": "done"}))
        
        if full_response:
            parsed_data = parse_full_response(full_response)
            if parsed_data["content"] or parsed_data["thoughts"]:
                if settings.get("persistent_stats", False) and parsed_data["stats"]:
                    save_emotional_state(chat_id, parse_stats_from_text(parsed_data["stats"]))
                
                assistant_ts = append_message_to_disk(
                    chat_id, "assistant",
                    parsed_data["content"],
                    thoughts=parsed_data["thoughts"],
                    stats=parsed_data["stats"],
                    final_thoughts=parsed_data["final_thoughts"]
                )
                await ws.send_text(json.dumps({"type": "assistant_ts", "ts": assistant_ts}))
                
                if settings.get("enable_memory", True) and parsed_data["content"]:
                    _start_embedding_thread(
                        chat_id, parsed_data["content"],
                        {"ts": assistant_ts, "role": "assistant", "content": parsed_data["content"]},
                        settings
                    )
        
        if any("CRITICAL CONTEXT: A NEW DAY/SESSION HAS BEGUN!" in m.get('content', '') for m in messages):
            append_message_to_disk(chat_id, "system", "New day greeting acknowledged.")

    try:
        while True:
            data = await ws.receive_text()
            obj = json.loads(data)
            if "chat_id" in obj:
                safe_id = get_safe_chat_id(obj["chat_id"])
                if os.path.exists(get_chat_dir(safe_id)): chat_id = safe_id
            
            msg_type = obj.get("type")
            if msg_type == "init":
                await ws.send_text(json.dumps({"type": "chat_switched", "chat_id": chat_id}))
                continue
            
            if msg_type == "stop":
                stop_generation.set()
                continue

            settings = obj.get("settings", {})
            if msg_type == "regenerate":
                ts_to_regen = obj.get("ts")
                if not ts_to_regen: continue

                await ws.send_text(json.dumps({"type": "start", "old_ts": ts_to_regen}))

                all_chat_messages = read_all_messages(chat_id)
                user_msg_for_regen = None
                regen_start_index = -1
                for i, msg in enumerate(all_chat_messages):
                    if msg.get("ts") == ts_to_regen:
                        regen_start_index = i
                        break
                if regen_start_index > 0 and all_chat_messages[regen_start_index - 1].get("role") == "user":
                    user_msg_for_regen = all_chat_messages[regen_start_index - 1]
                    delete_message_by_ts(chat_id, ts_to_regen)
                else: 
                    continue
                
                messages_for_regen = build_prompt_context(chat_id, user_msg_for_regen.get("content", ""), settings)
                await process_and_stream_response(messages_for_regen, settings, chat_id)
            
            elif msg_type == "message":
                user_msg_content = obj.get("message", "").strip()
                if not user_msg_content: continue
                
                user_ts = append_message_to_disk(chat_id, "user", user_msg_content)
                await ws.send_text(json.dumps({"type": "user_ts", "ts": user_ts}))
                
                if settings.get("enable_memory", True):
                    _start_embedding_thread(
                        chat_id, user_msg_content,
                        {"ts": user_ts, "role": "user", "content": user_msg_content},
                        settings
                    )
                
                messages = build_prompt_context(chat_id, user_msg_content, settings)
                await process_and_stream_response(messages, settings, chat_id)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for chat {chat_id}.")
    except Exception as e:
        import traceback
        print(f"An error occurred in websocket for chat {chat_id}: {e}")
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"Server error: {str(e)}"}))
        except: pass
    finally:
        stop_generation.set()

if __name__ == "__main__":
    default_avatar_path = os.path.join(STATIC_DIR, "default_avatar.png")
    if not os.path.exists(default_avatar_path):
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (256, 256), color = (40, 40, 40))
            d = ImageDraw.Draw(img)
            font = None
            try:
                # Use a common system font as a fallback
                font = ImageFont.truetype("arial.ttf", 60)
            except IOError:
                print("Arial font not found. Using default font.")
                font = ImageFont.load_default()
            
            # Center the text
            text_width, text_height = d.textsize("AI", font=font)
            position = ((256-text_width)/2,(256-text_height)/2)
            d.text(position, "AI", fill=(200,200,200), font=font)
            img.save(default_avatar_path, 'PNG')
            print(f"Created default avatar at {default_avatar_path}")
        except ImportError:
            print("Pillow not installed, cannot create default avatar. Place a 'default_avatar.png' in the 'static' directory.")
        except Exception as e:
            print(f"An unexpected error occurred during avatar creation: {e}")

    uvicorn.run(app, host="0.0.0.0", port=7860)
