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

def parse_full_response(full_response: str) -> Dict[str, str]:
    response_data = {"thoughts": "", "content": "", "stats": "", "final_thoughts": ""}
    remaining_text = full_response
    thoughts_match = re.search(r'<think>([\s\S]*?)</think>', remaining_text, re.DOTALL)
    if thoughts_match:
        response_data["thoughts"] = thoughts_match.group(1).strip()
        remaining_text = remaining_text.replace(thoughts_match.group(0), '', 1)
    final_thoughts_match = re.search(r'(\*\*\[\[Final Thoughts\]\]\*\*[\s\S]*)', remaining_text, re.IGNORECASE)
    if final_thoughts_match:
        response_data["final_thoughts"] = final_thoughts_match.group(0).strip()
        remaining_text = remaining_text[:final_thoughts_match.start()]
    stats_match = re.search(r'(\*\*\[\[Stats\]\]\*\*[\s\S]*)', remaining_text, re.IGNORECASE)
    if stats_match:
        response_data["stats"] = stats_match.group(0).strip()
        remaining_text = remaining_text[:stats_match.start()]
    response_data["content"] = remaining_text.strip()
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

    recent_messages = read_last_messages(chat_id, 5)
    greeting_delivered = any("New day greeting acknowledged" in m.get("content", "") for m in recent_messages if m.get("role") == "system")
    last_sys_msg = next((msg['content'] for msg in reversed(recent_messages) if msg['role'] == 'system'), "")
    if "-- NEW DAY:" in last_sys_msg and not greeting_delivered:
        days_passed_match = re.search(r'(\d+) days have passed', last_sys_msg)
        days_passed = int(days_passed_match.group(1)) if days_passed_match else 0
        greeting_instruction = (f"A new day has begun ({days_passed} days passed). Start with a morning greeting.") if days_passed > 0 else "This is a new session on the same day. Start with a greeting as if returning after a short break."
        system_parts.append(f"**CRITICAL CONTEXT: A NEW DAY/SESSION HAS BEGUN!**\n- **Instructions:** {greeting_instruction}\n- After the greeting, respond to the user's message as usual.\n")

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

    messages.append({"role": "user", "content": user_msg})

    messages.append({"role": "assistant", "content": "<think>"})

    return messages

# ---------------- Modern UI ----------------
HTML_UI = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Roleplay Assistant</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg-primary: #0a0a0a; --bg-secondary: #141414; --bg-tertiary: #1a1a1a;
  --text-primary: #ffffff; --text-secondary: #a0a0a0; --accent: #6366f1;
  --accent-hover: #4f46e5; --user-bubble: #262626; --border: rgba(255, 255, 255, 0.1);
  --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); --danger: #ef4444;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg-primary); color: var(--text-primary);
  height: 100vh; display: flex; overflow: hidden;
}
.container { display: flex; width: 100%; height: 100%; transition: padding 0.3s ease-in-out; }
.main-panel {
  flex: 1; display: flex; flex-direction: column; background: var(--bg-secondary);
  border-radius: 12px; margin: 12px; overflow: hidden; min-width: 0;
}
.chat-header {
  background: var(--bg-tertiary); padding: 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.chat-header h1 { font-size: 1.25rem; font-weight: 600; }
.status { display: flex; align-items: center; gap: 8px; font-size: 0.875rem; color: var(--text-secondary); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); }
.status-dot.connected { background: #10b981; }
.status-dot.generating { background: #f59e0b; animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.chat-messages {
  flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px;
}
.chat-messages::-webkit-scrollbar { width: 8px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }
.message { display: flex; flex-direction: column; gap: 4px; max-width: 80%; animation: fadeIn 0.3s ease-in; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.message.user { align-self: flex-end; }
.message.assistant { align-self: flex-start; }
.message.system { align-self: center; max-width: 90%; opacity: 0.7; }
.message-body { display: flex; flex-direction: column; width: 100%; }
.message-content {
    font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
    position: relative; overflow-wrap: anywhere; padding: 4px 0;
}
.message.user .message-body { background: var(--user-bubble); padding: 10px 15px; border-radius: 18px; text-align: left; }
.message.system .message-body {
  background: rgba(255,255,255,0.05); border: 1px solid var(--border); text-align: left;
  font-size: 0.875rem; color: var(--text-secondary); padding: 12px 18px; border-radius: 18px;
  white-space: pre-wrap;
}
.thought-container {
  font-size: 0.85em; color: var(--text-secondary); white-space: pre-wrap; word-break: break-word; font-family: monospace;
  border-bottom: 1px solid var(--border); padding: 8px 0 12px 0; margin-bottom: 8px;
  cursor: pointer; overflow: hidden; max-height: 70px; transition: max-height 0.3s ease-in-out; position: relative;
}
.thought-container .thought-content { overflow: hidden; height: 100%; }
.thought-container::before {
  content: 'Thoughts ▼'; font-weight: bold; display: block; margin-bottom: 8px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  color: #fff; position: sticky; top: 0; background: var(--bg-secondary);
}
.thought-container.expanded { max-height: 1000px; overflow-y: auto; }
.thought-container.expanded::before { content: 'Thoughts ▲'; }
.stats-container, .final-thoughts-container {
  padding: 8px 0 0 0; margin-top: 8px; border-top: 1px solid var(--border); font-size: 0.85em;
}
.message-footer {
  display: flex; align-items: center; gap: 12px;
  font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px;
  opacity: 0.6; transition: opacity 0.2s; padding: 0 5px;
}
.message:hover .message-footer { opacity: 1; }
.message.user .message-footer { align-self: flex-end; }
.message.assistant .message-footer { align-self: flex-start; }
.message-actions { display: flex; gap: 8px; }
.message-actions button {
  background: none; border: none; color: var(--text-secondary); cursor: pointer;
  font-size: 1.1rem; padding: 2px 4px; border-radius: 4px; transition: all 0.2s;
}
.message-actions button:hover { color: var(--text-primary); background: var(--bg-tertiary); }
.chat-input-container { padding: 20px; background: var(--bg-tertiary); border-top: 1px solid var(--border); }
.chat-input-wrapper { display: flex; gap: 12px; align-items: flex-end; }
.chat-input {
  flex: 1; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 24px;
  padding: 12px 20px; color: var(--text-primary); font-size: 0.95rem; resize: none;
  outline: none; transition: border-color 0.2s; max-height: 120px; overflow-y: auto;
}
.chat-input:focus { border-color: var(--accent); }
.send-button {
  background: var(--accent); color: white; border: none; border-radius: 50%; width: 44px;
  height: 44px; display: flex; align-items: center; justify-content: center; cursor: pointer;
  transition: all 0.2s; flex-shrink: 0;
}
.send-button:hover:not(:disabled) { background: var(--accent-hover); transform: scale(1.05); }
.send-button:disabled { opacity: 0.5; cursor: not-allowed; }
.stop-button { background: var(--danger); }
.stop-button:hover { background: #dc2626; }
.side-panel {
  width: 320px; background: var(--bg-tertiary); padding: 20px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 24px; transition: all 0.3s ease-in-out;
  flex-shrink: 0;
}
.side-panel.left-panel { border-right: 1px solid var(--border); }
.side-panel.right-panel { border-left: 1px solid var(--border); }
.side-panel.collapsed { width: 0; padding: 20px 0; overflow: hidden; }
.panel-toggle-handle {
  width: 20px; background: var(--bg-tertiary); cursor: pointer; display: flex;
  align-items: center; justify-content: center; color: var(--text-secondary);
  transition: all 0.3s ease-in-out;
}
.panel-toggle-handle:hover { background: var(--bg-secondary); color: var(--text-primary); }
.panel-section { display: flex; flex-direction: column; gap: 12px; }
.panel-title {
  font-size: 0.875rem; font-weight: 600; color: var(--text-secondary);
  text-transform: uppercase; letter-spacing: 0.05em; cursor: pointer;
}
.panel-title::after { content: ' ▼'; font-size: 0.8em; }
.panel-title.collapsed::after { content: ' ►'; }
.action-buttons { display: flex; flex-wrap: wrap; gap: 8px; }
.action-buttons .btn { flex: 1 1 120px; text-align: center;}
.btn {
  background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary);
  padding: 8px 12px; border-radius: 8px; font-size: 0.875rem; cursor: pointer; transition: all 0.2s;
}
.btn:hover { background: var(--accent); border-color: var(--accent); }
.btn.danger:hover { background: var(--danger); border-color: var(--danger); }
.collapsible-content { display: none; padding-top: 10px; border-top: 1px solid var(--border); margin-top: 10px; }
.collapsible-content.show { display: flex; flex-direction: column; gap: 12px; }
.slider-container, .toggle-container, .form-group { display: flex; flex-direction: column; gap: 10px; font-size: 0.875rem; color: var(--text-secondary); }
.toggle-container label { display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
.slider-container label { display: flex; justify-content: space-between; align-items: center; }
.slider-container input[type=range] { width: 100%; }
.side-panel-input, .side-panel-textarea {
  background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-primary); padding: 8px; width: 100%;
}
.side-panel-textarea { resize: vertical; min-height: 80px; }
.responding-indicator { color: var(--text-secondary); font-style: italic; animation: fadeIn 0.5s; }
.responding-indicator .dot { animation: blink 1.4s infinite both; }
.responding-indicator .dot:nth-child(2) { animation-delay: .2s; }
.responding-indicator .dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0; } 40% { opacity: 1; } }
.modal-backdrop {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.7);
  display: none; align-items: center; justify-content: center; z-index: 1000;
}
.modal-content {
  background: var(--bg-secondary); padding: 25px; border-radius: 12px; width: 90%; max-width: 700px;
  border: 1px solid var(--border); box-shadow: var(--shadow); display: flex; flex-direction: column;
  gap: 15px; max-height: 90vh;
}
.modal-body { overflow-y: auto; white-space: pre-wrap; }
.modal-content h2 { font-size: 1.2rem; }
.modal-close-btn { align-self: flex-end; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer; }
.form-group label { font-size: 0.9rem; color: var(--text-secondary); }
.modal-input, .modal-textarea {
  background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px; color: var(--text-primary); font-size: 0.9rem;
}
.modal-textarea { resize: vertical; min-height: 200px; font-family: monospace; }
.modal-footer { display: flex; gap: 10px; justify-content: flex-end; margin-top: 10px; }
.modal-footer .btn-group { display: flex; gap: 10px; flex-grow: 1; }
.icon { width: 20px; height: 20px; fill: currentColor; }
.chat-list { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.chat-list-item {
  padding: 10px; border-radius: 8px; cursor: pointer;
  background: var(--bg-secondary); border: 1px solid transparent; transition: all 0.2s;
  display: flex; justify-content: space-between; align-items: center;
}
.chat-list-item:hover { background: var(--bg-tertiary); border-color: var(--border); }
.chat-list-item.active { background: var(--accent); color: white; border-color: var(--accent); }
.delete-chat-btn {
    font-size: 1.2rem; color: var(--text-secondary); padding: 0 5px; border-radius: 4px;
    line-height: 1; opacity: 0.5; transition: all 0.2s;
}
.chat-list-item:hover .delete-chat-btn { opacity: 1; color: var(--danger); }
.delete-chat-btn:hover { background-color: rgba(255,255,255,0.1); }
#sys-info-content a { color: var(--accent); text-decoration: none; }
#sys-info-content a:hover { text-decoration: underline; }
#changelog-content { font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
.collapse-handle { display: none; }
@media (max-width: 1024px) {
  .side-panel { position: fixed; top: 0; height: 100%; z-index: 500; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
  .side-panel.left-panel { left: 0; transform: translateX(-100%); }
  .side-panel.left-panel.collapsed { transform: translateX(-100%); }
  .container.left-panel-open .side-panel.left-panel { transform: translateX(0); }
  .side-panel.right-panel { right: 0; transform: translateX(100%); }
  .side-panel.right-panel.collapsed { transform: translateX(100%); }
  .container.right-panel-open .side-panel.right-panel { transform: translateX(0); }
  .panel-toggle-handle { display: none; }
  .chat-header-buttons { display: flex !important; gap: 10px; }
  .collapse-handle { display: block; margin-bottom: 15px; }
}
</style>
</head>
<body>
<div id="app-container" class="container">
  <div id="left-panel-toggle" class="panel-toggle-handle left-handle">◀</div>
  <div id="left-panel" class="side-panel left-panel collapsed">
    <button class="btn collapse-handle">Zwiń &times;</button>
    <div class="panel-section">
      <div class="panel-title">Chats</div>
      <ul id="chat-list" class="chat-list"></ul>
      <div class="form-group" style="margin-top: 15px;">
        <input type="text" id="new-chat-name" class="side-panel-input" placeholder="New chat name...">
        <button id="add-chat-btn" class="btn" style="width:100%; margin-top: 8px;">Create Chat</button>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">Persona</div>
      <div class="collapsible-content show">
          <img id="persona-avatar" src="/static/default_avatar.png" style="width:100%; height:auto; aspect-ratio: 1/1; border-radius:12px; object-fit:cover; margin: 0 auto 10px auto; border: 2px solid var(--border);">
          <div style="display:flex; gap: 8px;">
            <select id="side-panel-persona-preset" class="side-panel-input" style="flex:1;"></select>
            <button class="btn" id="side-panel-load-btn">Load</button>
          </div>
          <button class="btn" id="open-persona-modal" style="width:100%; margin-top: 10px;">Full Persona Editor</button>
      </div>
    </div>
  </div>

  <div class="main-panel">
    <div class="chat-header">
      <div class="chat-header-buttons" style="display: none;">
        <button id="mobile-menu-left" class="btn">☰</button>
      </div>
      <h1 id="chat-title">AI Roleplay</h1>
      <div style="display: flex; align-items: center; gap: 15px;">
        <div class="status">
          <span id="status-text">Disconnected</span>
          <span id="status-dot" class="status-dot"></span>
        </div>
        <div class="chat-header-buttons" style="display: none;">
          <button id="mobile-menu-right" class="btn">⚙</button>
        </div>
      </div>
    </div>
    <div id="chat-messages" class="chat-messages"></div>
    <div class="chat-input-container">
      <div class="chat-input-wrapper">
        <textarea id="chat-input" class="chat-input" placeholder="Type your message..." rows="1"></textarea>
        <button id="send-btn" class="send-button" disabled>
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:24px; height:24px;"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
        </button>
        <button id="stop-btn" class="send-button stop-button" style="display: none;">
          <svg class="icon" viewBox="0 0 24 24" fill="currentColor" style="width:24px; height:24px;"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
        </button>
      </div>
    </div>
  </div>

  <div id="right-panel" class="side-panel right-panel collapsed">
    <button class="btn collapse-handle">Zwiń &times;</button>
    <div class="panel-section">
      <div class="panel-title panel-toggle">Quick Actions</div>
      <div class="collapsible-content show">
          <div class="action-buttons">
            <button class="btn" id="reload-chat">Reload</button>
            <button class="btn" id="new-day">New Day</button>
            <button class="btn" id="check-summary">Check Summary</button>
            <button class="btn" id="force-summary-btn">Summarize</button>
            <button class="btn danger" id="clear-memory">Clear Memory</button>
          </div>
          </div>
    </div>
    <div class="panel-section" id="memory-panel" style="display: none;">
      <div class="panel-title panel-toggle">🧠 Retrieved Memories</div>
      <div id="memory-content" class="collapsible-content show" style="font-size: 0.8em; max-height: 250px; overflow-y: auto; background: var(--bg-secondary); padding: 10px; border-radius: 6px;"></div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">Adv Settings</div>
      <div class="collapsible-content">
          <div class="form-group">
            <label for="model-select">Text Model:</label>
            <select id="model-select" class="side-panel-input"></select>
          </div>
          <div class="form-group">
            <label for="embedding-model-select">Embedding Model:</label>
            <select id="embedding-model-select" class="side-panel-input"></select>
          </div>
          <small style="color: var(--text-secondary); font-size: 0.75rem;">(Models must be loaded in LM Studio to appear here)</small>
          <div class="toggle-container">
            <label><span>Persistent Stats</span><input type="checkbox" id="persistent-stats-toggle"></label>
          </div>
          <div class="toggle-container">
            <label><span>Enable Memory (Embeddings)</span><input type="checkbox" id="enable-memory-toggle" checked></label>
          </div>
          <div class="slider-container">
              <label>Temperature: <span id="temp-value">1.0</span></label>
              <input type="range" id="temperature-slider" min="0.1" max="2.5" step="0.05" value="1.0">
          </div>
          <div class="slider-container">
              <label>Max Tokens: <span id="tokens-value">1024</span></label>
              <input type="range" id="tokens-slider" min="256" max="8192" step="128" value="1024">
          </div>
          <div class="slider-container">
              <label>Thought Ratio: <span id="thought-ratio-value">0.5</span></label>
              <input type="range" id="thought-ratio-slider" min="0.0" max="1.5" step="0.05" value="0.5">
          </div>
          <div class="slider-container">
              <label>Talkativeness: <span id="talkativeness-value">0.5</span></label>
              <input type="range" id="talkativeness-slider" min="0.0" max="1.5" step="0.05" value="0.5">
          </div>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">World Events</div>
      <div class="collapsible-content">
        <textarea id="world-event-input" class="side-panel-textarea" placeholder="e.g., The sky suddenly turns crimson."></textarea>
        <div style="display:flex; gap:8px; align-items:center;">
          <select id="event-type-select" class="side-panel-input" style="flex:1;">
            <option value="messages">Msg Count</option>
            <option value="time">Minutes</option>
          </select>
          <input type="number" id="event-value-input" class="side-panel-input" value="3" style="flex:1;">
        </div>
        <button class="btn" id="inject-event-btn">Inject Event</button>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">System Info</div>
      <div class="collapsible-content">
          <div style="font-size: 0.875rem; color: var(--text-secondary);">
            <div style="margin-top: 8px;"><button class="btn" id="test-text-model" style="width: 100%;">Test Selected Text Model</button></div>
            <div style="margin-top: 8px;"><button class="btn" id="test-embed" style="width: 100%;">Test Selected Embedding Model</button></div>
            <div style="margin-top: 8px;"><button class="btn" id="open-sys-info-modal" style="width: 100%;">About This App</button></div>
          </div>
      </div>
    </div>
  </div>
  <div id="right-panel-toggle" class="panel-toggle-handle right-handle">▶</div>
</div>

<div id="persona-modal" class="modal-backdrop">
  <div class="modal-content">
    <button id="persona-modal-close" class="modal-close-btn">&times;</button>
    <h2>Persona Editor</h2>
    <div class="modal-body">
        <div class="form-group">
        <label for="persona-prompt">Generate from simple prompt:</label>
        <div style="display:flex; gap: 10px;">
            <input type="text" id="persona-prompt" class="modal-input" placeholder="e.g., a brutal ninja from japan that is vicious">
            <button id="generate-persona-btn" class="btn">Generate</button>
        </div>
        </div>
        <div class="form-group">
        <label for="persona-editor">Persona JSON (edit directly):</label>
        <textarea id="persona-editor" class="modal-textarea"></textarea>
        </div>
    </div>
    <div class="modal-footer">
      <div class="btn-group">
        <select id="saved-personas-list" class="modal-input"></select>
        <button class="btn" id="load-persona-btn">Load</button>
      </div>
      <div class="btn-group">
        <input type="text" id="save-persona-name" class="modal-input" placeholder="New Persona Name">
        <button class="btn" id="save-persona-btn">Save</button>
      </div>
    </div>
  </div>
</div>

<div id="sys-info-modal" class="modal-backdrop">
  <div class="modal-content">
    <button id="sys-info-modal-close" class="modal-close-btn">&times;</button>
    <h2>About This Application</h2>
    <div id="sys-info-content" class="modal-body">
        <p>This is an advanced, self-hosted AI roleplaying chat server.</p>
        <p><strong>Version:</strong> <span id="sys-info-version"></span> | <strong>LLM:</strong> <span id="sys-info-model"></span></p>
        <p><strong>Author:</strong> <a id="sys-info-author-gh" href="https://github.com/wieszjak" target="_blank">WIESZJAK</a></p>
        <div class="panel-title panel-toggle" id="changelog-toggle">Changelog</div>
        <div id="changelog-content" class="collapsible-content"></div>
    </div>
  </div>
</div>

<div id="summary-modal" class="modal-backdrop">
    <div class="modal-content">
      <button id="summary-modal-close" class="modal-close-btn">&times;</button>
      <h2>Last Summary</h2>
      <div id="summary-modal-body" class="modal-body message-body" style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px;">
        <p>No summary available for this chat yet.</p>
      </div>
    </div>
</div>

<script>



'use strict';
// --- State and Constants ---
if (!window.appState) {
    window.appState = {
        ws: null,
        isGenerating: false,
        currentMessageContainer: null,
        fullResponseText: '',
        activeChatId: 'default_chat',
        isInitialized: false,
        isStreamInitialized: false
    };
}
const SETTINGS_KEY = 'aiRoleplaySettings_v3';
const DOM = {
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    stopBtn: document.getElementById('stop-btn'),
    statusText: document.getElementById('status-text'),
    statusDot: document.getElementById('status-dot'),
    chatTitle: document.getElementById('chat-title'),
    leftPanel: document.getElementById('left-panel'),
    rightPanel: document.getElementById('right-panel'),
    leftPanelToggle: document.getElementById('left-panel-toggle'),
    rightPanelToggle: document.getElementById('right-panel-toggle'),
    mobileMenuLeft: document.getElementById('mobile-menu-left'),
    mobileMenuRight: document.getElementById('mobile-menu-right'),
    appContainer: document.getElementById('app-container'),
    chatList: document.getElementById('chat-list'),
    newChatName: document.getElementById('new-chat-name'),
    addChatBtn: document.getElementById('add-chat-btn'),
    reloadChatBtn: document.getElementById('reload-chat'),
    newDayBtn: document.getElementById('new-day'),
    checkSummaryBtn: document.getElementById('check-summary'),
    forceSummaryBtn: document.getElementById('force-summary-btn'),
    clearMemoryBtn: document.getElementById('clear-memory'),
    modelSelect: document.getElementById('model-select'),
    embeddingModelSelect: document.getElementById('embedding-model-select'),
    persistentStatsToggle: document.getElementById('persistent-stats-toggle'),
    enableMemoryToggle: document.getElementById('enable-memory-toggle'),
    tempSlider: document.getElementById('temperature-slider'),
    tempValue: document.getElementById('temp-value'),
    tokensSlider: document.getElementById('tokens-slider'),
    tokensValue: document.getElementById('tokens-value'),
    thoughtSlider: document.getElementById('thought-ratio-slider'),
    thoughtValue: document.getElementById('thought-ratio-value'),
    talkSlider: document.getElementById('talkativeness-slider'),
    talkValue: document.getElementById('talkativeness-value'),
    personaAvatar: document.getElementById('persona-avatar'),
    sidePanelPersonaPreset: document.getElementById('side-panel-persona-preset'),
    sidePanelLoadBtn: document.getElementById('side-panel-load-btn'),
    openPersonaModalBtn: document.getElementById('open-persona-modal'),
    worldEventInput: document.getElementById('world-event-input'),
    eventTypeSelect: document.getElementById('event-type-select'),
    eventValueInput: document.getElementById('event-value-input'),
    injectEventBtn: document.getElementById('inject-event-btn'),
    testTextModelBtn: document.getElementById('test-text-model'),
    testEmbedBtn: document.getElementById('test-embed'),
    openSysInfoModalBtn: document.getElementById('open-sys-info-modal'),
    personaModal: document.getElementById('persona-modal'),
    personaModalClose: document.getElementById('persona-modal-close'),
    generatePersonaBtn: document.getElementById('generate-persona-btn'),
    personaPrompt: document.getElementById('persona-prompt'),
    personaEditor: document.getElementById('persona-editor'),
    savedPersonasList: document.getElementById('saved-personas-list'),
    loadPersonaBtn: document.getElementById('load-persona-btn'),
    savePersonaName: document.getElementById('save-persona-name'),
    savePersonaBtn: document.getElementById('save-persona-btn'),
    sysInfoModal: document.getElementById('sys-info-modal'),
    sysInfoModalClose: document.getElementById('sys-info-modal-close'),
    summaryModal: document.getElementById('summary-modal'),
    summaryModalClose: document.getElementById('summary-modal-close'),
    summaryModalBody: document.getElementById('summary-modal-body'),
    memoryPanel: document.getElementById('memory-panel'),
    memoryContent: document.getElementById('memory-content')
};

// --- Core Utility Functions (defined first to prevent ReferenceError) ---
function updateStatus(status) {
  DOM.statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  DOM.statusDot.className = 'status-dot ' + status;
  toggleSendStopButtons(status === 'generating');
}

function toggleSendStopButtons(showStop) {
  DOM.sendBtn.style.display = showStop ? 'none' : 'flex';
  DOM.stopBtn.style.display = showStop ? 'flex' : 'none';
}

function simpleMarkdown(text) {
    if (typeof text !== 'string') return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

function parseFullResponse(fullText) {
    let tempText = fullText;
    let thoughts = '', stats = '', finalThoughts = '', cleanContent = '';
    const thinkMatch = tempText.match(/<think\s*>([\s\S]*?)<\/think\s*>/is);
    if (thinkMatch) {
        thoughts = thinkMatch[1].trim();
        tempText = tempText.replace(thinkMatch[0], '');
    }
    const finalThoughtsMatch = tempText.match(/(\*\*\[\[Final Thoughts\]\]\*\*[\s\S]*)/i);
    if (finalThoughtsMatch) {
        finalThoughts = finalThoughtsMatch[0].trim();
        tempText = tempText.substring(0, finalThoughtsMatch.index);
    }
    const statsMatch = tempText.match(/(\*\*\[\[Stats\]\]\*\*[\s\S]*)/i);
    if (statsMatch) {
        stats = statsMatch[0].trim();
        tempText = tempText.substring(0, statsMatch.index);
    }
    cleanContent = tempText.trim();
    return { content: cleanContent, thoughts, stats, final_thoughts: finalThoughts };
}

function updateOrCreateElement(parent, selector, content, position = 'append') {
    if (!content || content.trim() === '') {
        const el = parent.querySelector(selector);
        if (el) el.style.display = 'none';
        return;
    }
    
    let element = parent.querySelector(selector);
    if (!element) {
        element = document.createElement('div');
        element.className = selector.substring(1); // e.g., '.thought-container' -> 'thought-container'
        if (position === 'prepend') parent.prepend(element);
        else parent.appendChild(element);
    }
    
    element.style.display = 'block';
    const htmlContent = simpleMarkdown(content).replace(/\n/g, '<br>');

    if (selector === '.thought-container') {
        const thoughtContentHTML = `<div class="thought-content">${htmlContent}</div>`;
        if (element.innerHTML !== thoughtContentHTML) {
            element.innerHTML = thoughtContentHTML;
        }
        if (!element.hasToggleListener) {
            element.addEventListener('click', () => element.classList.toggle('expanded'));
            element.hasToggleListener = true;
        }
    } else {
        if (element.innerHTML !== htmlContent) {
            element.innerHTML = htmlContent;
        }
    }
}

function renderMessage(msgWrapper, msgData) {
    const msgBody = msgWrapper.querySelector('.message-body');
    if (!msgBody) return;
    msgBody.innerHTML = ''; // Clear for final, clean render

    updateOrCreateElement(msgBody, '.thought-container', msgData.thoughts, 'prepend');
    updateOrCreateElement(msgBody, '.message-content', msgData.content, 'append');
    updateOrCreateElement(msgBody, '.stats-container', msgData.stats, 'append');
    updateOrCreateElement(msgBody, '.final-thoughts-container', msgData.final_thoughts, 'append');
}

// --- Main Application Logic ---
function connectWebSocket() {
    return new Promise((resolve, reject) => {
        if (appState.ws && appState.ws.readyState === WebSocket.OPEN) {
            appState.ws.send(JSON.stringify({ type: 'init', chat_id: appState.activeChatId }));
            resolve();
            return;
        }
        if (appState.ws && (appState.ws.readyState === WebSocket.OPEN || appState.ws.readyState === WebSocket.CONNECTING)) {
    try { appState.ws.send(JSON.stringify({ type: 'init', chat_id: (appState.activeChatId || 'default_chat') })); } catch(e){}
} else {
    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    appState.ws = new WebSocket(wsUrl);
}

        appState.ws.onopen = () => {
            updateStatus('connected');
            DOM.sendBtn.disabled = false;
            appState.ws.send(JSON.stringify({ type: 'init', chat_id: appState.activeChatId }));
            resolve();
        };
        appState.ws.onclose = () => {
            updateStatus('disconnected');
            DOM.sendBtn.disabled = true;
            appState.isGenerating = false;
            setTimeout(connectWebSocket, 3000);
        };
        appState.ws.onerror = (error) => {
            updateStatus('disconnected');
            appState.isGenerating = false;
            console.error("WebSocket Error:", error);
            reject(error);
        };
        appState.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error("Failed to parse WebSocket message:", event.data, e);
            }
        };
    });
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'start': // This is now mainly a server-side confirmation, UI is handled by sendMessage
            appState.isGenerating = true;
            updateStatus('generating');
            appState.fullResponseText = '';
            appState.isStreamInitialized = false;
            if(data.old_ts) {
                document.querySelector(`.message[data-ts="${data.old_ts}"]`)?.remove();
            }
            break;
        case 'partial':
            handlePartialMessage(data.chunk);
            break;
        case 'done':
        case 'stopped':
            appState.isGenerating = false;
            updateStatus('connected');
            if (appState.currentMessageContainer && appState.fullResponseText) {
                // Final render to clean up any streaming artifacts
                renderMessage(appState.currentMessageContainer, parseFullResponse(appState.fullResponseText));
            }
            appState.fullResponseText = '';
            appState.currentMessageContainer = null;
            break;
        case 'error':
            appState.isGenerating = false;
            updateStatus('connected');
            addMessage('system', `[SERVER ERROR] ${data.message}`);
            break;
        case 'user_ts':
            const el = Array.from(document.querySelectorAll('.message.user')).pop();
            if (el && !el.dataset.ts) {
                el.dataset.ts = data.ts;
                addMessageFooter(el, 'user');
            }
            break;
        case 'assistant_ts':
            if (appState.currentMessageContainer) {
                appState.currentMessageContainer.dataset.ts = data.ts;
                addMessageFooter(appState.currentMessageContainer, 'assistant');
            }
            appState.currentMessageContainer = null;
            break;
        case 'memory_info':
            let memoryHtml = data.content.replace('Relevant memories:', '').trim();
            memoryHtml = simpleMarkdown(memoryHtml).replace(/\n/g, '<br>');
            DOM.memoryContent.innerHTML = memoryHtml;
            DOM.memoryPanel.style.display = 'flex';
            break;
    }
}

function handlePartialMessage(chunk){
  if(!appState.currentMessageContainer) return;
  const body = appState.currentMessageContainer.querySelector('.message-body');
  if(!body) return;
  if(!appState.isStreamInitialized){
    body.querySelector('.responding-indicator')?.remove();
    appState.isStreamInitialized = true;
  }
  appState.fullResponseText += (chunk || '');
  const parsed = parseFullResponse(appState.fullResponseText);
  updateOrCreateElement(body, '.thought-container', parsed.thoughts, 'prepend');
  updateOrCreateElement(body, '.message-content', parsed.content, 'append');
  updateOrCreateElement(body, '.stats-container', parsed.stats, 'append');
  updateOrCreateElement(body, '.final-thoughts-container', parsed.final_thoughts, 'append');
  scrollToBottom();
}



function addMessage(role, content = '', ts = '', thoughts = '', stats = '', final_thoughts = '') {
    const msgWrapper = document.createElement('div');
    msgWrapper.className = `message ${role}`;
    if (ts) msgWrapper.dataset.ts = ts;
    const msgBody = document.createElement('div');
    msgBody.className = 'message-body';
    msgWrapper.appendChild(msgBody);
    
    if (role !== 'assistant' || (role === 'assistant' && !appState.isGenerating)) {
        renderMessage(msgWrapper, { content, thoughts, stats, final_thoughts });
    }

    if (role !== 'system' && ts) {
        addMessageFooter(msgWrapper, role);
    }
    DOM.chatMessages.appendChild(msgWrapper);
    scrollToBottom();
    return msgWrapper;
}

function addMessageFooter(msgWrapper, role) {
    msgWrapper.querySelector('.message-footer')?.remove();
    const footer = document.createElement('div');
    footer.className = 'message-footer';
    const timestamp = document.createElement('span');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date(msgWrapper.dataset.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'message-actions';
    actionsContainer.innerHTML = `<button title="Edit">📝</button>${role === 'assistant' ? '<button title="Regenerate">🔄</button>' : ''}<button title="Delete">🗑️</button>`;
    footer.appendChild(timestamp);
    footer.appendChild(actionsContainer);
    msgWrapper.appendChild(footer);
    actionsContainer.querySelector('[title="Delete"]').onclick = () => deleteMessage(msgWrapper.dataset.ts, msgWrapper);
    actionsContainer.querySelector('[title="Edit"]').onclick = () => editMessage(msgWrapper, role);
    if (role === 'assistant') {
        actionsContainer.querySelector('[title="Regenerate"]').onclick = () => regenerateMessage(msgWrapper.dataset.ts);
    }
}

async function deleteMessage(ts, element) {
  if (!ts) return;
  if (confirm('Delete this message?')) {
    await api(`/${appState.activeChatId}/edit_message`, 'POST', {ts});
    element.remove();
  }
}

function editMessage(msgWrapper, role) {
    const msgBody = msgWrapper.querySelector('.message-body');
    if (!msgBody || msgBody.querySelector('textarea')) return;
    const getTextFromContainer = (selector) => {
        const el = msgWrapper.querySelector(selector);
        if (!el || el.style.display === 'none') return '';
        const tempDiv = document.createElement('div');
        const contentSource = el.querySelector('.thought-content') || el;
        tempDiv.innerHTML = contentSource.innerHTML.replace(/<br\s*[\/]?>/gi, "\n");
        return tempDiv.textContent || tempDiv.innerText || "";
    };
    const thoughts = getTextFromContainer('.thought-container');
    const content = getTextFromContainer('.message-content');
    const stats = getTextFromContainer('.stats-container');
    const finalThoughts = getTextFromContainer('.final-thoughts-container');
    let fullRawText = '';
    if (thoughts) fullRawText += `<think>${thoughts.trim()}</think>\n\n`;
    fullRawText += content.trim();
    if (stats) fullRawText += `\n\n${stats.trim()}`;
    if (finalThoughts) fullRawText += `\n\n${finalThoughts.trim()}`;
    const originalHTML = msgBody.innerHTML;
    const editor = document.createElement('textarea');
    editor.className = 'chat-input';
    editor.style.width = '100%';
    editor.value = fullRawText.trim();
    const btnContainer = document.createElement('div');
    btnContainer.style.marginTop = '10px'; btnContainer.style.display = 'flex'; btnContainer.style.gap = '8px';
    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn'; saveBtn.textContent = 'Save';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn danger'; cancelBtn.textContent = 'Cancel';
    btnContainer.appendChild(saveBtn); btnContainer.appendChild(cancelBtn);
    msgBody.innerHTML = ''; msgBody.appendChild(editor); msgBody.appendChild(btnContainer);
    editor.focus(); editor.style.height = 'auto'; editor.style.height = `${editor.scrollHeight}px`;
    cancelBtn.onclick = () => { msgBody.innerHTML = originalHTML; };
    saveBtn.onclick = async () => {
        const newRawContent = editor.value;
        const result = await api(`/${appState.activeChatId}/edit_message`, 'POST', { ts: msgWrapper.dataset.ts, raw_content: newRawContent });
        renderMessage(msgWrapper, result.updated_message);
        addMessageFooter(msgWrapper, role);
    };
}

async function regenerateMessage(ts) {
    if (!ts || appState.isGenerating) return;
    if (!confirm('Regenerate this response? The current one will be deleted.')) return;
    appState.ws.send(JSON.stringify({ type: 'regenerate', ts, chat_id: appState.activeChatId, settings: getSettings() }));
    // Immediately show visual feedback for regeneration
    appState.isGenerating = true;
    updateStatus('generating');
    const oldMessage = document.querySelector(`.message[data-ts="${ts}"]`);
    if(oldMessage) {
        appState.currentMessageContainer = oldMessage;
        const body = oldMessage.querySelector('.message-body');
        if(body) body.innerHTML = '<div class="responding-indicator"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>';
    }
}

function getSettings() {
    return {
      model: DOM.modelSelect.value,
      embedding_model: DOM.embeddingModelSelect.value,
      temperature: parseFloat(DOM.tempSlider.value),
      max_tokens: parseInt(DOM.tokensSlider.value),
      thought_ratio: parseFloat(DOM.thoughtSlider.value),
      talkativeness: parseFloat(DOM.talkSlider.value),
      persistent_stats: DOM.persistentStatsToggle.checked,
      enable_memory: DOM.enableMemoryToggle.checked,
    };
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(getSettings()));
}

function loadSettings() {
    const savedSettings = localStorage.getItem(SETTINGS_KEY);
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            if(settings.model) DOM.modelSelect.value = settings.model;
            if(settings.embedding_model) DOM.embeddingModelSelect.value = settings.embedding_model;
            DOM.tempSlider.value = settings.temperature || 1.0;
            DOM.tokensSlider.value = settings.max_tokens || 1024;
            DOM.thoughtSlider.value = settings.thought_ratio || 0.5;
            DOM.talkSlider.value = settings.talkativeness || 0.5;
            DOM.persistentStatsToggle.checked = settings.persistent_stats === true;
            DOM.enableMemoryToggle.checked = settings.enable_memory !== false;
            DOM.tempValue.textContent = DOM.tempSlider.value;
            DOM.tokensValue.textContent = DOM.tokensSlider.value;
            DOM.thoughtValue.textContent = DOM.thoughtSlider.value;
            DOM.talkValue.textContent = DOM.talkSlider.value;
        } catch (e) { console.error("Failed to load settings", e); }
    }
}

function scrollToBottom() {
    setTimeout(() => {
        DOM.chatMessages.scrollTo({ top: DOM.chatMessages.scrollHeight, behavior: 'smooth' });
    }, 100);
}

async function sendMessage() {
    const message = DOM.chatInput.value.trim();
    if (!message || appState.isGenerating) return;
    DOM.memoryPanel.style.display = 'none';
    DOM.memoryContent.innerHTML = '';
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';
    addMessage('user', message, new Date().toISOString());
    try {
        await connectWebSocket();
        appState.ws.send(JSON.stringify({ type: 'message', message, settings: getSettings(), chat_id: appState.activeChatId }));
        // Start "responding" animation immediately for better UX
        appState.isGenerating = true;
        updateStatus('generating');
        appState.currentMessageContainer = addMessage('assistant');
const body = appState.currentMessageContainer.querySelector('.message-body');
// Pre-create streaming skeleton so structure is visible from the start
body.innerHTML = [
  '<div class="thought-container"><div class="thought-content">...</div></div>',
  '<div class="message-content">...</div>',
  '<div class="stats-container"><strong>[[Stats]]</strong><br>...</div>',
  '<div class="final-thoughts-container"><strong>[[Final Thoughts]]</strong><br>...</div>'
].join('');
    } catch (error) {
        addMessage('system', '[ERROR] Connection failed. Please check the server.');
    }
}

function stopGeneration() {
  if (appState.ws && appState.ws.readyState === WebSocket.OPEN && appState.isGenerating) {
    appState.ws.send(JSON.stringify({ type: 'stop', chat_id: appState.activeChatId }));
  }
}

async function api(path, method = 'GET', body = null) {
  try {
    const opts = { method };
    if (body) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
    }
    const response = await fetch(path, opts);
    const contentType = response.headers.get("content-type");
    const isJson = contentType && contentType.includes("application/json");
    if (!response.ok) {
        const errorData = isJson ? await response.json() : await response.text();
        const errorText = isJson ? errorData.error || JSON.stringify(errorData) : errorData;
        const errorMsg = `[API ERROR] ${response.status}: ${errorText}`;
        addMessage('system', errorMsg);
        console.error("API Error Response:", errorData);
        throw new Error(`API Error: ${response.status} ${errorText}`);
    }
    return isJson ? await response.json() : await response.text();
  } catch (error) {
      console.error("API call failed:", error);
      addMessage('system', `[API ERROR] Failed to fetch from ${path}. Check server console & .env config.`);
      throw error;
  }
}

async function clearMemory() {
  if (!confirm('Clear all memory for this chat? This deletes conversation history, summaries, stats and events.')) return;
  await api(`/${appState.activeChatId}/clear_memory`, 'POST');
  await reloadChat();
}

async function testEmbeddings() {
    const selectedModel = DOM.embeddingModelSelect.value;
    if (!selectedModel) {
        alert("Please select an embedding model to test.");
        return;
    }
    addMessage('system', `Testing embedding model: ${selectedModel}...`);
    try {
        const res = await api('/test_embeddings', 'POST', { model: selectedModel });
        addMessage('system', res.success ? `✅ Embedding test successful!` : `❌ Embedding test failed: ${res.error}`);
    } catch(e) {
        console.error("Failed to test embeddings:", e);
    }
}

async function loadAvailableModels() {
    try {
        const data = await api('/models');
        DOM.modelSelect.innerHTML = '';
        DOM.embeddingModelSelect.innerHTML = '';
        if (data.models && data.models.length > 0) {
            data.models.forEach(modelId => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = modelId;
                DOM.modelSelect.appendChild(option.cloneNode(true));
                DOM.embeddingModelSelect.appendChild(option);
            });
        } else {
             const errorHtml = '<option value="">No models found</option>';
             DOM.modelSelect.innerHTML = errorHtml;
             DOM.embeddingModelSelect.innerHTML = errorHtml;
        }
    } catch (e) {
        const errorHtml = '<option value="">Error loading models</option>';
        DOM.modelSelect.innerHTML = errorHtml;
        DOM.embeddingModelSelect.innerHTML = errorHtml;
    }
    loadSettings();
}

function setupPanelToggles() {
    document.querySelectorAll('.panel-toggle').forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            const content = e.target.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('show');
                e.target.classList.toggle('collapsed');
            }
        });
    });
    const toggleLogic = (panel, container, className) => {
        const isOpening = !container.classList.contains(className);
        const otherPanelClass = className === 'left-panel-open' ? 'right-panel-open' : 'left-panel-open';
        if (isOpening && window.innerWidth <= 1024) {
            container.classList.remove(otherPanelClass);
            if(otherPanelClass === 'left-panel-open') DOM.leftPanel.classList.add('collapsed');
            else DOM.rightPanel.classList.add('collapsed');
        }
        panel.classList.toggle('collapsed');
        container.classList.toggle(className);
    };
    DOM.leftPanelToggle.addEventListener('click', () => toggleLogic(DOM.leftPanel, DOM.appContainer, 'left-panel-open'));
    DOM.rightPanelToggle.addEventListener('click', () => toggleLogic(DOM.rightPanel, DOM.appContainer, 'right-panel-open'));
    DOM.mobileMenuLeft.addEventListener('click', () => toggleLogic(DOM.leftPanel, DOM.appContainer, 'left-panel-open'));
    DOM.mobileMenuRight.addEventListener('click', () => toggleLogic(DOM.rightPanel, DOM.appContainer, 'right-panel-open'));
    document.querySelectorAll('.collapse-handle').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const panel = e.target.closest('.side-panel');
            if (panel.id === 'left-panel') DOM.appContainer.classList.remove('left-panel-open');
            else if (panel.id === 'right-panel') DOM.appContainer.classList.remove('right-panel-open');
            panel.classList.add('collapsed');
        });
    });
}
function autoResizeTextarea(el) {
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 240) + 'px';
}

async function reloadChat() {
    if (!appState.activeChatId) return;
    try {
        const messages = await api(`/${appState.activeChatId}/messages`);
        DOM.chatMessages.innerHTML = '';
        (messages || []).forEach(msg => {
            if (!msg) return;
            const finalThoughts = msg.final_thoughts || msg.finalThoughts || '';
            addMessage(
                msg.role || 'assistant',
                msg.content || '',
                msg.ts,
                msg.thoughts || '',
                msg.stats || '',
                finalThoughts
            );
        });
        scrollToBottom();
    } catch (error) {
        console.error('Failed to reload chat history:', error);
        addMessage('system', '[ERROR] Failed to load chat history.');
    }
}

async function loadSavedPersonasIntoSelect(selectElement) {
    if (!selectElement) return;
    try {
        const personas = await api('/personas');
        selectElement.innerHTML = '';
        if (personas && personas.length) {
            personas.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                selectElement.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No saved personas';
            selectElement.appendChild(option);
        }
    } catch (error) {
        console.error('Failed to load saved personas:', error);
        selectElement.innerHTML = '<option value="">Error loading personas</option>';
    }
}

async function refreshPersonaLists() {
    await loadSavedPersonasIntoSelect(DOM.savedPersonasList);
    await loadSavedPersonasIntoSelect(DOM.sidePanelPersonaPreset);
}

async function refreshActivePersona() {
    try {
        const persona = await api(`/${appState.activeChatId}/persona`);
        DOM.personaEditor.value = JSON.stringify(persona, null, 2);
        DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
        return persona;
    } catch (error) {
        console.error('Failed to load active persona:', error);
        addMessage('system', '[ERROR] Unable to load the active persona.');
        return null;
    }
}
async function loadAndActivatePersona(name) {
    if (!name) {
        alert('Please select a persona to load.');
        return;
    }
    try {
        const persona = await api(`/personas/${encodeURIComponent(name)}`);
        await api(`/${appState.activeChatId}/persona`, 'POST', persona);
        DOM.personaEditor.value = JSON.stringify(persona, null, 2);
        DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
        addMessage('system', `Persona "${name}" loaded for this chat.`);
    } catch (error) {
        console.error('Failed to load persona:', error);
        alert('Failed to load persona. Please check the server logs.');
    }
}

async function generatePersonaFromPrompt() {
    const prompt = DOM.personaPrompt.value.trim();
    if (!prompt) {
        alert('Please provide a short description to generate a persona.');
        return;
    }
    DOM.generatePersonaBtn.disabled = true;
    DOM.generatePersonaBtn.textContent = 'Generating...';
    try {
        const result = await api('/generate_persona', 'POST', { description: prompt });
        if (result && result.persona) {
            DOM.personaEditor.value = JSON.stringify(result.persona, null, 2);
            DOM.personaAvatar.src = `/static/${result.persona.avatar || 'default_avatar.png'}`;
        }
    } catch (error) {
        console.error('Failed to generate persona:', error);
        alert('Failed to generate persona. Please try again.');
    } finally {
        DOM.generatePersonaBtn.textContent = 'Generate';
        DOM.generatePersonaBtn.disabled = false;
    }
}

async function savePersona() {
    const name = DOM.savePersonaName.value.trim();
    if (!name) {
        alert('Please enter a name to save the persona.');
        return;
    }
    let persona;
    try {
        persona = JSON.parse(DOM.personaEditor.value || '{}');
    } catch (error) {
        alert('Persona JSON is invalid. Please correct it before saving.');
        return;
    }
    try {
        await api(`/personas/${encodeURIComponent(name)}`, 'POST', persona);
        await refreshPersonaLists();
        DOM.savePersonaName.value = '';
        addMessage('system', `Persona "${name}" saved.`);
    } catch (error) {
        console.error('Failed to save persona:', error);
        alert('Failed to save persona.');
    }
}

async function injectWorldEvent() {
    const eventText = DOM.worldEventInput.value.trim();
    if (!eventText) {
        alert('Please describe the world event before injecting it.');
        return;
    }
    try {
        await api(`/${appState.activeChatId}/inject_event`, 'POST', {
            event: eventText,
            type: DOM.eventTypeSelect.value,
            value: parseInt(DOM.eventValueInput.value, 10)
        });
        DOM.worldEventInput.value = '';
        addMessage('system', `[WORLD EVENT INJECTED] ${eventText}`);
    } catch (error) {
        console.error('Failed to inject world event:', error);
        alert('Failed to inject world event.');
    }
}

async function markNewDay() {
    try {
        const response = await api(`/${appState.activeChatId}/new_day`, 'POST');
        if (response.marker) {
            addMessage('system', response.marker);
        }
        if (response.summary) {
            addMessage('system', `Summary Generated:\n${response.summary}`);
        }
        await reloadChat();
    } catch (error) {
        console.error('Failed to mark a new day:', error);
        alert('Failed to mark a new day.');
    }
}

async function checkSummary() {
    try {
        const result = await api(`/${appState.activeChatId}/last_summary`);
        const summary = result && result.summary ? result.summary.trim() : '';
        DOM.summaryModalBody.textContent = summary || 'No summary available yet.';
        DOM.summaryModal.style.display = 'flex';
    } catch (error) {
        console.error('Failed to fetch last summary:', error);
        alert('Failed to fetch the last summary.');
    }
}

async function testTextModel() {
    const selectedModel = DOM.modelSelect.value;
    if (!selectedModel) {
        alert('Please select a text model to test.');
        return;
    }
    addMessage('system', `Testing text model: ${selectedModel}...`);
    try {
        const res = await api('/test_text_model', 'POST', { model: selectedModel });
        addMessage('system', res.success ? '✅ Text model test successful!' : `❌ Text model test failed: ${res.error}`);
    } catch (error) {
        console.error('Failed to test text model:', error);
    }
}

function openModal(modal) {
    if (modal) modal.style.display = 'flex';
}

function closeModal(modal) {
    if (modal) modal.style.display = 'none';
}

function setupEventListeners() {
    DOM.sendBtn.addEventListener('click', sendMessage);
    DOM.stopBtn.addEventListener('click', stopGeneration);

    DOM.chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    DOM.chatInput.addEventListener('input', () => {
        const hasText = DOM.chatInput.value.trim().length > 0;
        DOM.sendBtn.disabled = !hasText || appState.isGenerating;
        autoResizeTextarea(DOM.chatInput);
    });

    DOM.addChatBtn.addEventListener('click', async () => {
        const name = DOM.newChatName.value.trim();
        if (!name) {
            alert('Please enter a chat name.');
            return;
        }
        try {
            const res = await api('/chats/create', 'POST', { name });
            DOM.newChatName.value = '';
            await loadChatList();
            if (res && res.chat_id) {
                await switchChat(res.chat_id);
            }
        } catch (error) {
            console.error('Failed to create chat:', error);
            alert('Failed to create chat.');
        }
    });

    DOM.reloadChatBtn.addEventListener('click', reloadChat);
    DOM.clearMemoryBtn.addEventListener('click', clearMemory);
    DOM.newDayBtn.addEventListener('click', markNewDay);
    DOM.checkSummaryBtn.addEventListener('click', checkSummary);
    DOM.forceSummaryBtn.addEventListener('click', forceSummarize);

    DOM.injectEventBtn.addEventListener('click', injectWorldEvent);

    DOM.testTextModelBtn.addEventListener('click', testTextModel);
    DOM.testEmbedBtn.addEventListener('click', testEmbeddings);

    DOM.openPersonaModalBtn.addEventListener('click', async () => {
        openModal(DOM.personaModal);
        await refreshPersonaLists();
    });
    DOM.personaModalClose.addEventListener('click', () => closeModal(DOM.personaModal));

    DOM.sidePanelLoadBtn.addEventListener('click', () => loadAndActivatePersona(DOM.sidePanelPersonaPreset.value));
    DOM.loadPersonaBtn.addEventListener('click', () => loadAndActivatePersona(DOM.savedPersonasList.value));
    DOM.generatePersonaBtn.addEventListener('click', generatePersonaFromPrompt);
    DOM.savePersonaBtn.addEventListener('click', savePersona);

    DOM.openSysInfoModalBtn.addEventListener('click', async () => {
        openModal(DOM.sysInfoModal);
        try {
            const info = await api('/system_info');
            const versionEl = document.getElementById('sys-info-version');
            const modelEl = document.getElementById('sys-info-model');
            if (versionEl) versionEl.textContent = info.version || 'unknown';
            if (modelEl) modelEl.textContent = info.model_name || 'unknown';
        } catch (error) {
            console.error('Failed to fetch system info:', error);
        }
    });
    DOM.sysInfoModalClose.addEventListener('click', () => closeModal(DOM.sysInfoModal));

    DOM.summaryModalClose.addEventListener('click', () => closeModal(DOM.summaryModal));

    const settingsControls = [
        DOM.modelSelect,
        DOM.embeddingModelSelect,
        DOM.tempSlider,
        DOM.tokensSlider,
        DOM.thoughtSlider,
        DOM.talkSlider,
        DOM.persistentStatsToggle,
        DOM.enableMemoryToggle
    ];
    const sliderLabels = {
        [DOM.tempSlider.id]: DOM.tempValue,
        [DOM.tokensSlider.id]: DOM.tokensValue,
        [DOM.thoughtSlider.id]: DOM.thoughtValue,
        [DOM.talkSlider.id]: DOM.talkValue
    };
    settingsControls.forEach(control => {
        control.addEventListener('input', (event) => {
            if (event.target.type === 'range' && sliderLabels[event.target.id]) {
                sliderLabels[event.target.id].textContent = event.target.value;
            }
            saveSettings();
        });
    });
    autoResizeTextarea(DOM.chatInput);
    DOM.chatInput.dispatchEvent(new Event('input'));
}

async function loadChatList() {
    const chats = await api('/chats');
    DOM.chatList.innerHTML = '';
    if (chats.length === 0) {
        await api('/chats/create', 'POST', { name: 'default_chat' });
        return loadChatList();
    }
    chats.forEach(chatId => {
        const li = document.createElement('li');
        li.className = 'chat-list-item'; li.dataset.chatId = chatId;
        const nameSpan = document.createElement('span');
        nameSpan.textContent = chatId; nameSpan.style.flexGrow = '1'; nameSpan.style.overflow = 'hidden'; nameSpan.style.textOverflow = 'ellipsis';
        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-chat-btn'; deleteBtn.innerHTML = '&times;'; deleteBtn.title = `Delete chat "${chatId}"`;
        li.appendChild(nameSpan); li.appendChild(deleteBtn);
        if (chatId === appState.activeChatId) li.classList.add('active');
        li.addEventListener('click', async (e) => {
            if (e.target !== deleteBtn) {
                try {
                    await switchChat(chatId);
                } catch (error) {
                    console.error('Failed to switch chat:', error);
                }
            }
        });
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                await deleteChat(chatId);
            } catch (error) {
                console.error('Failed to delete chat:', error);
            }
        });
        DOM.chatList.appendChild(li);
    });
}

async function deleteChat(chatId) {
    if (appState.isGenerating) { alert("Cannot delete a chat while a response is being generated."); return; }
    if (!confirm(`Are you sure you want to permanently delete the chat "${chatId}"? This cannot be undone.`)) return;
    await api(`/chats/${chatId}`, 'DELETE');
    if (appState.activeChatId === chatId) {
        const chats = await api('/chats');
        const newActiveChat = chats.length > 0 ? chats[0] : 'default_chat';
        await switchChat(newActiveChat);
    }
    await loadChatList();
}

async function switchChat(chatId) {
    if (!chatId) return;
    if (appState.activeChatId === chatId && appState.ws && appState.ws.readyState === WebSocket.OPEN) return;
    appState.activeChatId = chatId;
    DOM.chatTitle.textContent = `Chat: ${chatId}`;
    console.log(`Switching to chat: ${appState.activeChatId}`);
    document.querySelectorAll('.chat-list-item.active').forEach(el => el.classList.remove('active'));
    document.querySelector(`.chat-list-item[data-chat-id="${chatId}"]`)?.classList.add('active');
    appState.isGenerating = false;
    stopGeneration();
    await connectWebSocket();
    await reloadChat();
    await refreshActivePersona();
    DOM.chatInput.value = '';
    DOM.chatInput.dispatchEvent(new Event('input'));
}

async function forceSummarize() {
    addMessage('system', 'Generating summary using LLM...');
    const response = await api(`/${appState.activeChatId}/force_summarize`, 'POST');
    addMessage('system', `Summary Generated:\n${response.summary}`);
}

async function initializeApp() {
    if (window.appState.isInitialized) return;
    window.appState.isInitialized = true;
    try {
        setupEventListeners();
        setupPanelToggles();
        await loadAvailableModels();
        await loadChatList();
        if (!document.querySelector('.chat-list-item.active')) {
            const firstChat = document.querySelector('.chat-list-item');
            appState.activeChatId = firstChat ? firstChat.dataset.chatId : 'default_chat';
            firstChat?.classList.add('active');
        }
        DOM.chatTitle.textContent = `Chat: ${appState.activeChatId}`;
        DOM.persistentStatsToggle.checked = __PERSISTENT_STATS_ENABLED__;
        await connectWebSocket();
        await reloadChat();
        await refreshPersonaLists();
        await refreshActivePersona();
    } catch (error) {
        console.error("Initialization failed:", error);
        addMessage('system', 'A critical error occurred during initialization. Some UI elements may not work. Please check the browser console (F12) for details.');
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);



</script>
</body>
</html>
"""

# ---------------- HTTP endpoints ----------------
def get_safe_chat_id(chat_id: str):
    safe_chat_id = re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)
    return safe_chat_id or DEFAULT_CHAT_ID

@app.get("/", response_class=HTMLResponse)
async def index():
    final_html = HTML_UI.replace('__PERSISTENT_STATS_ENABLED__', "true" if PERSISTENT_STATS_ENABLED else "false")
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