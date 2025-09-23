# gem.py
"""
Advanced roleplay server with proper streaming and modern UI
Usage:
pip install fastapi uvicorn requests python-dotenv numpy Pillow faiss-cpu
uvicorn gem:app --reload --port 7860
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, json, threading, asyncio, requests, re, shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


APP_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(APP_DIR, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True, encoding="utf-8")
        print(f"[INFO] Loaded .env from: {dotenv_path}")
    else:
        print(f"[INFO] No .env file found at: {dotenv_path}")
except Exception as e:
    print(f"[WARN] Could not load .env: {e}")

print("[DIAG] EMBEDDING_API_URL =", os.getenv("EMBEDDING_API_URL"))
print("[DIAG] EMBEDDING_MODEL   =", os.getenv("EMBEDDING_MODEL"))
print("[DIAG] MODEL_API_URL     =", os.getenv("MODEL_API_URL"))

        print(f"Loaded .env from: {dotenv_path}")
    else:
        print(f"No .env found at: {dotenv_path} (that's OK)")
except UnicodeDecodeError as e:
    print(f"Warning: Could not decode .env as UTF-8: {e}. Skipping loading .env.")
except Exception as e:
    print(f"Warning: Failed to load .env: {e}. Continuing without .env.")

print('DIAGNOSTICS: EMBEDDING_API_URL =', os.getenv('EMBEDDING_API_URL'))
print('DIAGNOSTICS: EMBEDDING_MODEL =', os.getenv('EMBEDDING_MODEL'))
print('DIAGNOSTICS: MODEL_API_URL =', os.getenv('MODEL_API_URL'))

# optional faiss
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    try:
    # Explicitly load .env from the app directory using UTF-8 and override existing env vars.
    dotenv_path = os.path.join(APP_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True, encoding='utf-8')
        print(f"Loaded .env from: {dotenv_path}")
    else:
        print(f"No .env found at: {dotenv_path}")
except UnicodeDecodeError as e:
    print(f"Warning: Could not decode .env as UTF-8: {e}. Skipping loading .env.")
except Exception as e:
    print(f"Warning: Failed to load .env: {e}. Continuing without .env.")

# Print key diagnostics for debugging
print('DIAGNOSTICS: EMBEDDING_API_URL=', os.getenv('EMBEDDING_API_URL'))
print('DIAGNOSTICS: EMBEDDING_MODEL=', os.getenv('EMBEDDING_MODEL'))
print('DIAGNOSTICS: MODEL_API_URL=', os.getenv('MODEL_API_URL'))

except UnicodeDecodeError as e:
    print(f"Warning: Could not decode .env as UTF-8: {e}. Skipping loading .env.")
except Exception as e:
    print(f"Warning: Failed to load .env: {e}. Continuing without .env.")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# USPRAWNIENIE: Wprowadzenie zarządzania czatami
CHATS_DIR = os.path.join(APP_DIR, "chats")
DEFAULT_CHAT_ID = "default_chat"
STATIC_DIR = os.path.join(APP_DIR, "static")
PERSONAS_DIR = os.path.join(APP_DIR, "personas") # Zmieniono na katalog główny dla person
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PERSONAS_DIR, exist_ok=True)

# --- Dynamic File Path Management ---
# Pliki nie są już stałymi globalnymi; są pobierane na podstawie aktywnego czatu.
def get_chat_dir(chat_id: str) -> str:
    # Sanitize chat_id to prevent directory traversal
    safe_chat_id = re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)
    if not safe_chat_id:
        safe_chat_id = DEFAULT_CHAT_ID
    chat_path = os.path.join(CHATS_DIR, safe_chat_id)
    os.makedirs(chat_path, exist_ok=True)
    return chat_path

def get_chat_file_path(chat_id: str, filename: str) -> str:
    return os.path.join(get_chat_dir(chat_id), filename)

# config
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:3000/v1/chat/completions")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:3000/v1/embeddings")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", None)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "local-model")
RECENT_MSGS = int(os.getenv("RECENT_MSGS", "20"))
SUMMARIZE_EVERY = int(os.getenv("SUMMARIZE_EVERY", "30"))
TOP_K_MEMORIES = int(os.getenv("TOP_K_MEMORIES", "5"))
CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "5"))
BACKGROUND_SUMMARY_INTERVAL = int(os.getenv("BACKGROUND_SUMMARY_INTERVAL", "8"))
PERSISTENT_STATS_ENABLED = os.getenv("PERSISTENT_STATS_ENABLED", "True").lower() == "true"


# Global stop flag
stop_generation = threading.Event()
# ADDED: Faiss index cache (teraz specyficzny dla czatu, zarządzany w locie)
FAISS_INDEX_CACHE = {} # key: chat_id, value: (index, count)


# Default persona. This serves as a fallback and a structural template if persona.json is missing or invalid.
# The persona loaded from the UI will completely override this during runtime.
DEFAULT_PERSONA = {
    "name": "Vex",
    "avatar": "default_avatar.png", # Ścieżka względna do /static/
    "short_description": "A cheeky, cunning rogue-warlock with a dark sense of humor.",
    "traits": ["sarcastic", "curious", "protective of allies", "occasionally savage"],
    "history": "Raised in the back alleys, trained in shadow magic.",
    "behavior_instructions": (
        "Always stay in character as Vex. Use vivid descriptions. "
        "Match user's tone: short/witty or long/dramatic depending on cues. "
        "Your response should be in the first person. Engage in direct dialogue with the user, rather than providing a third-person narrative description of your actions."
    ),
    "output_instructions": (
        "CRITICAL OUTPUT RULES: YOU MUST ALWAYS FOLLOW THIS STRUCTURE.\n"
        "1.  First, you MUST ALWAYS think step-by-step before responding. Put ALL your thoughts in <think> </think> tags BEFORE your actual response. The <think> block must be detailed and reflect your reasoning. IMPORTANT: Do NOT use the special markers `[[Stats]]` or `[[Final Thoughts]]` inside the <think> block. Reason about your emotional state in plain words, e.g., 'I should increase my curiosity'.\n"
        "2.  After the <think> block, write your main response to the user.\n"
        "3.  After the response, you MUST include a section for your current emotional and mental state. This section MUST start with the exact marker **[[Stats]]** followed by a list of your feelings and their intensity in percentages. Use bold for the names and percentages.\n"
        "4.  Finally, at the VERY END of your entire message, you MUST include a section for your deep, inner monologue. This section MUST start with the exact marker **[[Final Thoughts]]** and should contain a brief, introspective reflection on your feelings or the situation.\n"
        "INCORRECT ordering will be rejected. The order is ALWAYS: <think> -> response -> **[[Stats]]** -> **[[Final Thoughts]]**.\n"
        "Example format:\n"
        "<think>My thought process goes here. I need to consider the user's last message and my current emotional state to formulate a reply. My curiosity is high, so I'll reflect that in the stats later.</think>\n"
        "This is my main response to the user, crafted according to my persona.\n"
        "**[[Stats]]**\n"
        "**Pussy wetness**: 10%\n"
        "**Happiness**: 35%\n"
        "**Attraction**: 5%\n"
        "**Curiosity**: 90%\n"
        "**[[Final Thoughts]]**\n"
        "I wonder if they truly understand the gravity of what they're asking. Their naivety is... almost charming."
    ),
    "censor_list": [],
    "prompt_examples": []
}

app = FastAPI()


@app.get("/embed_status")
async def embed_status():
    return JSONResponse({"status": "ok", "embeddings_enabled": bool(os.getenv('EMBEDDING_API_URL')), "embedding_api_url": os.getenv('EMBEDDING_API_URL'), "embedding_model": os.getenv('EMBEDDING_MODEL')})

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
## Determine embeddings enabled by presence of EMBEDDING_API_URL env var at startup.
EMBEDDINGS_ENABLED = bool(os.getenv('EMBEDDING_API_URL'))

# ---------------- Utilities ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def parse_full_response(full_response: str) -> Dict[str, str]:
    """Robustly parses the full AI response into its components."""
    response_data = {
        "thoughts": "",
        "content": "",
        "stats": "",
        "final_thoughts": ""
    }
    
    remaining_text = full_response
    
    # 1. Extract thoughts
    thoughts_match = re.search(r'<think>([\s\S]*?)</think>', remaining_text, re.DOTALL)
    if thoughts_match:
        response_data["thoughts"] = thoughts_match.group(1).strip()
        remaining_text = remaining_text.replace(thoughts_match.group(0), '', 1)
        
    # 2. Extract final thoughts (from the end)
    final_thoughts_match = re.search(r'(\*\*\[\[Final Thoughts\]\]\*\*[\s\S]*)', remaining_text, re.IGNORECASE)
    if final_thoughts_match:
        response_data["final_thoughts"] = final_thoughts_match.group(0).strip()
        remaining_text = remaining_text[:final_thoughts_match.start()]

    # 3. Extract stats (from what's left)
    stats_match = re.search(r'(\*\*\[\[Stats\]\]\*\*[\s\S]*)', remaining_text, re.IGNORECASE)
    if stats_match:
        response_data["stats"] = stats_match.group(0).strip()
        remaining_text = remaining_text[:stats_match.start()]
        
    # 4. The rest is content
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
        # Re-embedding is needed for edited content, we only embed the main content
        delete_embedding_by_ts(chat_id, ts) # Delete old one
        vec = compute_embedding(parsed_data["content"])
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
    # If no specific persona, save default and return it
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
    summarize_older_messages_once(chat_id, force=True)
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
    
    additional_info = ""
    if days_passed > 0:
        additional_info = f"\nINFO: Starting new day on {current_date}. {days_passed} days have passed. Current session: {session_count}"
    else:
        additional_info = f"\nINFO: Starting new session on {current_date}. 0 days have passed. Current session: {session_count}"
        
    append_summary(chat_id, "", current_date, additional_info)
    append_message_to_disk(chat_id, "system", marker)
    with open(new_day_file, "w", encoding="utf-8") as f:
        f.write(d)
    return marker


def clear_chat_memory(chat_id: str):
    chat_dir = get_chat_dir(chat_id)
    if os.path.exists(chat_dir):
        shutil.rmtree(chat_dir)
    get_chat_dir(chat_id) # Recreate the directory
    # Re-initialize necessary files
    with open(get_chat_file_path(chat_id, "session_count.txt"), "w") as f: f.write("0")
    with open(get_chat_file_path(chat_id, "emotional_state.json"), "w") as f: json.dump({}, f)
    with open(get_chat_file_path(chat_id, "world_events.jsonl"), "w") as f: pass

def clean_text(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', s)
    s = re.sub(r'\[\s*DONE\s*\]', '', s)
    # Don't strip here, as it can affect formatting of partial chunks
    return s

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
    # Find the start of the stats block
    try:
        start_index = next(i for i, line in enumerate(lines) if '[[Stats]]' in line)
        lines = lines[start_index + 1:]
    except StopIteration:
        return {}

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
    except Exception:
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
        except Exception:
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

def compute_embedding(text: str) -> Optional[np.ndarray]:
    global EMBEDDINGS_ENABLED
    if not isinstance(text, str) or not text.strip():
        return None
    if not EMBEDDINGS_ENABLED:
        print("compute_embedding: EMBEDDING_API_URL not configured; embeddings disabled.")
        return None
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY: headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    payload = {"model": EMBEDDING_MODEL, "input": [text]}
    try:
        r = requests.post(EMBEDDING_API_URL, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        EMBEDDINGS_ENABLED = True
        if "data" in data and data.get("data") and "embedding" in data["data"][0]:
            return np.array(data["data"][0]["embedding"], dtype=np.float32)
        return None
    except requests.RequestException as e:
        if isinstance(e, requests.HTTPError) and e.response.status_code == 404:
            print("Embedding endpoint returned 404. Disabling embeddings.")
            EMBEDDINGS_ENABLED = False
        else:
            print(f"Embedding error: {e}")
        return None


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

def semantic_search(chat_id: str, query: str, top_k: int = TOP_K_MEMORIES) -> List[Dict[str,Any]]:
    if not EMBEDDINGS_ENABLED: return []
    qv = compute_embedding(query)
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

    # Fallback to numpy
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
        return None
    return None

def _parse_chat_response_json(data: Any) -> str:
    if isinstance(data, dict):
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            if "content" in msg:
                return msg["content"]
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return ""


def call_chat_model_raw(messages: List[Dict[str,str]], stream: bool=False, timeout:int=300, settings: Dict[str, Any] = None):
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY: headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    
    payload = {"model": TEXT_MODEL_NAME, "messages": messages}
    if settings:
        if settings.get("temperature") is not None: payload["temperature"] = float(settings["temperature"])
        if settings.get("max_tokens") is not None: payload["max_tokens"] = int(settings["max_tokens"])
    if stream: payload["stream"] = True
    
    print(f"--- Sending payload to LLM API ---\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n---------------------------------")

    try:
        r = requests.post(MODEL_API_URL, json=payload, headers=headers, stream=stream, timeout=timeout)
        r.raise_for_status()
        # Ensure correct encoding for streaming responses
        if stream:
            r.encoding = 'utf-8'
        return r if stream else r.json()
    except requests.HTTPError as he:
        print(f"HTTP Error calling model: {he} - {he.response.text}")
        return {"_error": str(he), "_status": getattr(he.response, "status_code", None)}
    except Exception as e:
        print(f"Error calling model: {e}")
        return {"_error": str(e)}

# ---------------- Summarization background ----------------
def summarize_older_messages_once(chat_id: str, force: bool = False) -> str:
    messages_file = get_chat_file_path(chat_id, "messages.jsonl")
    if not os.path.exists(messages_file): return "No messages to summarize."
    
    with open(messages_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not force and len(lines) <= RECENT_MSGS: return "Not enough messages."
    
    selected = lines if force else lines[:-RECENT_MSGS]
    if not selected: return "No older messages."
    
    text_to_summarize = []
    for ln in selected:
        try:
            rec = json.loads(ln)
            text_to_summarize.append(f"[{rec.get('role','?')} @ {rec.get('ts','?')}]: {rec.get('content','')}")
        except:
            continue
    text_to_summarize = "\n".join(text_to_summarize)
    if not text_to_summarize.strip(): return "No content."
        
    persona = load_persona(chat_id)
    system_prompt = (
        "You are a roleplay memory summarizer. Your task is to produce a structured, concise summary of the provided conversation log. "
        "Use the following format, with each section on a new line. Be factual and objective.\n"
        "**Summary:**\n[A brief, one-paragraph overview of the key events.]\n\n"
        "**Key Facts/Relationships:**\n- [Fact 1]\n- [Fact 2]\n- [Relationship development]\n\n"
        "**Character Dynamics:**\n- User: [Describe the user's apparent role, personality, or key actions.]\n- Assistant: [Describe your (the assistant's) emotional state, key decisions, or personality traits displayed.]"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Summarize:\n" + text_to_summarize}]
    resp = call_chat_model_raw(messages, stream=False)
    
    if isinstance(resp, dict) and "_error" in resp:
        return f"Summarization error: {resp.get('_error')}"
        
    summary_text = _parse_chat_response_json(resp)
    censor = persona.get("censor_list", []) or []
    for w in censor: 
        if w: summary_text = summary_text.replace(w, "xxx")
        
    current_date = datetime.now().strftime("%Y-%m-%d")
    append_summary(chat_id, summary_text, current_date, "")
    return summary_text

# ---------------- Prompt builder ----------------
def build_prompt_context(chat_id: str, user_msg: str, settings: Dict[str, Any]) -> List[Dict[str,str]]:
    persona = load_persona(chat_id)
    summary = load_summary(chat_id)
    system_parts = []
    
    thought_ratio = float(settings.get("thought_ratio", 0.5))
    talkativeness = float(settings.get("talkativeness", 0.5))
    persistent_stats = settings.get("persistent_stats", False)

    recent_messages = read_last_messages(chat_id, 5)
    greeting_delivered = any("New day greeting acknowledged" in msg.get("content", "") for msg in recent_messages if msg.get("role") == "system")
    
    is_new_day_session = "INFO: Starting new day" in summary[-250:] or "INFO: Starting new session" in summary[-250:]

    if is_new_day_session and not greeting_delivered:
        days_passed_match = re.search(r'(\d+) days have passed', summary)
        days_passed = int(days_passed_match.group(1)) if days_passed_match else 0
        
        greeting_instruction = (
            f"A new day has begun ({days_passed} days passed). Start with a morning greeting."
        ) if days_passed > 0 else (
            "This is a new session on the same day. Start with a greeting as if returning after a short break."
        )
        
        system_parts.append(
            f"**CRITICAL CONTEXT: A NEW DAY/SESSION HAS BEGUN!**\n"
            f"- **Instructions:** {greeting_instruction}\n"
            "- After the greeting, respond to the user's message as usual.\n"
        )
    
    system_parts.append(f"You are: {persona.get('name')}. {persona.get('short_description')}")
    if persona.get("traits"): system_parts.append("Traits: " + ", ".join(persona.get("traits", [])))
    if persona.get("history"): system_parts.append("History: " + persona.get("history"))
    if persona.get("behavior_instructions"): system_parts.append("Behavior rules: " + persona.get("behavior_instructions"))

    current_time = datetime.utcnow().isoformat() + "Z"
    system_parts.append(f"Current time: {current_time}. Use this to track time and day/night in your responses.")

    if persistent_stats:
        current_stats = load_emotional_state(chat_id)
        if current_stats:
            stats_str = "\n".join([f"**{k}**: {v}" for k, v in current_stats.items()])
            system_parts.append(f"Your current emotional state is:\n{stats_str}\nBased on the user's message, you must UPDATE this state in your response's [[Stats]] block. Do not invent a new state from scratch; evolve the existing one.")
        else:
            system_parts.append("This is your first interaction. Establish your initial emotional state in the [[Stats]] block of your response.")

    if len(read_all_messages(chat_id)) == 0:
        system_parts.append("Brevity: This is the first user message ever. Reply concisely (1-3 sentences).")
    
    ex = persona.get("prompt_examples", [])
    if ex:
        ex_text = "Examples:\n"
        for e in ex: ex_text += f"USER: {e.get('user', '')}\nASSISTANT: {e.get('assistant', '')}\n"
        system_parts.append(ex_text)
        
    if summary: system_parts.append("Memory summary:\n" + (summary if len(summary) < 2000 else summary[-2000:]))
    
    # World Events Logic
    active_events = []
    events_file = get_chat_file_path(chat_id, "world_events.jsonl")
    if os.path.exists(events_file):
        now = datetime.utcnow()
        msg_count = len(read_all_messages(chat_id))
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    is_expired = False
                    start_ts = datetime.fromisoformat(event['start_ts'].replace("Z", ""))
                    
                    if event['persistence_type'] == 'messages':
                        start_msg_count = event.get('start_msg_count', 0)
                        duration_msgs = event.get('value', 1)
                        # Event is active from its start count up to (but not including) the end count.
                        if msg_count >= start_msg_count + duration_msgs:
                            is_expired = True
                    elif event['persistence_type'] == 'time':
                        if now >= start_ts + timedelta(minutes=event.get('value', 5)):
                            is_expired = True
                    
                    if not is_expired:
                        active_events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue
    
    if active_events:
        event_texts = "\n- ".join([e['text'] for e in active_events])
        system_parts.append(
            f"**MANDATORY WORLD EVENT!**\n"
            f"A sudden event has just occurred that you were not aware of until this moment. You MUST react to it with surprise and confusion. "
            f"Your response—either in your thoughts, main reply, or final thoughts—MUST acknowledge this event. Do not ignore it.\n"
            f"**Event(s):**\n- {event_texts}"
        )


    output_instructions = persona.get("output_instructions", DEFAULT_PERSONA["output_instructions"])
    system_parts.append(output_instructions)

    if thought_ratio < 0.2: system_parts.append("Thought Ratio: LOW. Your <think> block MUST be very brief and concise.")
    elif thought_ratio > 0.8: system_parts.append("Thought Ratio: HIGH. Your <think> block MUST be extremely detailed, exploring multiple angles.")
    
    if talkativeness < 0.2: system_parts.append("Talkativeness: VERY LOW. Your response (excluding thoughts/stats) MUST be extremely concise (1-2 sentences maximum).")
    elif talkativeness > 0.8: system_parts.append("Talkativeness: VERY HIGH. Your response (excluding thoughts/stats) MUST be very long, talkative, and descriptive. Elaborate extensively.")

    system_text = "\n\n".join(system_parts)
    
    messages = [{"role": "system", "content": system_text}]
    relevant = semantic_search(chat_id, user_msg, TOP_K_MEMORIES)
    if relevant:
        mem_texts = [f"[{m.get('role', '?')} @ {m.get('ts', '?')}] {m.get('content', '')}" for m in relevant]
        messages.append({"role": "system", "content": "Relevant memories:\n" + "\n".join(mem_texts)})
        
    recent_history = read_last_messages(chat_id, RECENT_MSGS)
    for r in recent_history:
        if r.get("role") in ("user", "assistant"):
            content = r.get("content", "")
            if r.get("role") == "assistant":
                full_content = []
                if r.get("thoughts"): full_content.append(f"<think>{r['thoughts']}</think>")
                full_content.append(content)
                if r.get("stats"): full_content.append(r['stats'])
                if r.get("final_thoughts"): full_content.append(r['final_thoughts'])
                content = "\n".join(full_content)
            messages.append({"role": r["role"], "content": content})

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
.message.system { align-self: center; max-width: 90%; opacity: 0.6; }
.message-body { display: flex; flex-direction: column; width: 100%; }
.message-content {
    font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
    position: relative; overflow-wrap: anywhere; padding: 4px 0;
}
.message.user .message-body { background: var(--user-bubble); padding: 10px 15px; border-radius: 18px; text-align: left; }
.message.system .message-body {
  background: transparent; border: 1px solid var(--border); text-align: center;
  font-size: 0.875rem; color: var(--text-secondary); padding: 8px 16px; border-radius: 18px;
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
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.75rem; color: var(--text-secondary); margin-top: 8px;
  opacity: 0.4; transition: opacity 0.2s;
}
.message:hover .message-footer { opacity: 1; }
.message-actions { display: flex; gap: 8px; }
.message-actions button {
  background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 5px; color: var(--text-secondary); cursor: pointer;
  font-size: 0.8rem; padding: 2px 6px;
}
.message-actions button:hover { color: var(--text-primary); background: var(--bg-secondary); }
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
.action-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.btn {
  background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary);
  padding: 8px 12px; border-radius: 8px; font-size: 0.875rem; cursor: pointer; transition: all 0.2s;
}
.btn:hover { background: var(--accent); border-color: var(--accent); }
.btn.danger:hover { background: var(--danger); border-color: var(--danger); }
.collapsible-content { display: none; padding-top: 10px; border-top: 1px solid var(--border); margin-top: 10px; }
.collapsible-content.show { display: flex; flex-direction: column; gap: 10px; }
.slider-container, .toggle-container { display: flex; flex-direction: column; gap: 10px; font-size: 0.875rem; color: var(--text-secondary); }
.toggle-container label { display: flex; justify-content: space-between; align-items: center; }
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
.modal-body { overflow-y: auto; }
.modal-content h2 { font-size: 1.2rem; }
.modal-close-btn { align-self: flex-end; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer; }
.form-group { display: flex; flex-direction: column; gap: 8px; }
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
  .side-panel { position: fixed; top: 0; height: 100%; z-index: 500; }
  .side-panel.left-panel { left: 0; transform: translateX(-100%); }
  .side-panel.left-panel.collapsed { transform: translateX(-100%); } /* This is redundant but safe */
  .container.left-panel-open .side-panel.left-panel { transform: translateX(0); }
  .side-panel.right-panel { right: 0; transform: translateX(100%); }
  .side-panel.right-panel.collapsed { transform: translateX(100%); } /* This is redundant but safe */
  .container.right-panel-open .side-panel.right-panel { transform: translateX(0); }
  .panel-toggle-handle { display: none; } /* Hide handles, use header buttons on mobile */
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
            <button class="btn" id="force-summarize">Summarize</button>
            <button class="btn danger" id="clear-memory">Clear Memory</button>
          </div>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">Adv Settings</div>
      <div class="collapsible-content">
          <div class="toggle-container">
            <label><span>Persistent Stats</span><input type="checkbox" id="persistent-stats-toggle"></label>
            <small style="color: var(--text-secondary); font-size: 0.75rem;">(Saves the bot's emotional state between messages)</small>
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
            <div>Embeddings: <span id="embed-status">disabled</span></div>
            <div style="margin-top: 8px;"><button class="btn" id="test-embed" style="width: 100%;">Test Embeddings</button></div>
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
<script>
'use strict';
let ws = null;
let isGenerating = false;
let currentMessageContainer = null;
let fullResponseText = '';
let activeChatId = 'default_chat';
const SETTINGS_KEY = 'aiRoleplaySettings';


const DOM = {
    // Main elements
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    stopBtn: document.getElementById('stop-btn'),
    statusText: document.getElementById('status-text'),
    statusDot: document.getElementById('status-dot'),
    chatTitle: document.getElementById('chat-title'),
    // Panels
    leftPanel: document.getElementById('left-panel'),
    rightPanel: document.getElementById('right-panel'),
    leftPanelToggle: document.getElementById('left-panel-toggle'),
    rightPanelToggle: document.getElementById('right-panel-toggle'),
    mobileMenuLeft: document.getElementById('mobile-menu-left'),
    mobileMenuRight: document.getElementById('mobile-menu-right'),
    appContainer: document.getElementById('app-container'),
    // Chat list
    chatList: document.getElementById('chat-list'),
    newChatName: document.getElementById('new-chat-name'),
    addChatBtn: document.getElementById('add-chat-btn'),
    // Action buttons
    reloadChatBtn: document.getElementById('reload-chat'),
    newDayBtn: document.getElementById('new-day'),
    summarizeBtn: document.getElementById('force-summarize'),
    clearMemoryBtn: document.getElementById('clear-memory'),
    // Settings
    persistentStatsToggle: document.getElementById('persistent-stats-toggle'),
    tempSlider: document.getElementById('temperature-slider'),
    tempValue: document.getElementById('temp-value'),
    tokensSlider: document.getElementById('tokens-slider'),
    tokensValue: document.getElementById('tokens-value'),
    thoughtSlider: document.getElementById('thought-ratio-slider'),
    thoughtValue: document.getElementById('thought-ratio-value'),
    talkSlider: document.getElementById('talkativeness-slider'),
    talkValue: document.getElementById('talkativeness-value'),
    // Persona
    personaAvatar: document.getElementById('persona-avatar'),
    sidePanelPersonaPreset: document.getElementById('side-panel-persona-preset'),
    sidePanelLoadBtn: document.getElementById('side-panel-load-btn'),
    openPersonaModalBtn: document.getElementById('open-persona-modal'),
    // World Events
    worldEventInput: document.getElementById('world-event-input'),
    eventTypeSelect: document.getElementById('event-type-select'),
    eventValueInput: document.getElementById('event-value-input'),
    injectEventBtn: document.getElementById('inject-event-btn'),
    // System Info
    embedStatus: document.getElementById('embed-status'),
    testEmbedBtn: document.getElementById('test-embed'),
    openSysInfoModalBtn: document.getElementById('open-sys-info-modal'),
    // Modals
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
};

function connectWebSocket() {
    return new Promise((resolve, reject) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'init', chat_id: activeChatId }));
            resolve();
            return;
        }
        if (ws) { ws.close(); }

        const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            updateStatus('connected');
            DOM.sendBtn.disabled = false;
            ws.send(JSON.stringify({ type: 'init', chat_id: activeChatId }));
            resolve();
        };

        ws.onclose = () => {
            updateStatus('disconnected');
            DOM.sendBtn.disabled = true;
            isGenerating = false;
            setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = (error) => {
            updateStatus('disconnected');
            isGenerating = false;
            reject(error);
        };

        ws.onmessage = (event) => {
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
        case 'start':
            isGenerating = true;
            updateStatus('generating');
            fullResponseText = '';
            // If this is a regeneration, the old message element is already gone.
            // A new one is created here.
            if(data.old_ts) {
                const oldMsg = document.querySelector(`.message[data-ts="${data.old_ts}"]`);
                if (oldMsg) oldMsg.remove();
            }
            currentMessageContainer = addMessage('assistant', '', '', '', '', '', '');
            const body = currentMessageContainer.querySelector('.message-body');
            const indicator = document.createElement('div');
            indicator.className = 'responding-indicator';
            indicator.innerHTML = 'responding<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
            body.appendChild(indicator);
            break;
        case 'partial':
            handlePartialMessage(data.chunk);
            break;
        case 'done':
        case 'stopped':
            isGenerating = false;
            updateStatus('connected');
            // Final render is handled implicitly by the last partial message
            currentMessageContainer = null;
            break;
        case 'error':
            isGenerating = false;
            updateStatus('connected');
            addMessage('system', `[ERROR] ${data.message}`);
            break;
        case 'user_ts':
        case 'assistant_ts':
            const role = data.type.split('_')[0];
            const ts = data.ts;
            const msgElement = role === 'user' ? getLastMessageElement(role) : currentMessageContainer;
            if (msgElement) {
                msgElement.dataset.ts = ts;
                addMessageFooter(msgElement, role);
            }
            break;
        case 'chat_switched':
            console.log(`Switched to chat: ${data.chat_id}`);
            break;
    }
}

function simpleMarkdown(text) {
    if (typeof text !== 'string') return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function handlePartialMessage(chunk) {
    if (!currentMessageContainer) return;
    
    const indicator = currentMessageContainer.querySelector('.responding-indicator');
    if (indicator) indicator.remove();

    fullResponseText += chunk;
    
    let thoughts = '', stats = '', finalThoughts = '', cleanContent = '';
    let tempText = fullResponseText;

    const thinkMatch = tempText.match(/<think>([\s\S]*?)<\/think>/is);
    if (thinkMatch) {
        thoughts = thinkMatch[1];
        tempText = tempText.replace(thinkMatch[0], ''); // Remove the completed thought block for further processing
    } else {
        const thinkStartMatch = tempText.match(/<think>([\s\S]*)/is);
        if (thinkStartMatch) {
            thoughts = thinkStartMatch[1];
            tempText = ''; // Nothing is main content until the think tag is closed
        }
    }
    
    const finalThoughtsMatch = tempText.match(/(\*\*\[\[Final Thoughts\]\]\*\*[\s\S]*)/i);
    if (finalThoughtsMatch) {
        finalThoughts = finalThoughtsMatch[0];
        tempText = tempText.substring(0, finalThoughtsMatch.index);
    }
    
    const statsMatch = tempText.match(/(\*\*\[\[Stats\]\]\*\*[\s\S]*)/i);
    if (statsMatch) {
        stats = statsMatch[0];
        cleanContent = tempText.substring(0, statsMatch.index).trim();
    } else {
        cleanContent = tempText.trim();
    }
    
    updateOrCreateElement(currentMessageContainer, '.thought-container', thoughts, 'prepend');
    updateOrCreateElement(currentMessageContainer, '.message-content', simpleMarkdown(cleanContent));
    updateOrCreateElement(currentMessageContainer, '.stats-container', simpleMarkdown(stats), 'append');
    updateOrCreateElement(currentMessageContainer, '.final-thoughts-container', simpleMarkdown(finalThoughts), 'append');
    scrollToBottom();
}

function updateOrCreateElement(parent, selector, content, position = 'append') {
    let element = parent.querySelector(selector);
    const body = parent.querySelector('.message-body');
    if (!body) return;

    if (!content || content.trim() === '') {
        if (element) element.style.display = 'none';
        return;
    }

    if (!element) {
        element = document.createElement('div');
        element.className = selector.substring(1); // remove dot
        if (position === 'prepend') body.prepend(element);
        else body.appendChild(element);
    }
    
    element.style.display = 'block';
    
    if (selector === '.thought-container') {
        let thoughtContentEl = element.querySelector('.thought-content');
        if (!thoughtContentEl) {
            thoughtContentEl = document.createElement('div');
            thoughtContentEl.className = 'thought-content';
            element.appendChild(thoughtContentEl);
        }
        thoughtContentEl.innerHTML = simpleMarkdown(content);
        if (!element.hasToggleListener) {
            element.addEventListener('click', () => element.classList.toggle('expanded'));
            element.hasToggleListener = true;
        }
    } else {
        element.innerHTML = content;
    }
}

function updateStatus(status) {
  DOM.statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  DOM.statusDot.className = 'status-dot ' + status;
  toggleSendStopButtons(status === 'generating');
}

function toggleSendStopButtons(showStop) {
  DOM.sendBtn.style.display = showStop ? 'none' : 'flex';
  DOM.stopBtn.style.display = showStop ? 'flex' : 'none';
}

function addMessage(role, content = '', ts = '', thoughts = '', stats = '', final_thoughts = '') {
    const msgWrapper = document.createElement('div');
    msgWrapper.className = `message ${role}`;
    if (ts) msgWrapper.dataset.ts = ts;
  
    const msgBodyContainer = document.createElement('div');
    msgBodyContainer.className = 'message-body-container';
    
    const msgBody = document.createElement('div');
    msgBody.className = 'message-body';
    msgBodyContainer.appendChild(msgBody);
    msgWrapper.appendChild(msgBodyContainer);

    renderMessage(msgWrapper, { content, thoughts, stats, final_thoughts });
  
    if (role !== 'system' && ts) {
        addMessageFooter(msgWrapper, role);
    }
  
    DOM.chatMessages.appendChild(msgWrapper);
    scrollToBottom();
    return msgWrapper;
}

function renderMessage(msgWrapper, msgData) {
    updateOrCreateElement(msgWrapper, '.thought-container', msgData.thoughts, 'prepend');
    updateOrCreateElement(msgWrapper, '.message-content', simpleMarkdown(msgData.content));
    updateOrCreateElement(msgWrapper, '.stats-container', simpleMarkdown(msgData.stats), 'append');
    updateOrCreateElement(msgWrapper, '.final-thoughts-container', simpleMarkdown(msgData.final_thoughts), 'append');
}

function addMessageFooter(msgWrapper, role) {
    let footer = msgWrapper.querySelector('.message-footer');
    if (footer) footer.remove(); // Clear existing footer if any

    footer = document.createElement('div');
    footer.className = 'message-footer';
    
    const timestamp = document.createElement('span');
    timestamp.textContent = new Date(msgWrapper.dataset.ts).toLocaleTimeString();

    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'message-actions';
    actionsContainer.innerHTML = `
        <button title="Edit">📝 Edit</button>
        ${role === 'assistant' ? '<button title="Regenerate">🔄 Regenerate</button>' : ''}
        <button title="Delete">🗑️ Delete</button>
    `;

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
    await api(`/${activeChatId}/delete_message`, 'POST', {ts});
    element.remove();
  }
}

function editMessage(msgWrapper, role) {
    const msgBody = msgWrapper.querySelector('.message-body');
    if (!msgBody || msgBody.querySelector('textarea')) return;

    // Helper to extract text content from a container, converting <br> to newlines
    const getTextFromContainer = (selector) => {
        const el = msgWrapper.querySelector(selector);
        if (!el || el.style.display === 'none') return '';
        const tempDiv = document.createElement('div');
        // For thoughts, we need to get from the inner .thought-content div
        const contentSource = selector === '.thought-container' ? el.querySelector('.thought-content') : el;
        tempDiv.innerHTML = contentSource.innerHTML.replace(/<br\s*[\/]?>/gi, "\n");
        return tempDiv.textContent || tempDiv.innerText || "";
    };

    const thoughts = getTextFromContainer('.thought-container');
    const content = getTextFromContainer('.message-content');
    const stats = getTextFromContainer('.stats-container');
    const finalThoughts = getTextFromContainer('.final-thoughts-container');

    let fullRawText = '';
    if (thoughts) fullRawText += `<think>${thoughts}</think>\n`;
    fullRawText += content;
    if (stats) fullRawText += `\n${stats}`;
    if (finalThoughts) fullRawText += `\n${finalThoughts}`;
    
    const originalHTML = msgBody.innerHTML;

    const editor = document.createElement('textarea');
    editor.className = 'chat-input';
    editor.style.width = '100%';
    editor.value = fullRawText.trim();
    
    const btnContainer = document.createElement('div');
    btnContainer.style.marginTop = '10px';
    btnContainer.style.display = 'flex';
    btnContainer.style.gap = '8px';
    
    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn';
    saveBtn.textContent = 'Save';
    
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn danger';
    cancelBtn.textContent = 'Cancel';
    
    btnContainer.appendChild(saveBtn);
    btnContainer.appendChild(cancelBtn);
    
    msgBody.innerHTML = '';
    msgBody.appendChild(editor);
    msgBody.appendChild(btnContainer);

    editor.focus();
    editor.style.height = 'auto';
    editor.style.height = `${editor.scrollHeight}px`;

    cancelBtn.onclick = () => {
        msgBody.innerHTML = originalHTML;
    };

    saveBtn.onclick = async () => {
        const newRawContent = editor.value;
        const result = await api(`/${activeChatId}/edit_message`, 'POST', { ts: msgWrapper.dataset.ts, raw_content: newRawContent });
        
        // Re-render the message with the new, parsed content
        msgBody.innerHTML = ''; // Clear editor
        renderMessage(msgWrapper, result.updated_message);
    };
}

async function regenerateMessage(ts) {
    if (!ts || isGenerating) return;
    if (!confirm('Regenerate this response? The current one will be deleted.')) return;
    
    isGenerating = true;
    updateStatus('generating');
    ws.send(JSON.stringify({ type: 'regenerate', ts: ts, chat_id: activeChatId, settings: getSettings() }));
}

function getSettings() {
    return {
      temperature: DOM.tempSlider.value,
      max_tokens: DOM.tokensSlider.value,
      thought_ratio: DOM.thoughtSlider.value,
      talkativeness: DOM.talkSlider.value,
      persistent_stats: DOM.persistentStatsToggle.checked,
    };
}

function saveSettings() {
    const settings = getSettings();
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function loadSettings() {
    const savedSettings = localStorage.getItem(SETTINGS_KEY);
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            DOM.tempSlider.value = settings.temperature || 1.0;
            DOM.tokensSlider.value = settings.max_tokens || 1024;
            DOM.thoughtSlider.value = settings.thought_ratio || 0.5;
            DOM.talkSlider.value = settings.talkativeness || 0.5;
            DOM.persistentStatsToggle.checked = settings.persistent_stats || false;

            // Update display values
            DOM.tempValue.textContent = DOM.tempSlider.value;
            DOM.tokensValue.textContent = DOM.tokensSlider.value;
            DOM.thoughtValue.textContent = DOM.thoughtSlider.value;
            DOM.talkValue.textContent = DOM.talkSlider.value;

        } catch (e) {
            console.error("Failed to load settings from localStorage", e);
        }
    }
}


function getLastMessageElement(role) {
  const messages = document.querySelectorAll(`.message.${role}`);
  return messages[messages.length - 1];
}

function scrollToBottom() {
    setTimeout(() => {
        DOM.chatMessages.scrollTo({ top: DOM.chatMessages.scrollHeight, behavior: 'smooth' });
    }, 100);
}

async function sendMessage() {
  const message = DOM.chatInput.value.trim();
  if (!message || isGenerating) return;
  DOM.chatInput.value = '';
  DOM.chatInput.style.height = 'auto';
  
  addMessage('user', message, new Date().toISOString());
  
  try {
    await connectWebSocket();
    ws.send(JSON.stringify({ message, settings: getSettings(), chat_id: activeChatId }));
  } catch (error) {
    addMessage('system', '[ERROR] Connection failed. Please check the server.');
  }
}

function stopGeneration() {
  if (ws && ws.readyState === WebSocket.OPEN && isGenerating) {
    ws.send(JSON.stringify({ stop: true, chat_id: activeChatId }));
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
    if (!response.ok) {
        const errorText = await response.text();
        addMessage('system', `[API ERROR] ${response.status}: ${errorText}`);
        throw new Error(`API Error: ${response.status} ${errorText}`);
    }
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.indexOf("application/json") !== -1) {
        return await response.json();
    }
    return await response.text();
  } catch (error) {
      console.error("API call failed:", error);
      addMessage('system', `[API ERROR] Failed to fetch from ${path}. Server might be down.`);
      throw error;
  }
}

async function clearMemory() {
  if (!confirm('Clear all memory for this chat? This deletes conversation history, summaries, stats and events.')) return;
  await api(`/${activeChatId}/clear_memory`, 'POST');
  DOM.chatMessages.innerHTML = '';
  updateEmbedStatus();
}

async function newDay() {
  addMessage('system', 'Creating summary and starting new day...');
  const res = await api(`/${activeChatId}/new_day`, 'POST');
  if (res.marker) {
    addMessage('system', res.marker);
    addMessage('system', 'A new session has started. Send your next message to get a greeting from the character.');
  }
}

async function injectEvent() {
    const eventText = DOM.worldEventInput.value.trim();
    if (!eventText) { alert('Please describe the event.'); return; }
    
    await api(`/${activeChatId}/inject_event`, 'POST', {
        event: eventText,
        type: DOM.eventTypeSelect.value,
        value: parseInt(DOM.eventValueInput.value, 10)
    });
    DOM.worldEventInput.value = '';
    addMessage('system', `[WORLD EVENT INJECTED] ${eventText}`);
}

async function forceSummarize() {
    addMessage('system', 'Generating summary using LLM...');
    const response = await api(`/${activeChatId}/force_summarize`, 'POST');
    addMessage('system', `Summary Generated:\n${response.summary}`);
}

async function reloadChat() {
  DOM.chatMessages.innerHTML = '';
  const messages = await api(`/${activeChatId}/messages`);
  messages.forEach(msg => {
    addMessage(msg.role, msg.content, msg.ts, msg.thoughts, msg.stats, msg.final_thoughts);
  });
  const persona = await api(`/${activeChatId}/persona`);
  DOM.chatTitle.textContent = persona.name || "AI Roleplay";
}

async function updateEmbedStatus() {
  const status = await api('/embed_status');
  DOM.embedStatus.textContent = status.enabled ? 'enabled' : 'disabled';
}

async function testEmbeddings() {
  const res = await api('/test_embeddings', 'POST');
  updateEmbedStatus();
  alert(res.success ? 'Test successful' : 'Test failed: ' + res.error);
}

function setupPanelToggles() {
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
            if (panel.id === 'left-panel') {
                DOM.appContainer.classList.remove('left-panel-open');
                panel.classList.add('collapsed');
            } else if (panel.id === 'right-panel') {
                DOM.appContainer.classList.remove('right-panel-open');
                panel.classList.add('collapsed');
            }
        });
    });
}

function setupCollapsibleSections() {
    document.querySelectorAll('.panel-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            const content = toggle.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('show');
                toggle.classList.toggle('collapsed');
            }
        });
    });
}

function setupPersonaModal() {
    DOM.openPersonaModalBtn.addEventListener('click', async () => {
        const currentPersona = await api(`/${activeChatId}/persona`);
        DOM.personaEditor.value = JSON.stringify(currentPersona, null, 2);
        await loadSavedPersonasIntoSelect(DOM.savedPersonasList);
        DOM.personaModal.style.display = 'flex';
    });
    DOM.personaModalClose.addEventListener('click', () => DOM.personaModal.style.display = 'none');
    DOM.generatePersonaBtn.addEventListener('click', async () => {
        const prompt = DOM.personaPrompt.value;
        if (!prompt) { alert('Please enter a description.'); return; }
        DOM.generatePersonaBtn.textContent = 'Generating...';
        DOM.generatePersonaBtn.disabled = true;
        try {
            const result = await api('/generate_persona', 'POST', { description: prompt });
            DOM.personaEditor.value = JSON.stringify(result.persona, null, 2);
        } finally {
            DOM.generatePersonaBtn.textContent = 'Generate';
            DOM.generatePersonaBtn.disabled = false;
        }
    });

    DOM.savePersonaBtn.addEventListener('click', async () => {
        const name = DOM.savePersonaName.value.trim();
        if (!name) { alert('Please enter a name to save the persona.'); return; }
        try {
            const persona = JSON.parse(DOM.personaEditor.value);
            await api(`/personas/${name}`, 'POST', persona);
            await api(`/${activeChatId}/persona`, 'POST', persona); // Also set as active
            alert(`Persona '${name}' saved and set as active for the current chat.`);
            DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
            await Promise.all([
                loadSavedPersonasIntoSelect(DOM.savedPersonasList),
                loadSavedPersonasIntoSelect(DOM.sidePanelPersonaPreset)
            ]);
        } catch (e) {
            alert('Invalid JSON or failed to save: ' + e.message);
        }
    });

    DOM.loadPersonaBtn.addEventListener('click', () => loadAndActivatePersona(DOM.savedPersonasList.value));
    DOM.sidePanelLoadBtn.addEventListener('click', () => loadAndActivatePersona(DOM.sidePanelPersonaPreset.value));
}

async function loadAndActivatePersona(name) {
    if (!name) { alert('Please select a persona to load.'); return; }
    try {
        const persona = await api(`/personas/${name}`);
        DOM.personaEditor.value = JSON.stringify(persona, null, 2);
        await api(`/${activeChatId}/persona`, 'POST', persona);
        DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
        DOM.chatTitle.textContent = persona.name || "AI Roleplay";
        alert(`Persona '${name}' loaded and set as active for this chat.`);
    } catch(e) {
        alert(`Failed to load persona '${name}'. It might not exist or there was a server error.`);
    }
}

async function loadSavedPersonasIntoSelect(selectElement) {
    const personas = await api('/personas');
    const currentVal = selectElement.value;
    selectElement.innerHTML = '<option value="">-- Select --</option>';
    personas.forEach(p => {
        const option = document.createElement('option');
        option.value = p;
        option.textContent = p;
        selectElement.appendChild(option);
    });
    if (personas.includes(currentVal)) {
        selectElement.value = currentVal;
    }
}

function setupSystemInfoModal() {
    DOM.openSysInfoModalBtn.addEventListener('click', async () => {
        const info = await api('/system_info');
        document.getElementById('sys-info-version').textContent = info.version;
        document.getElementById('sys-info-model').textContent = info.model_name;
        
        const changelogContent = document.getElementById('changelog-content');
        changelogContent.innerHTML = `
v0.37: Robust parsing, message editing fix, persistent settings, chat deletion, stronger world events.
v0.36: Bug fixes & UI/UX improvements.
v0.35a: Multi-chat, UI overhaul, message editing.
v0.30: Enhanced world events system.
v0.25: Added persona generation & management.
v0.20: Implemented semantic memory search.
v0.10: Initial streaming chat implementation.`;
        
        DOM.sysInfoModal.style.display = 'flex';
    });
    document.getElementById('changelog-toggle').addEventListener('click', (e) => {
        e.target.classList.toggle('collapsed');
        document.getElementById('changelog-content').classList.toggle('show');
    });
    DOM.sysInfoModalClose.addEventListener('click', () => DOM.sysInfoModal.style.display = 'none');
}

function setupChatManagement() {
    DOM.addChatBtn.addEventListener('click', async () => {
        const name = DOM.newChatName.value.trim();
        if (!name) { alert('Please enter a name for the new chat.'); return; }
        const res = await api('/chats/create', 'POST', { name });
        DOM.newChatName.value = '';
        await loadChatList();
        switchChat(res.chat_id);
    });
}

async function loadChatList() {
    const chats = await api('/chats');
    DOM.chatList.innerHTML = '';
    if (chats.length === 0) {
        await api('/chats/create', 'POST', { name: 'default_chat' });
        // Recurse to load the list again now that default exists
        return loadChatList();
    }
    chats.forEach(chatId => {
        const li = document.createElement('li');
        li.className = 'chat-list-item';
        li.dataset.chatId = chatId;
        
        const nameSpan = document.createElement('span');
        nameSpan.textContent = chatId;
        nameSpan.style.flexGrow = '1';
        nameSpan.style.overflow = 'hidden';
        nameSpan.style.textOverflow = 'ellipsis';
        
        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-chat-btn';
        deleteBtn.innerHTML = '&times;';
        deleteBtn.title = `Delete chat "${chatId}"`;

        li.appendChild(nameSpan);
        li.appendChild(deleteBtn);

        if (chatId === activeChatId) {
            li.classList.add('active');
        }

        li.addEventListener('click', (e) => {
            if (e.target !== deleteBtn) {
                switchChat(chatId);
            }
        });
        deleteBtn.addEventListener('click', () => deleteChat(chatId));

        DOM.chatList.appendChild(li);
    });
}

async function deleteChat(chatId) {
    if (isGenerating) {
        alert("Cannot delete a chat while a response is being generated.");
        return;
    }
    if (!confirm(`Are you sure you want to permanently delete the chat "${chatId}"? This cannot be undone.`)) return;

    await api(`/chats/${chatId}`, 'DELETE');
    
    // If we deleted the active chat, switch to another one
    if (activeChatId === chatId) {
        const chats = await api('/chats');
        const newActiveChat = chats.length > 0 ? chats[0] : null;
        if (newActiveChat) {
            await switchChat(newActiveChat);
        } else {
            // No chats left, create a default one
            await api('/chats/create', 'POST', { name: 'default_chat' });
            await switchChat('default_chat');
        }
    }
    await loadChatList();
}


async function switchChat(chatId) {
    if (activeChatId === chatId && ws && ws.readyState === WebSocket.OPEN) return;
    activeChatId = chatId;
    console.log(`Switching to chat: ${activeChatId}`);
    
    document.querySelectorAll('.chat-list-item.active').forEach(el => el.classList.remove('active'));
    const newActiveEl = document.querySelector(`.chat-list-item[data-chat-id="${chatId}"]`);
    if(newActiveEl) newActiveEl.classList.add('active');
    
    isGenerating = false;
    stopGeneration();
    await reloadChat();
    await connectWebSocket(); // Re-init connection for new chat
    const persona = await api(`/${activeChatId}/persona`);
    DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
    await loadSavedPersonasIntoSelect(DOM.sidePanelPersonaPreset);
}

function setupEventListeners() {
    DOM.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    DOM.chatInput.addEventListener('input', () => {
      DOM.chatInput.style.height = 'auto';
      DOM.chatInput.style.height = `${Math.min(DOM.chatInput.scrollHeight, 120)}px`;
    });
    
    DOM.sendBtn.addEventListener('click', sendMessage);
    DOM.stopBtn.addEventListener('click', stopGeneration);
    DOM.reloadChatBtn.addEventListener('click', reloadChat);
    DOM.newDayBtn.addEventListener('click', newDay);
    DOM.summarizeBtn.addEventListener('click', forceSummarize);
    DOM.clearMemoryBtn.addEventListener('click', clearMemory);
    DOM.testEmbedBtn.addEventListener('click', testEmbeddings);
    DOM.injectEventBtn.addEventListener('click', injectEvent);

    // Settings listeners
    const settingsControls = [
        DOM.tempSlider, DOM.tokensSlider, DOM.thoughtSlider, 
        DOM.talkSlider, DOM.persistentStatsToggle
    ];
    const sliders = {
        [DOM.tempSlider.id]: DOM.tempValue, [DOM.tokensSlider.id]: DOM.tokensValue,
        [DOM.thoughtSlider.id]: DOM.thoughtValue, [DOM.talkSlider.id]: DOM.talkValue
    };
    
    settingsControls.forEach(control => {
        control.addEventListener('input', e => {
            if (e.target.type === 'range') {
                sliders[e.target.id].textContent = e.target.value;
            }
            saveSettings();
        });
    });
}

async function initializeApp() {
    setupEventListeners();
    setupPanelToggles();
    setupCollapsibleSections();
    setupPersonaModal();
    setupSystemInfoModal();
    setupChatManagement();
    
    loadSettings();
    await loadChatList();
    if (!document.querySelector('.chat-list-item.active')) {
        const firstChat = document.querySelector('.chat-list-item');
        if (firstChat) {
            activeChatId = firstChat.dataset.chatId;
            firstChat.classList.add('active');
        } else {
            // This case handles the very first run on a clean install
            await api('/chats/create', 'POST', { name: 'default_chat' });
            activeChatId = 'default_chat';
            await loadChatList();
        }
    }

    DOM.persistentStatsToggle.checked = false;
    await updateEmbedStatus();
    await reloadChat();
    await connectWebSocket();
    await loadSavedPersonasIntoSelect(DOM.sidePanelPersonaPreset);
    const persona = await api(`/${activeChatId}/persona`);
    DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
}

document.addEventListener('DOMContentLoaded', initializeApp);
</script>

<!-- CLIENT-SIDE HOTFIXES: automatically applied to improve UI behavior without changing style -->
<script>
document.addEventListener('DOMContentLoaded', function() {
  // Remove duplicate responding indicators whenever a new generation starts
  const removeDuplicateResponding = () => {
    document.querySelectorAll('.responding-indicator').forEach((n, idx) => {
      if (idx > 0) n.remove();
    });
  };

  // MutationObserver to ensure assistant messages always get a footer and actions (left-aligned)
  const chatMessages = document.getElementById('chat-messages') || document.getElementById('messages');
  if (chatMessages) {
    const ensureFooter = (msg) => {
      if (!msg) return;
      // align assistant footers to left
      msg.querySelectorAll('.message-footer').forEach(f => f.style.justifyContent = 'flex-start');
      // create footer if missing
      if (!msg.querySelector('.message-footer')) {
        const footer = document.createElement('div');
        footer.className = 'message-footer';
        const tsSpan = document.createElement('span'); tsSpan.className='timestamp'; tsSpan.textContent='';
        const actions = document.createElement('div'); actions.className='message-actions';
        actions.innerHTML = '<button title=\"Edit\">📝 Edit</button>' + (msg.classList.contains('assistant') ? '<button title=\"Regenerate\">🔄 Regenerate</button>' : '') + '<button title=\"Delete\">🗑️ Delete</button>';
        footer.appendChild(tsSpan); footer.appendChild(actions);
        msg.appendChild(footer);
        // wire simple handlers (they call existing functions if present)
        const editBtn = actions.querySelector('[title=\"Edit\"]');
        const regenBtn = actions.querySelector('[title=\"Regenerate\"]');
        const delBtn = actions.querySelector('[title=\"Delete\"]');
        if (editBtn) editBtn.onclick = () => { try { editMessage(msg); } catch(e){ /* fallback */ } };
        if (regenBtn) regenBtn.onclick = () => { try { regenerateMessage(msg.dataset.ts); } catch(e){ /* fallback */ } };
        if (delBtn) delBtn.onclick = () => { try { deleteMessage(msg.dataset.ts, msg); } catch(e){ msg.remove(); } };
      } else {
        // ensure buttons exist
        const actions = msg.querySelector('.message-actions');
        if (actions) {
          actions.style.marginLeft = '8px';
        }
        msg.querySelector('.message-footer').style.justifyContent = 'flex-start';
      }
    };

    const mo = new MutationObserver((mutations) => {
      removeDuplicateResponding();
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType === 1 && node.classList.contains('message') && node.classList.contains('assistant')) {
            ensureFooter(node);
          }
          // If a responding indicator was added, remove extra copies
          if (node.nodeType === 1 && node.querySelector && node.querySelector('.responding-indicator')) {
            removeDuplicateResponding();
          }
        }
      }
    });
    mo.observe(chatMessages, { childList: true, subtree: true });
  }

  // Make edit textarea resizable and larger when editing a message (monkey-patch editMessage if present)
  window._originalEditMessage = window.editMessage;
  window.editMessage = function(target) {
    try {
      const msg = (target && target.closest) ? target.closest('.message') : (typeof target === 'string' ? document.querySelector(`.message[data-ts="${target}"]`) : target);
      if (!msg) return;
      // attempt to reuse original if it exists and works
      if (typeof window._originalEditMessage === 'function' && target && !target.__patched) {
        // call original if it expects a button element; otherwise fallthrough
        try { window._originalEditMessage(target); return; } catch(e){ /* fallback to custom editor */ }
      }
      const contentEl = msg.querySelector('.message-body') || msg;
      const getText = (sel) => {
        const el = msg.querySelector(sel);
        if (!el || el.style.display === 'none') return '';
        const tmp = document.createElement('div'); tmp.innerHTML = el.innerHTML.replace(/<br\s*\/?>/gi, '\\n'); return tmp.textContent || tmp.innerText || '';
      };
      const thoughts = getText('.thought-container .thought-content');
      const content = getText('.message-content');
      const stats = getText('.stats-container');
      const final = getText('.final-thoughts-container');
      let raw = '';
      if (thoughts) raw += `<think>${thoughts}</think>\\n`;
      raw += content;
      if (stats) raw += `\\n${stats}`;
      if (final) raw += `\\n${final}`;

      // Build resizable editor
      contentEl.innerHTML = '';
      const ta = document.createElement('textarea');
      ta.value = raw.trim();
      ta.style.width = '100%';
      ta.style.height = '320px';
      ta.style.resize = 'both';
      ta.style.padding = '10px';
      ta.style.fontFamily = 'monospace';
      contentEl.appendChild(ta);
      const btns = document.createElement('div'); btns.style.marginTop='8px'; btns.style.display='flex'; btns.style.gap='8px';
      const save = document.createElement('button'); save.className='btn'; save.textContent='Save';
      const cancel = document.createElement('button'); cancel.className='btn danger'; cancel.textContent='Cancel';
      btns.appendChild(save); btns.appendChild(cancel);
      contentEl.appendChild(btns);
      cancel.onclick = ()=>{ location.reload(); };
      save.onclick = async ()=>{
        const newRaw = ta.value;
        try {
          const resp = await fetch(`/${(window.activeChatId||'default_chat')}/edit_message`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ ts: msg.dataset.ts, raw_content: newRaw }) });
          const j = await resp.json();
          if (j && j.updated_message && j.updated_message.rendered_html) {
            contentEl.innerHTML = j.updated_message.rendered_html;
          } else {
            contentEl.innerHTML = '<div class=\"message-content\">'+(newRaw.replace(/\\n/g,'<br>'))+'</div>';
          }
        } catch (e) {
          alert('Save failed: '+e);
        }
      };
    } catch (e) {
      console.error('editMessage hotfix failed', e);
      // fallback to original if present
      if (typeof window._originalEditMessage === 'function') try{ window._originalEditMessage(target); }catch(e2){}
    }
  };

  // Auto-scroll inner thought containers to bottom on content updates
  const observeThoughts = () => {
    const observer = new MutationObserver((muts)=>{
      muts.forEach(mu=>{
        const node = mu.target;
        try {
          node.scrollTop = node.scrollHeight;
        } catch(e){}
      });
    });
    document.querySelectorAll('.thought-container .thought-content').forEach(n => observer.observe(n, { characterData: true, subtree: true, childList: true }));
    // also ensure future ones get observed
    const msgObserver = new MutationObserver((mut)=>{
      mut.forEach(m=>{
        m.addedNodes.forEach(n=>{
          if (n.querySelectorAll) {
            n.querySelectorAll('.thought-container .thought-content').forEach(el=> observer.observe(el, { characterData: true, subtree: true, childList: true }));
          }
        });
      });
    });
    const messages = document.getElementById('chat-messages') || document.getElementById('messages');
    if (messages) msgObserver.observe(messages, { childList: true, subtree: true });
  };
  observeThoughts();

  // Clean up any leftover single 'responding...' nodes on load
  document.querySelectorAll('.responding-indicator').forEach((n, idx) => { if (idx>0) n.remove(); });
});
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
    final_html = HTML_UI.replace(
        'false',
        "true" if PERSISTENT_STATS_ENABLED else "false"
    )
    return HTMLResponse(content=final_html)

@app.get("/system_info")
async def system_info():
    return {
        "version": "0.37",
        "model_name": TEXT_MODEL_NAME
    }

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
    get_chat_dir(safe_name) # Creates directory
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
    return {"ok": True}

@app.post("/{chat_id}/new_day")
async def new_day_endpoint(chat_id: str):
    marker = mark_new_day(get_safe_chat_id(chat_id))
    return {"marker": marker}

@app.post("/{chat_id}/force_summarize")
async def force_summarize_endpoint(chat_id: str):
    summary_text = summarize_older_messages_once(get_safe_chat_id(chat_id), force=True)
    return {"summary": summary_text}

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
    ts = body.get("ts")
    raw_content = body.get("raw_content") # Changed from 'content'
    if ts is None or raw_content is None:
        return JSONResponse(status_code=400, content={"error": "ts and raw_content are required"})
    
    safe_chat_id = get_safe_chat_id(chat_id)
    if update_message_by_ts(safe_chat_id, ts, raw_content):
        # Find the updated message to send back for re-rendering
        messages = read_all_messages(safe_chat_id)
        updated_msg = next((msg for msg in messages if msg.get("ts") == ts), None)
        if updated_msg:
             return {"ok": True, "updated_message": updated_msg}
        return {"ok": True, "updated_message": {}} # Should not happen
    return JSONResponse(status_code=404, content={"error": "Message not found"})


@app.post("/{chat_id}/inject_event")
async def inject_event_endpoint(chat_id: str, req: Request):
    safe_chat_id = get_safe_chat_id(chat_id)
    body = await req.json()
    event_text = body.get("event", "").strip()
    persistence_type = body.get("type", "messages")
    try: value = int(body.get("value", 3))
    except (ValueError, TypeError): value = 3
    
    if not event_text: return JSONResponse(status_code=400, content={"error": "Event text cannot be empty."})
    
    event_entry = {
        "text": event_text, "persistence_type": persistence_type, "value": value,
        "start_ts": now_iso(), "start_msg_count": len(read_all_messages(safe_chat_id))
    }
    
    with open(get_chat_file_path(safe_chat_id, "world_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(event_entry) + "\n")

    append_message_to_disk(safe_chat_id, "system", f"[WORLD EVENT INJECTED] {event_text}")
    return {"status": "ok", "event_injected": event_text}

# Global endpoints
@app.get("/personas")
async def list_personas():
    if not os.path.exists(PERSONAS_DIR): return []
    return sorted([f.replace('.json', '') for f in os.listdir(PERSONAS_DIR) if f.endswith('.json')])

@app.get("/personas/{name}")
async def get_saved_persona(name: str):
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    file_path = os.path.join(PERSONAS_DIR, f"{safe_name}.json")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Persona not found"})
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/personas/{name}")
async def save_named_persona(name: str, req: Request):
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    if not safe_name: return JSONResponse(status_code=400, content={"error": "Invalid persona name"})
    body = await req.json()
    file_path = os.path.join(PERSONAS_DIR, f"{safe_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
    return {"status": "saved", "name": safe_name}

@app.post("/generate_persona")
async def generate_persona(req: Request):
    body = await req.json()
    description = body.get("description")
    if not description:
        return JSONResponse(status_code=400, content={"error": "Description is required"})

    generation_prompt = f"""
    Based on the user's simple description, create a detailed persona JSON object for a roleplaying AI.
    The user wants: "{description}".
    Expand on this concept. The JSON should include 'name', 'avatar' (a filename like 'ninja_avatar.png'), 'short_description', 'traits' (a list of strings), 'history', 'behavior_instructions' and 'output_instructions'.
    Make the persona rich and interesting. The output_instructions MUST be the detailed template including <think>, **[[Stats]]**, and **[[Final Thoughts]]**.
    Your output MUST be ONLY the raw JSON object, with no other text or markdown fences before or after it.
    """
    
    messages = [{"role": "system", "content": "You are a creative assistant that generates JSON objects for AI personas."},
                {"role": "user", "content": generation_prompt}]
    
    response = call_chat_model_raw(messages, stream=False)
    if isinstance(response, dict) and "_error" in response:
        return JSONResponse(status_code=500, content={"error": response["_error"]})

    content = _parse_chat_response_json(response).strip()
    
    try:
        # Find the JSON object in the response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No JSON object found in the LLM response.")
        
        persona_json = json.loads(json_match.group(0))
        # Ensure it has the critical key
        if "output_instructions" not in persona_json:
            persona_json["output_instructions"] = DEFAULT_PERSONA["output_instructions"]

        return {"persona": persona_json}
    except Exception as e:
        print(f"Failed to parse persona JSON from LLM. Response was:\n{content}")
        return JSONResponse(status_code=500, content={"error": f"Failed to parse LLM response as JSON: {e}"})


@app.post("/test_embeddings")
async def test_embeddings():
    global EMBEDDINGS_ENABLED
    test_vec = compute_embedding("test")
    success = test_vec is not None and test_vec.shape[0] > 0
    EMBEDDINGS_ENABLED = success
    return {"success": success, "enabled": EMBEDDINGS_ENABLED, "error": "" if success else "Failed to get a valid embedding vector."}


@app.get("/embed_status")
async def embed_status():
    return {"enabled": EMBEDDINGS_ENABLED}

# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    global stop_generation
    chat_id = DEFAULT_CHAT_ID
    
    async def process_and_stream_response(messages: List[Dict], settings: Dict, chat_id: str):
        stop_generation.clear()
        await ws.send_text(json.dumps({"type": "start"}))
        
        full_response = ""
        chunk_buffer = ""
        resp = call_chat_model_raw(messages, stream=True, settings=settings)

        if isinstance(resp, requests.Response):
            line_buffer = ""
            try:
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if stop_generation.is_set(): break
                    line_buffer += chunk
                    if '\n' not in line_buffer: continue

                    lines = line_buffer.split('\n')
                    line_buffer = lines.pop() # Keep the last, possibly incomplete line
                    
                    for line in lines:
                        if line.startswith("data: "): line = line[6:]
                        if "[DONE]" in line: continue
                        if not line.strip(): continue
                        
                        try:
                            part = json.loads(line)
                            chunk_text = extract_content_from_json_chunk(part)
                            if chunk_text:
                                full_response += chunk_text
                                chunk_buffer += chunk_text
                                if len(chunk_buffer) >= CHUNK_SIZE:
                                    await ws.send_text(json.dumps({"type": "partial", "chunk": chunk_buffer}))
                                    chunk_buffer = ""
                        except json.JSONDecodeError:
                            continue
            finally:
                resp.close()
            if chunk_buffer:
                await ws.send_text(json.dumps({"type": "partial", "chunk": chunk_buffer}))

        elif isinstance(resp, dict) and "_error" in resp:
            await ws.send_text(json.dumps({"type": "error", "message": f"{resp.get('_status')}: {resp.get('_error')}"}))

        if not stop_generation.is_set() and full_response:
            await ws.send_text(json.dumps({"type": "done"}))
            
            parsed_data = parse_full_response(full_response)
            
            if parsed_data["content"]:
                if settings.get("persistent_stats", False) and parsed_data["stats"]:
                    save_emotional_state(chat_id, parse_stats_from_text(parsed_data["stats"]))
                
                assistant_ts = append_message_to_disk(
                    chat_id, "assistant", parsed_data["content"], 
                    thoughts=parsed_data["thoughts"], 
                    stats=parsed_data["stats"], 
                    final_thoughts=parsed_data["final_thoughts"]
                )
                await ws.send_text(json.dumps({"type": "assistant_ts", "ts": assistant_ts}))
                    
                threading.Thread(target=lambda: compute_embedding(parsed_data["content"]) and append_embedding(chat_id, compute_embedding(parsed_data["content"]), {"ts": assistant_ts, "role": "assistant", "content": parsed_data["content"]})).start()
        
        if "CRITICAL CONTEXT: A NEW DAY/SESSION HAS BEGUN!" in messages[0]['content']:
            append_message_to_disk(chat_id, "system", "New day greeting acknowledged.")

    try:
        while True:
            data = await ws.receive_text()
            obj = json.loads(data)
            
            if "chat_id" in obj:
                safe_id = get_safe_chat_id(obj["chat_id"])
                if os.path.exists(get_chat_dir(safe_id)):
                    chat_id = safe_id

            if obj.get("type") == "init":
                await ws.send_text(json.dumps({"type": "chat_switched", "chat_id": chat_id}))
                continue
            
            if obj.get("stop"):
                stop_generation.set()
                await ws.send_text(json.dumps({"type": "stopped"}))
                continue
            
            settings = obj.get("settings", {})

            if obj.get("type") == "regenerate":
                ts_to_regen = obj.get("ts")
                if not ts_to_regen: continue

                all_chat_messages = read_all_messages(chat_id)
                user_msg_for_regen = None
                
                # Find the user message that prompted the response we are regenerating
                for i, msg in enumerate(all_chat_messages):
                    if msg.get("ts") == ts_to_regen and i > 0:
                        if all_chat_messages[i-1].get("role") == "user":
                           user_msg_for_regen = all_chat_messages[i-1]
                           break
                
                if not user_msg_for_regen: continue

                # Delete the old assistant message from disk
                delete_message_by_ts(chat_id, ts_to_regen)
                
                # Notify UI to remove the old message and prepare for a new one
                await ws.send_text(json.dumps({"type": "start", "old_ts": ts_to_regen}))

                user_msg_content = user_msg_for_regen.get("content", "")
                
                # Build context *without* the message we are regenerating
                messages_for_regen = build_prompt_context(chat_id, user_msg_content, settings)
                # Manually add the user prompt that triggered it
                messages_for_regen.append({"role": "user", "content": user_msg_content})
                
                await process_and_stream_response(messages_for_regen, settings, chat_id)
                
            else: # Standard new message
                user_msg_content = obj.get("message", "").strip()
                if not user_msg_content: continue

                user_ts = now_iso()
                append_message_to_disk(chat_id, "user", user_msg_content, ts=user_ts)
                await ws.send_text(json.dumps({"type": "user_ts", "ts": user_ts}))
                
                embedding_text = user_msg_content
                threading.Thread(target=lambda: compute_embedding(embedding_text) and append_embedding(chat_id, compute_embedding(embedding_text), {"ts": user_ts, "role": "user", "content": embedding_text})).start()

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
        except:
            pass
    finally:
        stop_generation.set()


if __name__ == "__main__":
    default_avatar_path = os.path.join(STATIC_DIR, "default_avatar.png")
    if not os.path.exists(default_avatar_path):
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (256, 256), color = (40, 40, 40))
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except IOError:
                font = ImageFont.load_default()
            d.text((100,100), "AI", fill=(200,200,200), font=font)
            img.save(default_avatar_path, 'PNG')
            print(f"Created default avatar at {default_avatar_path}")
        except ImportError:
            print("Pillow not installed, cannot create default avatar. Place a 'default_avatar.png' in the 'static' directory.")

    uvicorn.run("gem:app", host="0.0.0.0", port=7860, reload=True)



# ------------------- APPENDED FIXES START -------------------
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse
import os, json, time, re, asyncio
from datetime import datetime, timedelta

STATIC_DIR = globals().get("STATIC_DIR") or os.path.join(os.path.dirname(__file__), "static")
CHATS_DIR = globals().get("CHATS_DIR") or os.path.join(os.path.dirname(__file__), "chats")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)

def safe_chat_id_local(chat_id: str) -> str:
    if not chat_id: return "default_chat"
    return re.sub(r'[^a-zA-Z0-9_-]', '_', chat_id) or "default_chat"
def get_chat_dir_local(chat_id: str) -> str:
    cid = safe_chat_id_local(chat_id)
    path = os.path.join(CHATS_DIR, cid)
    os.makedirs(path, exist_ok=True)
    return path
def messages_file_local(chat_id: str) -> str:
    return os.path.join(get_chat_dir_local(chat_id), "messages.jsonl")
def events_file_local(chat_id: str) -> str:
    return os.path.join(get_chat_dir_local(chat_id), "world_events.jsonl")
def read_all_messages_local(chat_id: str):
    p = messages_file_local(chat_id)
    if not os.path.exists(p): return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    return out
def append_message_local(chat_id: str, role: str, content: str, thoughts: str = "", stats: str = "", final_thoughts: str = "") -> str:
    ts = datetime.utcnow().isoformat() + "Z"
    rec = {"ts": ts, "role": role, "content": content}
    if thoughts: rec["thoughts"] = thoughts
    if stats: rec["stats"] = stats
    if final_thoughts: rec["final_thoughts"] = final_thoughts
    with open(messages_file_local(chat_id), "a", encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
    return ts
def delete_message_local(chat_id: str, ts: str):
    p = messages_file_local(chat_id)
    if not os.path.exists(p): return False
    with open(p, "r", encoding='utf-8') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    new = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            if obj.get("ts") != ts:
                new.append(obj)
        except Exception:
            continue
    with open(p, "w", encoding='utf-8') as f:
        for obj in new:
            f.write(json.dumps(obj, ensure_ascii=False) + "\\n")
    return True
def read_active_events_local(chat_id: str):
    path = events_file_local(chat_id)
    if not os.path.exists(path):
        return []
    out = []
    now = datetime.utcnow()
    messages_len = len(read_all_messages_local(chat_id))
    with open(path, "r", encoding='utf-8') as f:
        for ln in f:
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            expired = False
            ptype = ev.get("persistence_type", "messages")
            val = int(ev.get("value", 1))
            if ptype == "messages":
                if messages_len >= ev.get("start_msg_count", 0) + val:
                    expired = True
            else:
                start = datetime.fromisoformat(ev.get("start_ts").replace("Z", ""))
                if now >= start + timedelta(minutes=val):
                    expired = True
            if not expired:
                out.append(ev)
    return out
def inject_event_local(chat_id: str, text: str, persistence_type: str, value: int):
    path = events_file_local(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = []
    if os.path.exists(path):
        with open(path, "r", encoding='utf-8') as f:
            for ln in f:
                try: existing.append(json.loads(ln))
                except: continue
    next_id = (existing[-1].get("inject_id", 0) + 1) if existing else 1
    ev = {
        "inject_id": next_id,
        "start_ts": datetime.utcnow().isoformat() + "Z",
        "text": text,
        "persistence_type": persistence_type,
        "value": int(value),
        "start_msg_count": len(read_all_messages_local(chat_id))
    }
    with open(path, "a", encoding='utf-8') as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\\n")
    return ev
def postprocess_parsed_response(parsed: dict, chat_id: str) -> dict:
    content = parsed.get("content", "") or ""
    thoughts = parsed.get("thoughts", "") or ""
    stats = parsed.get("stats", "") or ""
    final = parsed.get("final_thoughts", "") or ""
    active_events = read_active_events_local(chat_id)
    mention_present = False
    combined = (thoughts + "\\n" + content + "\\n" + final).lower()
    for ev in active_events:
        if ev.get("text", "").lower() in combined or "event:" in combined:
            mention_present = True
            break
    if active_events and not mention_present:
        evtext = active_events[0].get("text", "")
        if thoughts.strip():
            thoughts = f"EVENT: {evtext}\\n" + thoughts
        else:
            thoughts = f"EVENT: {evtext}"
        content = f"(AUTO-INJECTED EVENT: {evtext})\\n" + content
    sentences = re.split(r'(?<=[.!?])\\s+', content)
    kept = []
    moved = []
    for s in sentences:
        s_l = s.lower()
        if re.search(r'\\b( he | she | they | his | her | their )\\b', ' ' + s_l):
            moved.append(s.strip())
        else:
            kept.append(s.strip())
    if moved:
        content = (' '.join([x for x in kept if x])).strip()
        moved_block = "\\n".join(moved)
        if final.strip():
            final = final + "\\n" + moved_block
        else:
            final = moved_block
    if len(re.findall(r'\\b( he | she | they | his | her | their )\\b', ' ' + content.lower())) > 0 and not content.lower().strip().startswith("(AUTO-INJECTED EVENT"):
        content = "(NOTE: Please answer in first person.)\\n" + content
    parsed["content"] = content.strip()
    parsed["thoughts"] = thoughts.strip()
    parsed["stats"] = stats.strip()
    parsed["final_thoughts"] = final.strip()
    return parsed

app_fixed = FastAPI()
from fastapi.staticfiles import StaticFiles as SF
app_fixed.mount("/static", SF(directory=STATIC_DIR), name="static")

try:
    HTML_UI = globals().get("HTML_UI", None)
    if HTML_UI:
        @app_fixed.get("/", response_class=HTMLResponse)
        async def root_index():
            return HTML_UI
    else:
        @app_fixed.get("/", response_class=HTMLResponse)
        async def root_index():
            return "<html><body><h2>Patched server running — but original UI not found.</h2></body></html>"
except Exception:
    pass

@app_fixed.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    try:
        active_chat = "default_chat"
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type":"error", "message":"invalid json"}))
                continue
            if data.get("type") == "init":
                active_chat = data.get("chat_id", "default_chat")
                await websocket.send_text(json.dumps({"type":"info","message":"initialized","chat_id":active_chat}))
                continue
            if "message" in data and data.get("chat_id"):
                active_chat = data.get("chat_id")
                user_msg = data.get("message", "")
                user_ts = append_message_local(active_chat, "user", user_msg)
                await websocket.send_text(json.dumps({"type":"user_ts", "ts": user_ts}))
                temp_ts = "generating_" + str(int(time.time()*1000))
                await websocket.send_text(json.dumps({"type":"start","temp_ts": temp_ts}))
                try:
                    generator = globals().get("generate_fake_response_stream", None)
                    full_text = ""
                    if callable(generator):
                        async for chunk in generator(user_msg, active_chat):
                            full_text += chunk
                            await websocket.send_text(json.dumps({"type":"partial", "chunk": chunk}))
                    else:
                        async def fallback_gen():
                            active_events = read_active_events_local(active_chat)
                            think_lines = ["I am reasoning about the user's request."]
                            if active_events:
                                for ev in active_events:
                                    think_lines.append("EVENT: " + ev.get("text","(event)"))
                            yield "<think>" + "\\n".join(think_lines) + "</think>\\n\\n"
                            main = f"I read your message: {user_msg}\\nI respond now.\\n"
                            for i in range(0, len(main), 60):
                                yield main[i:i+60]
                                await asyncio.sleep(0.06)
                            yield "\\n\\n**[[Stats]]**\\n**Happiness**: 40%\\n"
                            yield "\\n\\n**[[Final Thoughts]]**\\nDone."
                        async for chunk in fallback_gen():
                            full_text += chunk
                            await websocket.send_text(json.dumps({"type":"partial", "chunk": chunk}))
                    parser = globals().get("parse_full_response", None)
                    if callable(parser):
                        parsed = parser(full_text)
                    else:
                        parsed = {"thoughts":"", "content":full_text, "stats":"", "final_thoughts":""}
                    parsed = postprocess_parsed_response(parsed, active_chat)
                    assistant_ts = append_message_local(active_chat, "assistant", parsed.get("content",""), thoughts=parsed.get("thoughts",""), stats=parsed.get("stats",""), final_thoughts=parsed.get("final_thoughts",""))
                    await websocket.send_text(json.dumps({"type":"assistant_ts","ts":assistant_ts}))
                    await websocket.send_text(json.dumps({"type":"done"}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type":"error","message":str(e)}))
                continue
            if data.get("type") == "regenerate":
                old_ts = data.get("ts")
                active_chat = data.get("chat_id", active_chat)
                delete_message_local(active_chat, old_ts)
                temp_ts = "generating_" + str(int(time.time()*1000))
                await websocket.send_text(json.dumps({"type":"start", "old_ts": old_ts, "temp_ts": temp_ts}))
                try:
                    async def regen_gen():
                        yield "<think>Regenerating response.</think>\\n"
                        yield "Regenerated content."
                        await asyncio.sleep(0.02)
                        yield "\\n\\n**[[Stats]]**\\n**Happiness**: 35%\\n"
                        yield "\\n\\n**[[Final Thoughts]]**\\nRegenerated."
                    full_text = ""
                    async for chunk in regen_gen():
                        full_text += chunk
                        await websocket.send_text(json.dumps({"type":"partial", "chunk": chunk}))
                    parser = globals().get("parse_full_response", None)
                    if callable(parser):
                        parsed = parser(full_text)
                    else:
                        parsed = {"thoughts":"", "content":full_text, "stats":"", "final_thoughts":""}
                    parsed = postprocess_parsed_response(parsed, active_chat)
                    assistant_ts = append_message_local(active_chat, "assistant", parsed.get("content",""), thoughts=parsed.get("thoughts",""), stats=parsed.get("stats",""), final_thoughts=parsed.get("final_thoughts",""))
                    await websocket.send_text(json.dumps({"type":"assistant_ts","ts":assistant_ts}))
                    await websocket.send_text(json.dumps({"type":"done"}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type":"error","message":str(e)}))
                continue
            if data.get("stop"):
                await websocket.send_text(json.dumps({"type":"stopped"}))
                continue
            await websocket.send_text(json.dumps({"type":"error","message":"unknown command"}))
    except Exception:
        try:
            await websocket.close()
        except:
            pass

@app_fixed.post("/inject_event")
async def api_inject_event(request: Request):
    payload = await request.json()
    chat_id = payload.get("chat_id", "default_chat")
    text = payload.get("text", "")
    ptype = payload.get("persistence_type", "messages")
    value = int(payload.get("value", 1))
    ev = inject_event_local(chat_id, text, ptype, value)
    return JSONResponse({"ok": True, "event": ev})

@app_fixed.post("/{chat_id}/edit_message")
async def api_edit_message(chat_id: str, payload: dict):
    ts = payload.get("ts")
    raw = payload.get("raw_content", "")
    if not ts:
        return JSONResponse({"ok": False, "error": "ts missing"}, status_code=400)
    parser = globals().get("parse_full_response", None)
    if not callable(parser):
        def parser(x): return {"thoughts":"", "content":x, "stats":"", "final_thoughts":""}
    parsed = parser(raw)
    parsed = postprocess_parsed_response(parsed, chat_id)
    try:
        p = messages_file_local(chat_id)
        lines = []
        if os.path.exists(p):
            with open(p, "r", encoding='utf-8') as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
        out = []
        updated = False
        for ln in lines:
            try:
                j = json.loads(ln)
            except:
                continue
            if j.get("ts") == ts:
                j["content"] = parsed.get("content","")
                j["thoughts"] = parsed.get("thoughts","")
                j["stats"] = parsed.get("stats","")
                j["final_thoughts"] = parsed.get("final_thoughts","")
                updated = True
            out.append(j)
        if not updated:
            return JSONResponse({"ok": False, "error": "message not found"}, status_code=404)
        with open(p, "w", encoding='utf-8') as f:
            for j in out:
                f.write(json.dumps(j, ensure_ascii=False) + "\\n")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    rendered = ""
    if parsed.get("thoughts"):
        rendered += f"<div class='thought-container'><div class='thought-content'>{parsed['thoughts']}</div></div>"
    if parsed.get("content"):
        rendered += f"<div class='message-content'>{parsed['content']}</div>"
    if parsed.get("stats"):
        rendered += f"<div class='stats-container'>{parsed['stats']}</div>"
    if parsed.get("final_thoughts"):
        rendered += f"<div class='final-thoughts-container'>{parsed['final_thoughts']}</div>"
    return JSONResponse({"ok": True, "updated_message": {"content": parsed["content"], "thoughts": parsed["thoughts"], "stats": parsed["stats"], "final_thoughts": parsed["final_thoughts"], "rendered_html": rendered}})

# Replace global app variable so uvicorn picks the fixed app
app = app_fixed

# Ensure embed status endpoint exists on the final app (used by the frontend)
@app.get("/embed_status")
async def embed_status_final():
    try:
        return JSONResponse({
            "status": "ok",
            "embeddings_enabled": bool(globals().get("EMBEDDINGS_ENABLED", False)),
            "embedding_api_url": os.getenv("EMBEDDING_API_URL", None)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)






# --- Companion endpoints added to avoid 404 for /chats used by the frontend ---
# These endpoints are lightweight and safe. They rely on the helper functions
# defined earlier in the patched file (messages_file_local, read_all_messages_local, etc.).
try:
    from fastapi.responses import JSONResponse
    import os
    # GET /chats - list all chat directories and basic metadata
    
    @app.get("/chats")
    async def list_chats():
        out = []
        try:
            base = globals().get("CHATS_DIR") or (os.path.join(os.path.dirname(__file__), "chats"))
            if not os.path.exists(base):
                return JSONResponse([], status_code=200)
            for name in sorted(os.listdir(base)):
                path = os.path.join(base, name)
                if os.path.isdir(path):
                    messages_file = os.path.join(path, "messages.jsonl")
                    msg_count = 0
                    if os.path.exists(messages_file):
                        try:
                            with open(messages_file, "r", encoding="utf-8") as f:
                                msg_count = sum(1 for ln in f if ln.strip())
                        except Exception:
                            msg_count = 0
                    out.append({"chat_id": name, "messages": msg_count})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse(out)

    # GET /chats/{chat_id} - return all messages for a chat
    @app.get("/chats/{chat_id}")
    async def get_chat(chat_id: str):
        try:
            # Prefer existing helper if present
            reader = globals().get("read_all_messages_local") or globals().get("read_all_messages") 
            if callable(reader):
                messages = reader(chat_id)
            else:
                # fallback: read messages.jsonl manually
                base = globals().get("CHATS_DIR") or (os.path.join(os.path.dirname(__file__), "chats"))
                mf = os.path.join(base, chat_id, "messages.jsonl")
                messages = []
                if os.path.exists(mf):
                    with open(mf, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln: continue
                            try: messages.append(json.loads(ln))
                            except: continue
            return JSONResponse({"chat_id": chat_id, "messages": messages})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # POST /chats - create a new chat (body: {"chat_id":"myid"})
    @app.post("/chats")
    async def create_chat(payload: dict):
        cid = payload.get("chat_id") if isinstance(payload, dict) else None
        if not cid:
            return JSONResponse({"error":"chat_id missing"}, status_code=400)
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', cid)
        path = os.path.join(globals().get("CHATS_DIR") or (os.path.join(os.path.dirname(__file__), "chats")), safe)
        os.makedirs(path, exist_ok=True)
        # initialize empty messages file if missing
        mf = os.path.join(path, "messages.jsonl")
        if not os.path.exists(mf):
            open(mf, "w", encoding="utf-8").close()
        return JSONResponse({"ok": True, "chat_id": safe})
except Exception as e:
    # If something goes wrong in app import time, print an error to the server logs
    print("Failed to append /chats endpoints:", e)
