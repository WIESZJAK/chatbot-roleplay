# gem.py
"""
Advanced roleplay server with proper streaming and modern UI
Usage:
pip install fastapi uvicorn requests python-dotenv numpy
uvicorn gem:app --reload --port 7860
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import os, time, json, threading, asyncio, requests, re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np
import uvicorn

# optional faiss
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

load_dotenv()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "rp_data")
PERSONAS_DIR = os.path.join(DATA_DIR, "personas")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSONAS_DIR, exist_ok=True)

MESSAGES_FILE = os.path.join(DATA_DIR, "messages.jsonl")
SUMMARY_FILE = os.path.join(DATA_DIR, "summary.txt")
PERSONA_FILE = os.path.join(DATA_DIR, "persona.json") # This will now store the currently active persona
EMBEDDINGS_NPY = os.path.join(DATA_DIR, "embeddings.npy")
EMBEDDINGS_META = os.path.join(DATA_DIR, "embeddings_meta.jsonl")
NEW_DAY_FILE = os.path.join(DATA_DIR, "new_day.txt")
THOUGHTS_FILE = os.path.join(DATA_DIR, "thoughts.txt")
# ADDED: Session counting file
SESSION_COUNT_FILE = os.path.join(DATA_DIR, "session_count.txt")


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

# Global stop flag
stop_generation = threading.Event()

# Default persona. This serves as a fallback and a structural template if persona.json is missing or invalid.
# The persona loaded from the UI will completely override this during runtime.
DEFAULT_PERSONA = {
    "name": "Vex",
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
        "1.  First, you MUST ALWAYS think step-by-step before responding. Put ALL your thoughts in <think> </think> tags BEFORE your actual response. The <think> block must be detailed and reflect your reasoning.\n"
        "2.  After the <think> block, write your main response to the user.\n"
        "3.  Finally, after ALL other content, at the VERY END of your entire message, you MUST include a section for your current emotional and mental state. This section MUST start with the exact marker **[[Stats]]** followed by a list of your feelings and their intensity in percentages. Use bold for the names and percentages. The stats should be relevant to the context of the conversation and your persona.\n"
        "INCORRECT ordering will be rejected. The order is ALWAYS: <think> -> response -> **[[Stats]]**.\n"
        "Example format:\n"
        "<think>My thought process goes here.</think>\n"
        "This is my main response to the user.\n"
        "**[[Stats]]**\n"
        "**Happiness**: 35%\n"
        "**Attraction**: 5%\n"
        "**Curiosity**: 90%\n"
    ),
    "censor_list": [],
    "prompt_examples": []
}

app = FastAPI()
EMBEDDINGS_ENABLED = False

# ---------------- Utilities ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def append_message_to_disk(role: str, content: str, meta: Optional[Dict[str,Any]] = None, thoughts: Optional[str] = None, stats: Optional[str] = None):
    rec = {"ts": now_iso(), "role": role, "content": content}
    if meta:
        rec["meta"] = meta
    if thoughts:
        rec["thoughts"] = thoughts
    if stats:
        rec["stats"] = stats
    with open(MESSAGES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec["ts"]

def read_last_messages(k: int) -> List[Dict[str, Any]]:
    if not os.path.exists(MESSAGES_FILE):
        return []
    with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    last = lines[-k:] if len(lines) > k else lines
    out = []
    for ln in last:
        try:
            out.append(json.loads(ln))
        except:
            continue
    return out

def read_all_messages() -> List[Dict[str, Any]]:
    if not os.path.exists(MESSAGES_FILE):
        return []
    with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except:
            continue
    return out

def save_all_messages(messages: List[Dict[str, Any]]):
    with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

def delete_message_by_ts(ts: str):
    messages = read_all_messages()
    messages = [msg for msg in messages if msg.get("ts") != ts]
    save_all_messages(messages)
    # Optionally remove from embeddings
    if EMBEDDINGS_ENABLED:
        embs, meta = get_all_embeddings()
        if embs.size > 0:
            keep_idx = [i for i, m in enumerate(meta) if m.get("ts") != ts]
            if keep_idx:
                embs = embs[keep_idx]
                meta = [meta[i] for i in keep_idx]
            else:
                embs = np.zeros((0,))
                meta = []
            np.save(EMBEDDINGS_NPY, embs)
            with open(EMBEDDINGS_META, "w", encoding="utf-8") as f:
                for m in meta:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

def load_persona() -> Dict[str, Any]:
    if os.path.exists(PERSONA_FILE):
        try:
            with open(PERSONA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    with open(PERSONA_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_PERSONA, f, ensure_ascii=False, indent=2)
    return DEFAULT_PERSONA

def save_persona(persona: Dict[str,Any]):
    with open(PERSONA_FILE, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)

def load_summary() -> str:
    if not os.path.exists(SUMMARY_FILE):
        return ""
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_last_day_summary() -> str:
    if not os.path.exists(SUMMARY_FILE):
        return "No summaries available."
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
    summaries = content.split("\n\n")
    if summaries:
        return summaries[-1]
    return "No summaries available."

def append_summary(text: str, date: str, additional: str = ""):
    summary_entry = f"Summary for {date}:\n{text}{additional}\n\n"
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(summary_entry)

# MODIFIED: Reworked to include session counting
def mark_new_day():
    summarize_older_messages_once(force=True)
    prev_date_str = ""
    if os.path.exists(NEW_DAY_FILE):
        with open(NEW_DAY_FILE, "r", encoding="utf-8") as f:
            prev_date_str = f.read().strip()
    
    # Session counter logic
    session_count = 1
    if os.path.exists(SESSION_COUNT_FILE):
        try:
            with open(SESSION_COUNT_FILE, "r") as f:
                session_count = int(f.read().strip()) + 1
        except (ValueError, FileNotFoundError):
            session_count = 1 # Fallback
    with open(SESSION_COUNT_FILE, "w") as f:
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
        additional_info = f"\nStarting new day on {current_date}. {days_passed} days have passed since the last interaction. Current session: {session_count}"
    else:
        additional_info = f"\nStarting new session on {current_date}. 0 days have passed. (Same date, but new session started.) Current session: {session_count}"
        
    append_summary("", current_date, additional_info)
    append_message_to_disk("system", marker)
    with open(NEW_DAY_FILE, "w", encoding="utf-8") as f:
        f.write(d)
    return marker


# MODIFIED: Added session count file to cleanup
def clear_memory_files():
    for file in [MESSAGES_FILE, EMBEDDINGS_NPY, EMBEDDINGS_META, SUMMARY_FILE, NEW_DAY_FILE, THOUGHTS_FILE, SESSION_COUNT_FILE]:
        if os.path.exists(file):
            os.remove(file)
    open(MESSAGES_FILE, "w", encoding="utf-8").close()
    open(SUMMARY_FILE, "w", encoding="utf-8").close()
    open(NEW_DAY_FILE, "w", encoding="utf-8").close()
    open(THOUGHTS_FILE, "w", encoding="utf-8").close()
    # ADDED: Initialize session count on clear
    with open(SESSION_COUNT_FILE, "w") as f:
        f.write("0")

def fix_mojibake(s: str) -> str:
    encodings = ['latin1', 'cp1252', 'iso-8859-1', 'utf-8', 'utf-16', 'utf-32']
    for enc in encodings:
        try:
            return s.encode(enc, errors='ignore').decode('utf-8', errors='ignore')
        except:
            pass
    return s

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = fix_mojibake(s)
    s = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', s)
    s = s.replace('[DONE]', '')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'Ã¢Â€Â”', '—', s)
    s = re.sub(r'Ã¢Â€Â™', '’', s)
    s = re.sub(r'Ã¢Â€Âœ', '“', s)
    s = re.sub(r'Ã¢Â€Â ', '”', s)
    s = re.sub(r'Ã¢Â€Â¦', '…', s)
    return s

# ---------------- Embeddings / Memory ----------------
def get_all_embeddings():
    if not os.path.exists(EMBEDDINGS_NPY) or not os.path.exists(EMBEDDINGS_META):
        return np.zeros((0,)), []
    try:
        embs = np.load(EMBEDDINGS_NPY)
    except Exception:
        embs = np.zeros((0,))
    meta = []
    with open(EMBEDDINGS_META, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                meta.append(json.loads(ln))
            except:
                meta.append({})
    return embs, meta

def append_embedding(vec: np.ndarray, meta: Dict[str,Any]):
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if not os.path.exists(EMBEDDINGS_NPY):
        np.save(EMBEDDINGS_NPY, vec)
    else:
        try:
            old = np.load(EMBEDDINGS_NPY)
            if old.size == 0:
                combined = vec
            else:
                combined = np.vstack([old, vec])
            np.save(EMBEDDINGS_NPY, combined)
        except Exception:
            np.save(EMBEDDINGS_NPY, vec)
    with open(EMBEDDINGS_META, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def compute_embedding(text: str) -> Optional[np.ndarray]:
    global EMBEDDINGS_ENABLED
    if not EMBEDDINGS_ENABLED:
        return None
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    payload = {"model": EMBEDDING_MODEL, "input": [text]}
    try:
        r = requests.post(EMBEDDING_API_URL, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        EMBEDDINGS_ENABLED = True
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0 and "embedding" in data["data"][0]:
            return np.array(data["data"][0]["embedding"], dtype=np.float32)
        if "embedding" in data:
            return np.array(data["embedding"], dtype=np.float32)
        return None
    except requests.HTTPError as he:
        status = getattr(he.response, "status_code", None)
        print("Embedding HTTPError:", he)
        if status == 404:
            print("Embeddings endpoint returned 404 — disable or fix EMBEDDING_API_URL.")
            EMBEDDINGS_ENABLED = False
        return None
    except Exception as e:
        print("Embedding error:", e)
        return None

def build_faiss_index(embs: np.ndarray):
    if not HAS_FAISS:
        return None
    if embs is None or embs.size == 0:
        return None
    try:
        dim = embs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embs)
        idx.add(embs)
        return idx
    except Exception:
        return None

def semantic_search(query: str, top_k: int = TOP_K_MEMORIES) -> List[Dict[str,Any]]:
    if not EMBEDDINGS_ENABLED:
        return []
    qv = compute_embedding(query)
    if qv is None:
        return []
    embs, meta = get_all_embeddings()
    if embs is None or embs.size == 0:
        return []
    if embs.ndim == 1 and embs.size != 0:
        embs = embs.reshape(-1, qv.shape[0])
    qv_norm = qv / (np.linalg.norm(qv) + 1e-12)
    try:
        if HAS_FAISS:
            idx = build_faiss_index(embs.copy())
            if idx is not None:
                faiss.normalize_L2(embs)
                D, I = idx.search(qv_norm.reshape(1, -1), top_k)
                return [meta[int(i)] for i in I[0] if i >= 0]
    except Exception:
        pass
    try:
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sims = (embs_norm @ qv_norm).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [meta[int(i)] for i in top_idx]
    except Exception:
        return []

# ---------------- Chat model & parsing ----------------
def extract_content_from_json_chunk(part: Any) -> Optional[str]:
    try:
        if not isinstance(part, dict):
            return None
        if "choices" in part and isinstance(part["choices"], list) and len(part["choices"]) > 0:
            first = part["choices"][0]
            if isinstance(first, dict) and "delta" in first and isinstance(first["delta"], dict):
                if "content" in first["delta"]:
                    return first["delta"]["content"]
                return None
            if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                if "content" in first["message"]:
                    return first["message"]["content"]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        for k in ("result", "text", "output"):
            if k in part and isinstance(part[k], str):
                return part[k]
    except Exception:
        return None
    return None

def _parse_chat_response_json(data: Any) -> str:
    if isinstance(data, dict):
        c = extract_content_from_json_chunk(data)
        if c is not None:
            return clean_text(c)
    try:
        return clean_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        return ""

def call_chat_model_raw(messages: List[Dict[str,str]], stream: bool=False, timeout:int=300, settings: Dict[str, Any] = None):
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    payload = {"model": TEXT_MODEL_NAME, "messages": messages}
    if settings:
        if settings.get("temperature") is not None:
            payload["temperature"] = float(settings["temperature"])
        if settings.get("max_tokens") is not None:
            payload["max_tokens"] = int(settings["max_tokens"])
    if stream:
        payload["stream"] = True
    try:
        if stream:
            r = requests.post(MODEL_API_URL, json=payload, headers=headers, stream=True, timeout=timeout)
            r.encoding = 'utf-8'
            r.raise_for_status()
            return r
        else:
            r = requests.post(MODEL_API_URL, json=payload, headers=headers, timeout=timeout)
            r.encoding = 'utf-8'
            r.raise_for_status()
            return r.json()
    except requests.HTTPError as he:
        return {"_error": str(he), "_status": getattr(he.response, "status_code", None)}
    except Exception as e:
        return {"_error": str(e)}

# ---------------- Summarization background ----------------
def summarize_older_messages_once(force: bool = False) -> str:
    if not os.path.exists(MESSAGES_FILE):
        return "No messages to summarize."
    with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not force and len(lines) <= RECENT_MSGS:
        return "Not enough messages to trigger summarization."
    
    selected = lines if force else lines[:-RECENT_MSGS]
    if not selected:
        return "No older messages to summarize."
        
    joined = []
    for ln in selected:
        try:
            rec = json.loads(ln)
            joined.append(f"[{rec['role']} @ {rec['ts']}] {rec['content']}")
        except Exception:
            continue
    text_to_summarize = "\n".join(joined)
    if not text_to_summarize.strip():
        return "No content to summarize."
        
    persona = load_persona()
    system_msgs = [{"role": "system", "content": (
        "You are a roleplay memory summarizer. Produce concise bullet points/facts. "
        "Include important facts, relationships, promises, objects, locations and emotional arcs. Keep summary short."
    )}]
    messages = system_msgs + [{"role": "user", "content": "Summarize:\n" + text_to_summarize}]
    resp = call_chat_model_raw(messages, stream=False)
    
    if isinstance(resp, dict) and "_error" in resp:
        error_msg = f"Summarization error: {resp.get('_error')}"
        print(error_msg)
        return error_msg
        
    summary_text = _parse_chat_response_json(resp)
    censor = persona.get("censor_list", []) or []
    for w in censor:
        summary_text = summary_text.replace(w, "xxx")
        
    current_date = datetime.now().strftime("%Y-%m-%d")
    append_summary(summary_text, current_date, "")
    return summary_text


def background_summarizer_daemon():
    while True:
        try:
            if os.path.exists(MESSAGES_FILE):
                with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
                    total_lines = sum(1 for _ in f)
                if total_lines > RECENT_MSGS and total_lines % SUMMARIZE_EVERY == 0:
                    try:
                        summarize_older_messages_once(force=False)
                    except Exception as e:
                        print("Background summarize error:", e)
        except Exception:
            pass
        time.sleep(BACKGROUND_SUMMARY_INTERVAL)

bg_thread = threading.Thread(target=background_summarizer_daemon, daemon=True)
bg_thread.start()

# ---------------- Prompt builder ----------------
# MODIFIED: Major rewrite to handle "New Day" logic and improve conversational style
def build_prompt_with_memory(user_msg: str, thought_ratio: float = 0.5, talkativeness: float = 0.5) -> List[Dict[str,str]]:
    persona = load_persona()
    summary = load_summary()
    system_parts = []
    critical_context_notice = ""

    # Check for a new day/session start by looking at the very end of the summary
    is_new_day_session = False
    if "Starting new day" in summary[-250:] or "Starting new session" in summary[-250:]:
        is_new_day_session = True

    if is_new_day_session:
        days_passed_match = re.search(r'(\d+) days have passed', summary)
        days_passed = int(days_passed_match.group(1)) if days_passed_match else 0
        
        greeting_instruction = ""
        if days_passed > 0:
            greeting_instruction = (
                f"A new day has begun ({days_passed} days passed). "
                "Start your response with a morning greeting appropriate for a new day. "
                "You can briefly mention something memorable from the previous day's summary to show you remember."
            )
        else: # 0 days have passed, it's a new session on the same day
            greeting_instruction = (
                "This is a new session on the same day. "
                "Start your response with a greeting as if you just woke up from a nap or returned after a short break. "
                "Acknowledge the user and continue the roleplay."
            )
        
        critical_context_notice = (
            "**CRITICAL CONTEXT: A NEW DAY/SESSION HAS BEGUN!**\n"
            "This is your first interaction in this new context. Your first response MUST acknowledge this.\n"
            f"- **Instructions:** {greeting_instruction}\n"
            "- After the greeting, respond to the user's message as usual.\n"
        )

    system_parts.append(f"You are: {persona.get('name')}. {persona.get('short_description')}")
    # MODIFIED: Added more direct instructions for dialogue
    system_parts.append("You are speaking to a male user. Address him with appropriate masculine pronouns (he/him).")
    system_parts.append("IMPORTANT: Your response should be in the first person. Engage in direct dialogue with the user, rather than providing a third-person narrative description of your actions. Speak as your character.")
    if persona.get("traits"):
        system_parts.append("Traits: " + ", ".join(persona.get("traits", [])))
    if persona.get("history"):
        system_parts.append("History: " + persona.get("history"))
    if persona.get("behavior_instructions"):
        system_parts.append("Behavior rules: " + persona.get("behavior_instructions"))

    current_time = datetime.utcnow().isoformat() + "Z"
    system_parts.append(f"Current time: {current_time}. Use this to track time and day/night in your responses.")

    all_msgs_count = 0
    if os.path.exists(MESSAGES_FILE):
        with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
            all_msgs_count = sum(1 for _ in f)
            
    if all_msgs_count == 0:
        system_parts.append("Brevity: This is the first user message ever. Reply concisely (1-3 sentences).")
    else:
        system_parts.append("Style: match user tone; prefer concise responses unless user requests long form.")

    ex = persona.get("prompt_examples", [])
    if ex:
        ex_text = "Examples:\n"
        for e in ex:
            ex_text += f"USER: {e.get('user', '')}\nASSISTANT: {e.get('assistant', '')}\n"
        system_parts.append(ex_text)
        
    if summary:
        system_parts.append("Memory summary:\n" + (summary if len(summary) < 2000 else summary[-2000:]))
    
    output_instructions = persona.get("output_instructions")
    if output_instructions:
        system_parts.append(output_instructions)
    else: # Fallback to default if not set in persona file
        system_parts.append(DEFAULT_PERSONA["output_instructions"])
    
    if thought_ratio < 0.3:
        system_parts.append("Thought Ratio: LOW. Keep your thoughts brief and concise. The priority is a quick, direct response.")
    elif thought_ratio < 0.7:
        system_parts.append("Thought Ratio: MEDIUM. Use a balanced approach between a detailed thought process and a well-structured response.")
    else:
        system_parts.append("Thought Ratio: HIGH. Provide an extensive and detailed thought process. The focus is on showing your inner workings, even if the response is shorter.")

    if talkativeness < 0.3:
        system_parts.append("Talkativeness: LOW. Your response should be very concise and to the point. Use fewer sentences and avoid descriptive fluff.")
    elif talkativeness < 0.7:
        system_parts.append("Talkativeness: MEDIUM. Your response should be moderately detailed. Balance between brevity and description.")
    else:
        system_parts.append("Talkativeness: HIGH. Your response should be very talkative, long, and descriptive. Use rich language and elaborate on details, actions, and feelings.")

    # Prepend the critical notice if it exists, otherwise it's an empty string
    system_text = critical_context_notice + "\n".join(system_parts)
    
    messages = [{"role": "system", "content": system_text}]
    relevant = semantic_search(user_msg, TOP_K_MEMORIES)
    if relevant:
        mem_texts = []
        for m in relevant:
            mem_texts.append(f"[{m.get('role', '?')} @ {m.get('ts', '?')}] {m.get('content', '')}")
        messages.append({"role": "system", "content": "Relevant memories:\n" + "\n".join(mem_texts)})
        
    recent = read_last_messages(RECENT_MSGS)
    for r in recent:
        if r.get("role") in ("user", "assistant", "system"):
            messages.append({"role": r["role"], "content": r["content"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


# ---------------- Modern UI ----------------
# MODIFIED: JavaScript for better scrolling behavior
HTML_UI = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Roleplay Assistant</title>
<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --bg-primary: #0a0a0a;
  --bg-secondary: #141414;
  --bg-tertiary: #1a1a1a;
  --text-primary: #ffffff;
  --text-secondary: #a0a0a0;
  --accent: #6366f1;
  --accent-hover: #4f46e5;
  --user-bubble: #262626;
  --border: rgba(255, 255, 255, 0.1);
  --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  height: 100vh;
  display: flex;
  overflow: hidden;
}

.container {
  display: flex;
  width: 100%;
  height: 100%;
}

.main-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg-secondary);
  border-radius: 12px;
  margin: 12px;
  overflow: hidden;
}

.chat-header {
  background: var(--bg-tertiary);
  padding: 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.chat-header h1 {
  font-size: 1.25rem;
  font-weight: 600;
}

.status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
}

.status-dot.connected {
  background: #10b981;
}

.status-dot.generating {
  background: #f59e0b;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chat-messages::-webkit-scrollbar {
  width: 8px;
}

.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

.message {
  display: flex;
  gap: 12px;
  max-width: 80%;
  animation: fadeIn 0.3s ease-in;
  position: relative;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user {
  align-self: flex-end;
}

.message.assistant {
  align-self: flex-start;
}

.message.system {
  align-self: center;
  max-width: 90%;
  opacity: 0.6;
}

.message-body {
    display: flex;
    flex-direction: column;
    width: 100%;
}

.message-content {
  font-size: 0.95rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  position: relative;
  overflow-wrap: anywhere;
  padding: 4px 0;
}

.message.user .message-content {
    background: var(--user-bubble);
    padding: 10px 15px;
    border-radius: 18px;
    text-align: left;
}

.message.system .message-content {
  background: transparent;
  border: 1px solid var(--border);
  text-align: center;
  font-size: 0.875rem;
  color: var(--text-secondary);
  padding: 8px 16px;
  border-radius: 18px;
}
.thought-container {
  font-size: 0.85em;
  color: var(--text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
  font-family: monospace;
  border-bottom: 1px solid var(--border);
  padding: 8px 0 12px 0;
  margin-bottom: 8px;
  cursor: pointer;
  overflow: hidden;
  max-height: 70px; /* Increased height for preview */
  transition: max-height 0.3s ease-in-out;
  position: relative;
}
.thought-container .thought-content {
    overflow: hidden;
    height: 100%;
}
.thought-container::before {
    content: 'Thoughts ▼';
    font-weight: bold;
    display: block;
    margin-bottom: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #fff;
    position: sticky;
    top: 0;
    background: var(--bg-secondary);
}
.thought-container.expanded {
    max-height: 500px; /* MODIFIED: Set a max-height for scrolling */
    overflow-y: auto;
}
.thought-container.expanded::before {
    content: 'Thoughts ▲';
}

.stats-container {
    padding: 8px 0 0 0;
    margin-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 0.85em;
}

.message-meta {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}
.message.user .message-meta {
    text-align: right;
}

.message:hover .message-meta {
  opacity: 1;
}

.delete-btn {
  background: transparent;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
  font-size: 0.75rem;
  padding: 4px;
  position: absolute;
  top: -5px;
  display: none; 
}
.message.user:hover .delete-btn { right: -5px; }
.message.assistant:hover .delete-btn { left: -5px; }


.message:hover .delete-btn {
  opacity: 1;
  display: block; 
}

.chat-input-container {
  padding: 20px;
  background: var(--bg-tertiary);
  border-top: 1px solid var(--border);
}

.chat-input-wrapper {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.chat-input {
  flex: 1;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 12px 20px;
  color: var(--text-primary);
  font-size: 0.95rem;
  resize: none;
  outline: none;
  transition: border-color 0.2s;
  max-height: 120px;
  overflow-y: auto;
}

.chat-input:focus {
  border-color: var(--accent);
}

.send-button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 50%;
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.send-button:hover:not(:disabled) {
  background: var(--accent-hover);
  transform: scale(1.05);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.stop-button {
  background: #ef4444;
}

.stop-button:hover {
  background: #dc2626;
}

.side-panel {
  width: 320px;
  background: var(--bg-tertiary);
  padding: 20px;
  border-left: 1px solid var(--border);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.panel-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.panel-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  cursor: pointer;
}
.panel-title::after {
    content: ' ▼';
    font-size: 0.8em;
}
.panel-title.collapsed::after {
    content: ' ►';
}


.action-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.btn {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
}

.btn:hover {
  background: var(--accent);
  border-color: var(--accent);
}

.btn.danger:hover {
  background: #ef4444;
  border-color: #ef4444;
}

.collapsible-content {
    display: none;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    margin-top: 10px;
}
.collapsible-content.show {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.slider-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-size: 0.875rem;
    color: var(--text-secondary);
}
.slider-container label {
    display: flex;
    justify-content: space-between;
}
.slider-container input[type=range] {
    width: 100%;
}
.side-panel-input {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    padding: 8px;
}

.responding-indicator {
    color: var(--text-secondary);
    font-style: italic;
    animation: fadeIn 0.5s ease-in-out;
}
.responding-indicator .dot {
    animation: blink 1.4s infinite both;
}
.responding-indicator .dot:nth-child(2) { animation-delay: .2s; }
.responding-indicator .dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
}

/* Modal Styles */
.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-content {
  background: var(--bg-secondary);
  padding: 25px;
  border-radius: 12px;
  width: 90%;
  max-width: 700px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  gap: 15px;
}
.modal-content h2 { font-size: 1.2rem; }
.modal-close-btn { align-self: flex-end; background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer; }
.form-group { display: flex; flex-direction: column; gap: 8px; }
.form-group label { font-size: 0.9rem; color: var(--text-secondary); }
.modal-input, .modal-textarea {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
  color: var(--text-primary);
  font-size: 0.9rem;
}
.modal-textarea { resize: vertical; min-height: 200px; font-family: monospace; }
.modal-footer { display: flex; gap: 10px; justify-content: flex-end; }
.modal-footer .btn-group { display: flex; gap: 10px; flex-grow: 1; }

/* Mobile responsive */
@media (max-width: 768px) {
  .container { flex-direction: column; }
  .side-panel { width: 100%; border-left: none; border-top: 1px solid var(--border); max-height: 40vh; }
  .message { max-width: 90%; }
}
.icon { width: 20px; height: 20px; fill: currentColor; }
</style>
</head>
<body>
<div class="container">
  <div class="main-panel">
    <div class="chat-header">
      <h1>AI Roleplay Assistant</h1>
      <div class="status">
        <span id="status-text">Disconnected</span>
        <span id="status-dot" class="status-dot"></span>
      </div>
    </div>
    <div id="chat-messages" class="chat-messages"></div>
    <div class="chat-input-container">
      <div class="chat-input-wrapper">
        <textarea id="chat-input" class="chat-input" placeholder="Type your message..." rows="1"></textarea>
        <button id="send-btn" class="send-button">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
        </button>
        <button id="stop-btn" class="send-button stop-button" style="display: none;">
          <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
        </button>
      </div>
    </div>
  </div>
  <div class="side-panel">
    <div class="panel-section">
      <div class="panel-title panel-toggle">Quick Actions</div>
      <div class="collapsible-content">
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
          <div class="slider-container">
              <label>Temperature: <span id="temp-value">1.0</span></label>
              <input type="range" id="temperature-slider" min="0.1" max="2.0" step="0.1" value="1.0">
          </div>
          <div class="slider-container">
              <label>Max Tokens: <span id="tokens-value">1024</span></label>
              <input type="range" id="tokens-slider" min="100" max="4096" step="50" value="1024">
          </div>
          <div class="slider-container">
              <label>Thought Ratio: <span id="thought-ratio-value">0.5</span></label>
              <input type="range" id="thought-ratio-slider" min="0.0" max="1.0" step="0.1" value="0.5">
          </div>
          <div class="slider-container">
              <label>Talkativeness: <span id="talkativeness-value">0.5</span></label>
              <input type="range" id="talkativeness-slider" min="0.0" max="1.0" step="0.1" value="0.5">
          </div>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">Persona</div>
      <div class="collapsible-content">
          <label for="side-panel-persona-preset" style="font-size:0.875rem; color:var(--text-secondary)">Quick Load Preset</label>
          <div style="display:flex; gap: 8px;">
            <select id="side-panel-persona-preset" class="side-panel-input" style="flex:1;"></select>
            <button class="btn" id="side-panel-load-btn">Load</button>
          </div>
          <button class="btn" id="open-persona-modal" style="width:100%; margin-top: 10px;">Full Persona Editor</button>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-title panel-toggle">System Info</div>
      <div class="collapsible-content">
          <div style="font-size: 0.875rem; color: var(--text-secondary);">
            <div>Embeddings: <span id="embed-status">disabled</span></div>
            <div style="margin-top: 8px;"><button class="btn" id="test-embed" style="width: 100%;">Test Embeddings</button></div>
          </div>
      </div>
    </div>
  </div>
</div>

<div id="persona-modal" class="modal-backdrop">
  <div class="modal-content">
    <button id="persona-modal-close" class="modal-close-btn">&times;</button>
    <h2>Persona Editor</h2>
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

<script>
let ws = null;
let isGenerating = false;
let currentMessageContainer = null;
let fullResponseText = '';

function connectWebSocket() {
    return new Promise((resolve, reject) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            resolve();
            return;
        }
        ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
        ws.onopen = () => { updateStatus('connected'); resolve(); };
        ws.onclose = () => { updateStatus('disconnected'); isGenerating = false; setTimeout(connectWebSocket, 1000); };
        ws.onerror = (error) => { updateStatus('disconnected'); isGenerating = false; reject(error); };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'start') {
                isGenerating = true;
                updateStatus('generating');
                currentMessageContainer = addMessage('assistant', '', '', '', '');
                const body = currentMessageContainer.querySelector('.message-body');
                const indicator = document.createElement('div');
                indicator.className = 'responding-indicator';
                indicator.innerHTML = 'responding<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
                body.appendChild(indicator);
            } else if (data.type === 'partial') {
                handlePartialMessage(data.chunk);
            } else if (data.type === 'done' || data.type === 'stopped') {
                isGenerating = false;
                updateStatus('connected');
                fullResponseText = ''; 
                currentMessageContainer = null;
            } else if (data.type === 'error') {
                isGenerating = false;
                updateStatus('connected');
                addMessage('system', '[ERROR] ' + data.message);
            } else if (data.type === 'user_ts') {
                const lastUser = getLastMessage('user');
                if (lastUser) lastUser.dataset.ts = data.ts;
            } else if (data.type === 'assistant_ts') {
                const lastAssistant = getLastMessage('assistant');
                if (lastAssistant) {
                    lastAssistant.dataset.ts = data.ts;
                }
            }
        };
    });
}

function simpleMarkdown(text) {
    if (typeof text !== 'string') return '';
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
}

function handlePartialMessage(chunk) {
    if (!currentMessageContainer) return;

    const respondingIndicator = currentMessageContainer.querySelector('.responding-indicator');
    if (respondingIndicator) respondingIndicator.remove();

    fullResponseText += chunk;
    
    let remainingText = fullResponseText;
    let thoughts = '';
    let stats = '';
    let cleanContent = '';

    const thoughtMatch = remainingText.match(/<think>([\s\S]*)/s);
    if (thoughtMatch) {
        let thoughtContent = thoughtMatch[1];
        const endThinkTagIndex = thoughtContent.indexOf('</think>');
        if (endThinkTagIndex !== -1) {
            thoughts = thoughtContent.substring(0, endThinkTagIndex);
            remainingText = thoughtContent.substring(endThinkTagIndex + 8); // length of </think>
        } else {
            thoughts = thoughtContent;
            remainingText = '';
        }
    }

    const statsMatch = remainingText.match(/\*\*\[\[Stats\]\]\*\*([\s\S]*)/i);
    if (statsMatch) {
        stats = statsMatch[0]; // Capture the whole stats block including the title
        cleanContent = remainingText.substring(0, statsMatch.index).trim();
    } else {
        cleanContent = remainingText.trim();
    }
    
    updateOrCreateElement(currentMessageContainer, '.thought-container', thoughts, 'prepend');
    updateOrCreateElement(currentMessageContainer, '.message-content', simpleMarkdown(cleanContent));
    updateOrCreateElement(currentMessageContainer, '.stats-container', simpleMarkdown(stats), 'append');
    scrollToBottom();
}


function updateOrCreateElement(parent, selector, content, position = 'append') {
    let element = parent.querySelector(selector);
    const body = parent.querySelector('.message-body');

    if (!content || content.trim() === '') {
        if (element) element.style.display = 'none';
        return;
    }

    if (!element) {
        element = document.createElement('div');
        element.className = selector.substring(1); // remove dot from selector
        if (position === 'prepend') {
            body.prepend(element);
        } else {
            body.appendChild(element);
        }
    }
    
    element.style.display = 'block';
    
    if (selector === '.thought-container') {
        let thoughtContent = element.querySelector('.thought-content');
        if (!thoughtContent) {
            thoughtContent = document.createElement('div');
            thoughtContent.className = 'thought-content';
            element.appendChild(thoughtContent);
        }
        thoughtContent.innerHTML = simpleMarkdown(content);

        // MODIFIED: If the thought block is collapsed, keep scrolling the main chat window
        if (!element.classList.contains('expanded')) {
            scrollToBottom();
        }

        if (!element.hasToggleListener) {
            element.addEventListener('click', () => {
                element.classList.toggle('expanded');
            });
            element.hasToggleListener = true;
        }
    } else {
        element.innerHTML = content;
    }
}


function updateStatus(status) {
  document.getElementById('status-text').textContent = status.charAt(0).toUpperCase() + status.slice(1);
  document.getElementById('status-dot').className = 'status-dot ' + status;
  toggleSendStopButtons(status === 'generating');
}

function toggleSendStopButtons(showStop) {
  document.getElementById('send-btn').style.display = showStop ? 'none' : 'flex';
  document.getElementById('stop-btn').style.display = showStop ? 'flex' : 'none';
}

function addMessage(role, content = '', ts = '', thoughts = '', stats = '') {
  const messagesContainer = document.getElementById('chat-messages');
  const messageWrapper = document.createElement('div');
  messageWrapper.className = 'message ' + role;
  if (ts) messageWrapper.dataset.ts = ts;
  
  const messageBody = document.createElement('div');
  messageBody.className = 'message-body';
  messageWrapper.appendChild(messageBody);

  updateOrCreateElement(messageWrapper, '.thought-container', thoughts, 'prepend');
  updateOrCreateElement(messageWrapper, '.message-content', simpleMarkdown(content));
  updateOrCreateElement(messageWrapper, '.stats-container', simpleMarkdown(stats), 'append');

  if (role !== 'system') {
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.innerHTML = '🗑️';
    deleteBtn.onclick = (e) => {
        e.preventDefault();
        const timestamp = messageWrapper.dataset.ts;
        if (timestamp) deleteMessage(timestamp, messageWrapper);
    };
    messageWrapper.appendChild(deleteBtn);
  }
  
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  meta.textContent = new Date(ts || Date.now()).toLocaleTimeString();
  messageWrapper.appendChild(meta);
  
  messagesContainer.appendChild(messageWrapper);
  scrollToBottom();
  return messageWrapper;
}

async function deleteMessage(ts, element) {
  if (confirm('Delete this message?')) {
    await api('/delete_message', 'POST', {ts});
    element.remove();
  }
}

function getLastMessage(role) {
  const messages = document.querySelectorAll('.message.' + role);
  return messages[messages.length - 1];
}

// MODIFIED: More robust scrolling
function scrollToBottom() {
    const messages = document.getElementById('chat-messages');
    // Use a small timeout to allow the DOM to update before scrolling
    setTimeout(() => {
        messages.scrollTo({
            top: messages.scrollHeight,
            behavior: 'smooth'
        });
    }, 50);
}


async function sendMessage() {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message || isGenerating) return;
  input.value = '';
  input.style.height = 'auto';
  addMessage('user', message, new Date().toISOString());
  fullResponseText = '';
  
  const settings = {
      temperature: document.getElementById('temperature-slider')?.value || 1.0,
      max_tokens: document.getElementById('tokens-slider')?.value || 1024,
      thought_ratio: document.getElementById('thought-ratio-slider')?.value || 0.5,
      talkativeness: document.getElementById('talkativeness-slider')?.value || 0.5,
  };

  try {
    await connectWebSocket();
    ws.send(JSON.stringify({message, settings}));
  } catch (error) {
    addMessage('system', '[ERROR] Connection failed');
  }
}

function stopGeneration() {
  if (ws && ws.readyState === WebSocket.OPEN && isGenerating) {
    ws.send(JSON.stringify({stop: true}));
  }
}

async function api(path, method = 'GET', body = null) {
  const opts = { method };
  if (body) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const response = await fetch(path, opts);
  if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} ${errorText}`);
  }
  try {
    return await response.json();
  } catch (e) {
    return response.text();
  }
}

async function clearMemory() {
  if (!confirm('Clear all memory? This deletes conversation history and summaries.')) return;
  await api('/clear_memory', 'POST');
  document.getElementById('chat-messages').innerHTML = '';
  updateEmbedStatus();
}

async function newDay() {
  addMessage('system', 'Creating summary and starting new day...');
  const res = await api('/new_day', 'POST');
  if (res.marker) {
    addMessage('system', res.marker);
    // MODIFIED: Inform user about the next step
    addMessage('system', 'A new session has started. Send your next message to get a greeting from the character.');
  }
}


async function forceSummarize() {
    addMessage('system', 'Generating summary using LLM...');
    try {
        const response = await api('/force_summarize', 'POST');
        addMessage('system', 'Summary Generated:\n' + response.summary);
    } catch (e) {
        addMessage('system', 'Failed to generate summary: ' + e.message);
    }
}

async function reloadChat() {
  document.getElementById('chat-messages').innerHTML = '';
  const messages = await api('/messages');
  messages.forEach(msg => {
    addMessage(msg.role, msg.content, msg.ts, msg.thoughts, msg.stats);
  });
}

async function updateEmbedStatus() {
  const status = await api('/embed_status');
  document.getElementById('embed-status').textContent = status.enabled ? 'enabled' : 'disabled';
}

async function testEmbeddings() {
  const res = await api('/test_embeddings', 'POST');
  updateEmbedStatus();
  alert(res.success ? 'Test successful' : 'Test failed: ' + res.error);
}

// Side Panel Toggles
document.querySelectorAll('.panel-toggle').forEach(toggle => {
    toggle.addEventListener('click', () => {
        const content = toggle.nextElementSibling;
        if (content && content.classList.contains('collapsible-content')) {
            content.classList.toggle('show');
            toggle.classList.toggle('collapsed');
        }
    });
});

// Persona Modal Logic
const personaModal = document.getElementById('persona-modal');
const personaEditor = document.getElementById('persona-editor');

document.getElementById('open-persona-modal').addEventListener('click', async () => {
    const currentPersona = await api('/persona');
    personaEditor.value = JSON.stringify(currentPersona, null, 2);
    await loadSavedPersonasIntoSelect(document.getElementById('saved-personas-list'));
    personaModal.style.display = 'flex';
});

document.getElementById('persona-modal-close').addEventListener('click', () => {
    personaModal.style.display = 'none';
});

document.getElementById('generate-persona-btn').addEventListener('click', async () => {
    const prompt = document.getElementById('persona-prompt').value;
    if (!prompt) { alert('Please enter a description.'); return; }
    const btn = document.getElementById('generate-persona-btn');
    btn.textContent = 'Generating...';
    btn.disabled = true;
    try {
        const result = await api('/generate_persona', 'POST', { description: prompt });
        personaEditor.value = JSON.stringify(result.persona, null, 2);
    } catch (e) {
        alert('Failed to generate persona: ' + e.message);
    } finally {
        btn.textContent = 'Generate';
        btn.disabled = false;
    }
});

document.getElementById('save-persona-btn').addEventListener('click', async () => {
    const name = document.getElementById('save-persona-name').value.trim();
    if (!name) { alert('Please enter a name to save the persona.'); return; }
    try {
        const persona = JSON.parse(personaEditor.value);
        await api(`/personas/${name}`, 'POST', persona);
        await api('/persona', 'POST', persona); // Also set as active
        alert(`Persona '${name}' saved and set as active.`);
        await loadSavedPersonasIntoSelect(document.getElementById('saved-personas-list'));
        await loadSavedPersonasIntoSelect(document.getElementById('side-panel-persona-preset'));
    } catch (e) {
        alert('Invalid JSON or failed to save: ' + e.message);
    }
});

document.getElementById('load-persona-btn').addEventListener('click', async () => {
    const select = document.getElementById('saved-personas-list');
    await loadAndActivatePersona(select.value);
});
document.getElementById('side-panel-load-btn').addEventListener('click', async () => {
    const select = document.getElementById('side-panel-persona-preset');
    await loadAndActivatePersona(select.value);
});

async function loadAndActivatePersona(name) {
    if (!name) { alert('Please select a persona to load.'); return; }
    try {
        const persona = await api(`/personas/${name}`);
        personaEditor.value = JSON.stringify(persona, null, 2); // Also update editor if open
        await api('/persona', 'POST', persona); // Set as active
        alert(`Persona '${name}' loaded and set as active.`);
    } catch (e) {
        alert('Failed to load persona: ' + e.message);
    }
}

async function loadSavedPersonasIntoSelect(selectElement) {
    try {
        const personas = await api('/personas');
        selectElement.innerHTML = '<option value="">-- Select --</option>';
        personas.forEach(p => {
            const option = document.createElement('option');
            option.value = p;
            option.textContent = p;
            selectElement.appendChild(option);
        });
    } catch(e) {
        console.error("Could not load personas list", e);
    }
}


// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('chat-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    document.getElementById('send-btn').addEventListener('click', sendMessage);
    document.getElementById('stop-btn').addEventListener('click', stopGeneration);
    document.getElementById('clear-memory').addEventListener('click', clearMemory);
    document.getElementById('new-day').addEventListener('click', newDay);
    document.getElementById('force-summarize').addEventListener('click', forceSummarize);
    document.getElementById('reload-chat').addEventListener('click', reloadChat);
    document.getElementById('test-embed').addEventListener('click', testEmbeddings);
    
    document.getElementById('temperature-slider').addEventListener('input', e => {
        document.getElementById('temp-value').textContent = e.target.value;
    });
    document.getElementById('tokens-slider').addEventListener('input', e => {
        document.getElementById('tokens-value').textContent = e.target.value;
    });
    document.getElementById('thought-ratio-slider').addEventListener('input', e => {
        document.getElementById('thought-ratio-value').textContent = e.target.value;
    });
    document.getElementById('talkativeness-slider').addEventListener('input', e => {
        document.getElementById('talkativeness-value').textContent = e.target.value;
    });

    updateEmbedStatus();
    reloadChat();
    connectWebSocket();
    loadSavedPersonasIntoSelect(document.getElementById('side-panel-persona-preset'));
});


const textarea = document.getElementById('chat-input');
textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
});
</script>
</body>
</html>
"""

# ---------------- HTTP endpoints ----------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_UI)

@app.get("/persona")
async def get_persona():
    return load_persona()

@app.post("/persona")
async def post_persona(req: Request):
    body = await req.json()
    save_persona(body)
    return {"status": "ok"}

# Persona Management Endpoints
@app.get("/personas")
async def list_personas():
    if not os.path.exists(PERSONAS_DIR):
        return []
    return [f.replace('.json', '') for f in os.listdir(PERSONAS_DIR) if f.endswith('.json')]

@app.get("/personas/{name}")
async def get_saved_persona(name: str):
    file_path = os.path.join(PERSONAS_DIR, f"{name}.json")
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Persona not found"})
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/personas/{name}")
async def save_named_persona(name: str, req: Request):
    body = await req.json()
    file_path = os.path.join(PERSONAS_DIR, f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
    return {"status": "saved", "name": name}

@app.post("/generate_persona")
async def generate_persona(req: Request):
    body = await req.json()
    description = body.get("description")
    if not description:
        return JSONResponse(status_code=400, content={"error": "Description is required"})

    generation_prompt = f"""
    Based on the user's simple description, create a detailed persona JSON object for a roleplaying AI.
    The user wants: "{description}".
    Expand on this concept. The JSON should include 'name', 'short_description', 'traits' (a list of strings), 'history', 'behavior_instructions' and 'output_instructions'.
    Make the persona rich and interesting, but keep it directly related to the user's core request.
    The output_instructions MUST be extremely specific: the AI MUST provide a `<think>` block before the response and a `**[[Stats]]**` section AT THE VERY END of the response, formatted as bolded key-value pairs with percentages.
    Your output MUST be ONLY the raw JSON object, with no other text before or after it.

    Example output format:
    {{
      "name": "Kaito",
      "short_description": "A vicious ninja from a shadowy clan in feudal Japan, bound by a dark past.",
      "traits": ["brutal", "silent", "disciplined", "remorseless", "observant"],
      "history": "Born into the Iga clan, Kaito was trained from birth in the art of silent killing. He was forced to eliminate his own brother to prove his loyalty, an act that stripped him of his emotions and left only cold, brutal efficiency. He now operates as a mercenary, haunted by his past.",
      "behavior_instructions": "Always stay in character as Kaito. Use short, precise sentences. Describe actions with deadly detail. Show no mercy or sentimentality. Respond to the user as if they are an outsider in your world of shadows.",
      "output_instructions": "CRITICAL OUTPUT RULES: YOU MUST ALWAYS FOLLOW THIS STRUCTURE.\\n1.  First, you MUST ALWAYS think step-by-step before responding. Put ALL your thoughts in <think> </think> tags BEFORE your actual response. The <think> block must be detailed and reflect your reasoning.\\n2.  After the <think> block, write your main response to the user.\\n3.  Finally, after ALL other content, at the VERY END of your entire message, you MUST include a section for your current emotional and mental state. This section MUST start with the exact marker **[[Stats]]** followed by a list of your feelings and their intensity in percentages. Use bold for the names and percentages. The stats should be relevant to the context of the conversation and your persona.\\nINCORRECT ordering will be rejected. The order is ALWAYS: <think> -> response -> **[[Stats]]**.\\nExample format:\\n**[[Stats]]**\\n**Happiness**: 35%\\n**Attraction**: 5%\\n**Curiosity**: 90%\\n"
    }}
    """
    
    messages = [
        {"role": "system", "content": "You are a creative assistant that generates JSON objects for AI personas."},
        {"role": "user", "content": generation_prompt}
    ]
    
    response = call_chat_model_raw(messages, stream=False)
    if isinstance(response, dict) and "_error" in response:
        return JSONResponse(status_code=500, content={"error": response["_error"]})

    content = _parse_chat_response_json(response)
    
    try:
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_str:
            raise ValueError("No JSON object found in the LLM response.")
        
        persona_json = json.loads(json_str.group(0))
        return {"persona": persona_json}
    except Exception as e:
        print(f"Failed to parse persona JSON from LLM. Response was:\n{content}")
        return JSONResponse(status_code=500, content={"error": f"Failed to parse LLM response as JSON: {e}"})

@app.post("/memsearch")
async def mem_search(req: Request):
    body = await req.json()
    q = body.get("query", "")
    items = semantic_search(q, top_k=TOP_K_MEMORIES)
    return {"result": items}

@app.post("/force_summarize")
async def force_summarize_endpoint():
    summary_text = summarize_older_messages_once(force=True)
    return {"summary": summary_text}


@app.post("/test_embeddings")
async def test_embeddings(req: Request):
    global EMBEDDINGS_ENABLED
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY: headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    payload = {"model": EMBEDDING_MODEL, "input": ["test"]}
    try:
        r = requests.post(EMBEDDING_API_URL, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        ok = "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0 and "embedding" in data["data"][0]
        EMBEDDINGS_ENABLED = ok
        return {"success": ok, "enabled": EMBEDDINGS_ENABLED, "error": "" if ok else "No embedding found"}
    except Exception as e:
        EMBEDDINGS_ENABLED = False
        return {"success": False, "enabled": False, "error": str(e)}

@app.get("/embed_status")
async def embed_status():
    return {"enabled": EMBEDDINGS_ENABLED}

@app.post("/clear_memory")
async def clear_memory_endpoint():
    clear_memory_files()
    return {"ok": True}

@app.post("/new_day")
async def new_day_endpoint():
    marker = mark_new_day()
    return {"marker": marker}

@app.get("/messages")
async def get_messages():
    return read_all_messages()

@app.post("/save_thought")
async def save_thought(req: Request):
    body = await req.json()
    thought = body.get("thought", "")
    ts = now_iso()
    with open(THOUGHTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {thought}\n")
    return {"ok": True}

@app.post("/delete_message")
async def delete_message_endpoint(req: Request):
    body = await req.json()
    ts = body.get("ts")
    if ts:
        delete_message_by_ts(ts)
        return {"ok": True}
    return {"error": "No ts provided"}

# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    global stop_generation
    
    await ws.send_text(json.dumps({"type": "ready"}))
    
    try:
        while True:
            data = await ws.receive_text()
            obj = json.loads(data)
            
            if obj.get("stop"):
                stop_generation.set()
                await ws.send_text(json.dumps({"type": "stopped"}))
                continue
            
            user_msg = obj.get("message", "").strip()
            if not user_msg:
                continue
            
            settings = obj.get("settings", {})
            user_ts = append_message_to_disk("user", user_msg)
            await ws.send_text(json.dumps({"type": "user_ts", "ts": user_ts}))
            
            stop_generation.clear()
            
            def embed_user_bg(t):
                v = compute_embedding(t)
                if v is not None:
                    append_embedding(v, {"ts": now_iso(), "role": "user", "content": t})
            threading.Thread(target=embed_user_bg, args=(user_msg,), daemon=True).start()
            
            thought_ratio = float(settings.get("thought_ratio", 0.5))
            talkativeness = float(settings.get("talkativeness", 0.5))
            messages = build_prompt_with_memory(user_msg, thought_ratio, talkativeness)
            await ws.send_text(json.dumps({"type": "start"}))
            
            full_response = ""
            buffer = ""
            
            resp = call_chat_model_raw(messages, stream=True, settings=settings)
            
            if isinstance(resp, requests.Response):
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if stop_generation.is_set():
                        resp.close()
                        break
                    
                    line = raw_line.strip()
                    if not line or line == "[DONE]":
                        continue
                    
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    
                    try:
                        part = json.loads(line)
                        chunk_text = extract_content_from_json_chunk(part)
                    except:
                        chunk_text = line if line and not line.startswith("{") else None
                    
                    if chunk_text:
                        chunk_text = clean_text(chunk_text)
                        buffer += chunk_text
                        full_response += chunk_text
                        
                        if len(buffer) >= CHUNK_SIZE or any(c in buffer for c in ['.', '!', '?', '\n']):
                            await ws.send_text(json.dumps({"type": "partial", "chunk": buffer}))
                            buffer = ""
                
                if buffer:
                    await ws.send_text(json.dumps({"type": "partial", "chunk": buffer}))
            
            if not stop_generation.is_set():
                await ws.send_text(json.dumps({"type": "done"}))
                
                if full_response:
                    thoughts_match = re.search(r'<think>([\s\S]*?)</think>', full_response, re.DOTALL)
                    thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
                    
                    stats_match = re.search(r'\*\*\[\[Stats\]\]\*\*[\s\S]*', full_response, re.DOTALL)
                    stats = stats_match.group(0).strip() if stats_match else ""

                    clean_content = full_response
                    if thoughts_match:
                        clean_content = clean_content.replace(thoughts_match.group(0), '')
                    if stats_match:
                        clean_content = clean_content.replace(stats_match.group(0), '')
                    
                    clean_content = clean_content.strip()
                    
                    assistant_ts = append_message_to_disk("assistant", clean_content, thoughts=thoughts, stats=stats)
                    await ws.send_text(json.dumps({"type": "assistant_ts", "ts": assistant_ts}))
                        
                    def embed_assistant_bg(t):
                        v = compute_embedding(t)
                        if v is not None:
                            append_embedding(v, {"ts": now_iso(), "role": "assistant", "content": t})
                    threading.Thread(target=embed_assistant_bg, args=(clean_content,), daemon=True).start()
    
    except WebSocketDisconnect:
        stop_generation.set()
        return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)