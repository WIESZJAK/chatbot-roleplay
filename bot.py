# bot.py (FULL FEATURES RESTORED)
"""
Advanced roleplay server - Cleaned & Refactored with Summary & New Day Logic
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, json, threading, requests, re, shutil, asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import uvicorn

# --- IMPORTS ---
from config import (
    APP_DIR, CHATS_DIR, STATIC_DIR, PERSONAS_DIR, DEFAULT_CHAT_ID,
    MODEL_API_URL, MODEL_API_KEY, EMBEDDING_API_URL, EMBEDDING_MODEL,
    TEXT_MODEL_NAME, RECENT_MSGS, TOP_K_MEMORIES, PERSISTENT_STATS_ENABLED,
    DEFAULT_PERSONA, DEFAULT_SETTINGS  # <--- DODAJ TO TUTAJ (pamiętaj o przecinku przed)
)
import storage
from response_parser import parse_full_response

# --- OpenAI client helper ---
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Init app
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# NOWE: Udostępniamy folder personas/img pod adresem /avatars
app.mount("/avatars", StaticFiles(directory=os.path.join(PERSONAS_DIR, "img")), name="avatars")

# Global state
stop_generation = threading.Event()
FAISS_INDEX_CACHE = {} 
EMBEDDINGS_ENABLED = True 

# --- Clients ---
def get_openai_client():
    if not HAS_OPENAI: return None
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    try:
        return OpenAI(base_url=base_url, api_key=MODEL_API_KEY)
    except Exception: return None

def get_async_openai_client():
    if not HAS_OPENAI: return None
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    try:
        return AsyncOpenAI(base_url=base_url, api_key=MODEL_API_KEY)
    except Exception: return None

# --- Helper Wrappers ---
def get_safe_chat_id(chat_id: str):
    safe_chat_id = re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)
    return safe_chat_id or DEFAULT_CHAT_ID

# --- Embeddings Logic ---
def get_all_embeddings(chat_id: str):
    embeddings_npy = storage.get_chat_file_path(chat_id, "embeddings.npy")
    embeddings_meta = storage.get_chat_file_path(chat_id, "embeddings_meta.jsonl")
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
    embeddings_npy = storage.get_chat_file_path(chat_id, "embeddings.npy")
    if not os.path.exists(embeddings_npy):
        np.save(embeddings_npy, vec)
    else:
        try:
            old = np.load(embeddings_npy)
            combined = np.vstack([old, vec]) if old.size > 0 else vec
            np.save(embeddings_npy, combined)
        except Exception:
            np.save(embeddings_npy, vec)
    with open(storage.get_chat_file_path(chat_id, "embeddings_meta.jsonl"), "a", encoding="utf-8") as f:
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
            np.save(storage.get_chat_file_path(chat_id, "embeddings.npy"), embs)
            with open(storage.get_chat_file_path(chat_id, "embeddings_meta.jsonl"), "w", encoding="utf-8") as f:
                for m in meta: f.write(json.dumps(m, ensure_ascii=False) + '\n')

def compute_embedding(text: str, model_name: Optional[str] = None) -> Optional[np.ndarray]:
    if not text or not text.strip(): return None
    
    # Czyszczenie tekstu
    clean_text = re.sub(r'<br\s*/?>', '\n', text)
    clean_text = re.sub(r'<[^>]+>', '', clean_text).strip()
    if not clean_text: return None
    
    model_to_use = model_name or EMBEDDING_MODEL
    
    # ZMIANA: Używamy WYŁĄCZNIE requests z twardym timeoutem.
    # Pomijamy klienta OpenAI tutaj, bo potrafi blokować wątek przy błędach sieci.
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY: headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    
    try:
        # Timeout 5 sekund - jeśli LM Studio "zamuli", bot nie będzie czekał wiecznie.
        r = requests.post(
            EMBEDDING_API_URL, 
            json={"model": model_to_use, "input": [clean_text]}, 
            headers=headers, 
            timeout=5
        )
        r.raise_for_status()
        return np.array(r.json()['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Embedding error (Skipping memory for this turn): {e}")
        return None

def _start_embedding_thread(chat_id: str, text_to_embed: str, meta: dict, settings: dict):
    def _job():
        try:
            role = meta.get("role", "unknown")
            vec = compute_embedding(f"{role}: {text_to_embed}", model_name=settings.get("embedding_model"))
            if vec is not None: append_embedding(chat_id, vec, meta)
        except Exception: pass
    t = threading.Thread(target=_job, daemon=True)
    t.start()

def semantic_search(chat_id: str, query: str, top_k: int = TOP_K_MEMORIES, settings: Dict = {}) -> List[Dict[str,Any]]:
    if not settings.get("enable_memory", True): return []
    try:
        qv = compute_embedding(f"user: {query}", model_name=settings.get("embedding_model"))
    except: return []
    if qv is None: return []
    embs, meta = get_all_embeddings(chat_id)
    if embs.size == 0: return []
    qv_norm = qv / (np.linalg.norm(qv) + 1e-12)
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = (embs_norm @ qv_norm).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    return [meta[int(i)] for i in top_idx]

# --- Core Logic (UPDATED) ---

def generate_summary(chat_id: str) -> str:
    """Reads recent messages and uses LLM to generate a summary."""
    messages = storage.read_all_messages(chat_id)
    if not messages: return "No messages to summarize."
    
    # Używamy ostatnich 20 wiadomości dla kontekstu (około 10 wymian)
    recent = messages[-20:] 
    conversation_text = ""
    for m in recent:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        conversation_text += f"[{role}]: {content}\n"
        
    if not conversation_text.strip(): return "Not enough content."

    sys_prompt = "You are a summarizer. Summarize the key events, facts, and emotional state of the following conversation in 3-4 factual sentences."
    
    client = get_openai_client()
    if not client: return "Summarization unavailable (No OpenAI client)."
    
    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL_NAME, 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": conversation_text}
            ],
            max_tokens=200
        )
        summary = resp.choices[0].message.content
        
        # Zapisz podsumowanie na dysk
        current_date = datetime.now().strftime("%Y-%m-%d")
        storage.append_summary(chat_id, summary, current_date)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def mark_new_day(chat_id: str):
    """Generates summary, saves 'new day' marker, and creates 'pending' state file."""
    # 1. Generuj Podsumowanie z poprzedniej sesji
    summary = generate_summary(chat_id)
    
    # 2. Zapisz wizualny separator w wiadomościach
    now = datetime.now()
    marker = f"-- NEW DAY: {now.strftime('%Y-%m-%d %H:%M:%S')} --"
    storage.append_message_to_disk(chat_id, "system", marker)
    
    # 3. Utwórz plik stanu 'new_day_state.json'
    # Ten plik flaguje, że następna odpowiedź bota powinna być powitaniem.
    state_file = storage.get_chat_file_path(chat_id, "new_day_state.json")
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"status": "pending", "timestamp": now.isoformat()}, f)
    except Exception as e:
        return f"Error saving new day state: {str(e)}", summary # Zwróć błąd, jeśli zapis się nie uda
        
    return marker, summary

def update_message_by_ts(chat_id: str, ts: str, raw_content: str):
    messages = storage.read_all_messages(chat_id)
    updated = False
    parsed_data = parse_full_response(raw_content)
    for msg in messages:
        if msg.get("ts") == ts:
            msg.update(parsed_data)
            updated = True
            break
    if updated:
        storage.save_all_messages(chat_id, messages)
        delete_embedding_by_ts(chat_id, ts)
        vec = compute_embedding(parsed_data["content"], model_name=None)
        if vec is not None:
            append_embedding(chat_id, vec, {"ts": ts, "role": "assistant", "content": parsed_data["content"]})
    return updated

def build_prompt_context(chat_id: str, user_msg: str, settings: Dict[str, Any]) -> List[Dict[str,str]]:
    persona = storage.load_persona(chat_id)
    summary = storage.load_summary(chat_id)
    
    # 1. System Prompt
    sys = [f"You are: {persona.get('name')}. {persona.get('short_description')}"]
    if persona.get("traits"): sys.append(f"Traits: {', '.join(persona['traits'])}")
    if persona.get("history"): sys.append(f"History: {persona['history']}")
    if persona.get("behavior_instructions"): sys.append(str(persona['behavior_instructions']))
    
    # --- NEW DAY LOGIC ---
    # Check if we are in a "New Day" state
    state_file = storage.get_chat_file_path(chat_id, "new_day_state.json")
    is_new_day = False
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                if state.get("status") == "pending":
                    is_new_day = True
                    sys.append(f"*** IMPORTANT: A NEW DAY HAS BEGUN. Current time: {datetime.now().strftime('%H:%M')}. ***")
                    sys.append("Greet the user as if seeing them for the first time today. Comment on the time of day or how long it's been.")
        except: pass

    if settings.get("persistent_stats", False):
        stats = storage.load_emotional_state(chat_id)
        stats_str = "\n".join([f"**{k}**: {v}" for k,v in stats.items()]) if stats else "Establish initial stats."
        sys.append(f"Current emotional state:\n{stats_str}\nUpdate this in your response.")

    if summary: sys.append(f"Memory summary:\n{summary}")
    
    sys.append(persona.get("output_instructions", DEFAULT_PERSONA["output_instructions"]))
    
    messages = [{"role": "system", "content": "\n\n".join(sys)}]
    
    # 2. Memories
    mems = semantic_search(chat_id, user_msg, TOP_K_MEMORIES, settings)
    if mems:
        mem_lines = []
        for m in mems:
            role = m.get('role', '?')
            content = m.get('content', '')
            if role == 'assistant':
                extra = []
                if m.get('thoughts'): extra.append(f"Thought: {m['thoughts']}")
                if m.get('stats'): extra.append(f"Stats: {m['stats']}")
                if extra:
                    mem_lines.append(f"[{role}] (Internal: {'; '.join(extra)})\nSaid: {content}")
                else:
                    mem_lines.append(f"[{role}]: {content}")
            else:
                mem_lines.append(f"[{role}]: {content}")
        
        messages.append({"role": "system", "content": "Relevant memories:\n" + "\n---\n".join(mem_lines)})
        
    # 3. History Reconstruction
    history = storage.read_last_messages(chat_id, RECENT_MSGS)
    for m in history:
        role = m.get("role")
        if role == "user":
            messages.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            thoughts = m.get("thoughts") or "..."
            content = m.get("content") or "..."
            stats = m.get("stats") or "Status: Unknown"
            final = m.get("final_thoughts") or "..."
            
            reconstructed_msg = (
                f"<think>\n{thoughts}\n</think>\n\n"
                f"**[[Response]]**\n{content}\n\n"
                f"**[[Stats]]**\n{stats}\n\n"
                f"**[[Final Thoughts]]**\n{final}"
            )
            messages.append({"role": "assistant", "content": reconstructed_msg})
        elif role == "system":
             messages.append({"role": "system", "content": m.get("content", "")})
        
    if not history or history[-1].get("content") != user_msg:
        messages.append({"role": "user", "content": user_msg})
    
    # 4. Enforcement
    messages.append({
        "role": "system", 
        "content": "SYSTEM REMINDER: You MUST use the required format:\n<think>...</think>\n**[[Response]]**\n...\n**[[Stats]]**\n...\n**[[Final Thoughts]]**\n..."
    })
        
    return messages

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read().replace('__PERSISTENT_STATS_ENABLED__', "true" if PERSISTENT_STATS_ENABLED else "false")

@app.get("/chats")
async def list_chats():
    if not os.path.exists(CHATS_DIR): return []
    return sorted([d for d in os.listdir(CHATS_DIR) if os.path.isdir(os.path.join(CHATS_DIR, d))])

@app.post("/chats/create")
async def create_chat(req: Request):
    body = await req.json()
    safe_name = get_safe_chat_id(body.get("name", ""))
    if not safe_name: return JSONResponse({"error": "Invalid name"}, 400)
    storage.get_chat_dir(safe_name)
    storage.save_persona(safe_name, DEFAULT_PERSONA)
    # NOWE: Zapisz domyślne ustawienia dla nowego czatu
    storage.save_chat_settings(safe_name, DEFAULT_SETTINGS) 
    return {"status": "ok", "chat_id": safe_name}

@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(chat_id: str):
    safe_id = get_safe_chat_id(chat_id)
    path = storage.get_chat_dir(safe_id)
    if os.path.exists(path): shutil.rmtree(path)
    return {"status": "ok"}

@app.get("/{chat_id}/messages")
async def get_messages(chat_id: str):
    return storage.read_all_messages(get_safe_chat_id(chat_id))

@app.get("/{chat_id}/settings")
async def get_chat_settings(chat_id: str):
    return storage.load_chat_settings(get_safe_chat_id(chat_id))

@app.post("/{chat_id}/settings")
async def save_chat_settings_endpoint(chat_id: str, req: Request):
    body = await req.json()
    storage.save_chat_settings(get_safe_chat_id(chat_id), body)
    return {"status": "ok"}

@app.get("/{chat_id}/persona")
async def get_persona(chat_id: str):
    return storage.load_persona(get_safe_chat_id(chat_id))

@app.post("/{chat_id}/persona")
async def post_persona(chat_id: str, req: Request):
    storage.save_persona(get_safe_chat_id(chat_id), await req.json())
    return {"status": "ok"}

@app.post("/{chat_id}/clear_memory")
async def clear_memory_endpoint(chat_id: str):
    safe_id = get_safe_chat_id(chat_id)
    shutil.rmtree(storage.get_chat_dir(safe_id))
    storage.save_persona(safe_id, DEFAULT_PERSONA)
    return {"status": "ok"}

# --- NEW DAY & SUMMARY ENDPOINTS ---

@app.post("/{chat_id}/new_day")
async def new_day_endpoint(chat_id: str):
    marker, summary = mark_new_day(get_safe_chat_id(chat_id))
    return {"marker": marker, "summary": summary}

@app.post("/{chat_id}/force_summarize")
async def force_summarize_endpoint(chat_id: str):
    summary = generate_summary(get_safe_chat_id(chat_id))
    return {"summary": summary}

@app.get("/{chat_id}/last_summary")
async def get_last_summary(chat_id: str):
    summary = storage.load_summary(get_safe_chat_id(chat_id))
    return {"summary": summary}

@app.post("/{chat_id}/delete_message")
async def delete_message_endpoint(chat_id: str, req: Request):
    body = await req.json()
    ts = body.get("ts")
    if ts:
        storage.delete_message_by_ts(get_safe_chat_id(chat_id), ts)
    return {"ok": True}

@app.post("/{chat_id}/edit_message")
async def edit_message_endpoint(chat_id: str, req: Request):
    body = await req.json()
    if update_message_by_ts(get_safe_chat_id(chat_id), body.get("ts"), body.get("raw_content")):
        messages = storage.read_all_messages(get_safe_chat_id(chat_id))
        msg = next((m for m in messages if m["ts"] == body.get("ts")), {})
        return {"ok": True, "updated_message": msg}
    return JSONResponse({"error": "Not found"}, 404)

@app.get("/system_info")
async def system_info():
    return {"version": "0.9.5-BETA", "model_name": TEXT_MODEL_NAME}

@app.get("/models")
async def get_models():
    try:
        r = requests.get(f"{os.getenv('LMSTUDIO_BASE_URL', 'http://127.0.0.1:1234/v1')}/models")
        return {"models": [m['id'] for m in r.json()['data']]}
    except: return {"models": []}

@app.post("/test_text_model")
async def test_model(req: Request):
    return {"success": True}

@app.get("/personas")
async def list_personas():
    if not os.path.exists(PERSONAS_DIR): return []
    return [f.replace('.json','') for f in os.listdir(PERSONAS_DIR) if f.endswith('.json')]

@app.get("/personas/{name}")
async def get_saved_persona(name: str):
    # Zabezpiecz nazwę
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    # Dodaj rozszerzenie .json, jeśli go nie ma w nazwie pliku na dysku
    file_path = os.path.join(PERSONAS_DIR, f"{safe_name}.json")
    
    if not os.path.exists(file_path):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)

@app.post("/generate_persona")
async def generate_persona(req: Request):
    description = (await req.json()).get("description")
    if not description: return JSONResponse({"error": "No description"}, 400)
    return {"persona": DEFAULT_PERSONA}

@app.post("/test_embeddings")
async def test_embeddings(req: Request):
    return {"success": True}

@app.post("/{chat_id}/inject_event")
async def inject_event_endpoint(chat_id: str, req: Request):
    safe_id = get_safe_chat_id(chat_id)
    body = await req.json()
    event_text = body.get("event", "")
    storage.append_message_to_disk(safe_id, "system", f"[WORLD EVENT]: {event_text}")
    return {"status": "ok"}

# --- WebSocket ---
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    chat_id = DEFAULT_CHAT_ID
    
    async def process_stream(messages, settings):
        stop_generation.clear()
        await ws.send_json({"type": "start"})
        
        client = get_async_openai_client()
        full_resp = ""
        
        try:
            stream = await client.chat.completions.create(
                model=settings.get("model", TEXT_MODEL_NAME),
                messages=messages,
                temperature=settings.get("temperature", 1.0),
                max_tokens=settings.get("max_tokens", 1024),
                stream=True
            )
            async for chunk in stream:
                if stop_generation.is_set(): break
                if chunk.choices[0].delta.content:
                    txt = chunk.choices[0].delta.content
                    full_resp += txt
                    await ws.send_json({"type": "partial", "chunk": txt})
                    
            await ws.send_json({"type": "done"})
            
            if full_resp:
                parsed = parse_full_response(full_resp)
                
                ts = storage.append_message_to_disk(
                    chat_id, "assistant", 
                    parsed["content"], 
                    thoughts=parsed["thoughts"],
                    stats=parsed["stats"], 
                    final_thoughts=parsed["final_thoughts"]
                )
                
                if settings.get("enable_memory"):
                    # Rich embedding with memory context
                    rich_text_to_embed = (
                        f"Thought: {parsed['thoughts']}\n"
                        f"Response: {parsed['content']}\n"
                        f"Stats: {parsed['stats']}\n"
                        f"Final Thoughts: {parsed['final_thoughts']}"
                    )
                    meta = {
                        "ts": ts, 
                        "role": "assistant", 
                        "content": parsed["content"],
                        "thoughts": parsed["thoughts"],
                        "stats": parsed["stats"],
                        "final_thoughts": parsed["final_thoughts"]
                    }
                    _start_embedding_thread(chat_id, rich_text_to_embed, meta, settings)
                
                # Check if we need to close the "New Day" state
                state_file = storage.get_chat_file_path(chat_id, "new_day_state.json")
                if os.path.exists(state_file):
                    try:
                        with open(state_file, "r") as f:
                            state = json.load(f)
                        if state.get("status") == "pending":
                            # Mark as completed
                            with open(state_file, "w") as f:
                                json.dump({"status": "completed", "timestamp": state["timestamp"]}, f)
                    except: pass

                await ws.send_json({"type": "assistant_ts", "ts": ts})
                
        except Exception as e:
            print(f"Stream error: {e}")
            await ws.send_json({"type": "error", "message": str(e)})

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "init":
                chat_id = get_safe_chat_id(data.get("chat_id", DEFAULT_CHAT_ID))
                await ws.send_json({"type": "chat_switched", "chat_id": chat_id})
            
            elif msg_type == "stop":
                stop_generation.set()
            
            elif msg_type == "message":
                content = data.get("message", "")
                ts = storage.append_message_to_disk(chat_id, "user", content)
                await ws.send_json({"type": "user_ts", "ts": ts})
                
                if data.get("settings", {}).get("enable_memory"):
                    _start_embedding_thread(chat_id, content, {"ts": ts, "role": "user", "content": content}, data.get("settings"))
                
                msgs = build_prompt_context(chat_id, content, data.get("settings", {}))
                await process_stream(msgs, data.get("settings", {}))
                
            elif msg_type == "regenerate":
                ts = data.get("ts")
                all_msgs = storage.read_all_messages(chat_id)
                user_content = ""
                for i, m in enumerate(all_msgs):
                    if m["ts"] == ts:
                        if i > 0 and all_msgs[i-1]["role"] == "user":
                            user_content = all_msgs[i-1]["content"]
                        break
                if user_content:
                    storage.delete_message_by_ts(chat_id, ts)
                    msgs = build_prompt_context(chat_id, user_content, data.get("settings", {}))
                    await process_stream(msgs, data.get("settings", {}))
                
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)