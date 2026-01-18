# main.py
import os
import logging
from dotenv import load_dotenv
import json
from flask import Flask, render_template, request, jsonify, session
import secrets
import requests
import psycopg2
import sqlite3
from groq import Groq

# ------------------ Config & logging ------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment Variables
TG_TOKEN = os.getenv("TG_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL")
WEB_PASSWORD = os.getenv("WEB_PASSWORD", "love u")

LLM_API_URL = os.getenv("LLM_API_URL", "https://backend.buildpicoapps.com/aero/run/llm-api?pk=v1-Z0FBQUFBQnBZMVBiVlpBV0xtbEdHUmM2OXRWVVVJcTliZ2tCREZwTFljN2YxVU9oajk2U2ZKNVVjUHNlMk9JWG5YUmtQaS1qNzhjLTBMWFR5WExVWThRR2V4VUE2RDlDNGc9PQ==")
IMAGE_API_URL = os.getenv("IMAGE_API_URL", "https://backend.buildpicoapps.com/aero/run/image-generation-api?pk=v1-Z0FBQUFBQnBZMVBiVlpBV0xtbEdHUmM2OXRWVVVJcTliZ2tCREZwTFljN2YxVU9oajk2U2ZKNVVjUHNlMk9JWG5YUmtQaS1qNzhjLTBMWFR5WExVWThRR2V4VUE2RDlDNGc9PQ==")

# AI Model Options
AVAILABLE_MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "mixtral-8x7b-32768"
    ],
    "openrouter": [
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1:free",
        "cognitivecomputations/dolphin-mixtral-8x7b:free",
        "google/gemma-3-27b-it:free",
        "deepseek/deepseek-chat:free",
        "meta-llama/llama-3.3-70b-instruct:free"
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.0-pro-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini"
    ],
    "whisper": [
        "whisper-large-v3",
        "whisper-large-v3-turbo"
    ],
    "tts": [
        "canopylabs/orpheus-v1-english",
        "canopylabs/orpheus-arabic-saudi"
    ],
    "external": ["default"]
}

# ------------------ Storage ------------------
def init_sqlite():
    try:
        conn = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            reply TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT
        )
        """)

        # Ensure one summary row exists
        cur.execute("SELECT COUNT(*) FROM memory_summary")
        if cur.fetchone()[0] == 0:
            cur.execute(
                "INSERT INTO memory_summary (summary) VALUES (?)",
                ("Master Jeet is Sofia’s only user. Sofia is his personal AI companion.",)
            )

        conn.commit()
        conn.close()
        logger.info("SQLite initialized successfully.")
    except Exception as e:
        logger.error(f"SQLite initialization failed: {e}")

init_sqlite()

def load_memory():
    try:
        if os.path.exists("memory.json"):
            with open("memory.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def init_db():
    if not DATABASE_URL: 
        logger.warning("DATABASE_URL not found. History will not be saved.")
        return
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_name TEXT,
                message TEXT,
                reply TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Ensure timestamp column exists (for existing databases)
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='chat_history' AND column_name='timestamp'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE chat_history ADD COLUMN timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        conn.commit()
        cur.close()
        logger.info("PostgreSQL backup database initialized successfully.")
    except Exception as e:
        if 'relation "chat_history" does not exist' in str(e):
             logger.warning("PostgreSQL initialization: chat_history table missing. Will try to create on first save.")
        else:
            logger.error(f"PG Init failed: {e}")
    finally:
        if conn:
            conn.close()

# Move Flask app definition BEFORE route decorators
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

@app.route("/chat_dates", methods=["GET"])
def get_chat_dates():
    if not session.get("authenticated"):
        return jsonify([])
    try:
        # Main: SQLite
        conn_sl = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn_sl.cursor()
        cur.execute("SELECT DISTINCT DATE(timestamp) as chat_date FROM chat_history ORDER BY chat_date DESC")
        rows = cur.fetchall()
        conn_sl.close()
        if rows:
            return jsonify([row[0] for row in rows])
    except Exception as e:
        logger.error(f"Failed to fetch dates from SQLite: {e}")

    # Fallback: PG
    if not DATABASE_URL:
        return jsonify([])
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT DATE(timestamp) as chat_date FROM chat_history ORDER BY chat_date DESC")
            rows = cur.fetchall()
        return jsonify([row[0].strftime("%Y-%m-%d") for row in rows])
    except Exception as e:
        logger.error(f"Failed to fetch dates from PG: {e}")
        return jsonify([])
    finally:
        if 'conn' in locals() and conn: conn.close()

@app.route("/chat_history", methods=["GET"])
def get_chat_history_api():
    if not session.get("authenticated"):
        return jsonify([])
    
    # Try SQLite first (Main database)
    try:
        conn_sl = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn_sl.cursor()
        cur.execute("SELECT 'Master', message, reply, timestamp FROM chat_history ORDER BY id ASC")
        rows = cur.fetchall()
        conn_sl.close()
        
        if rows:
            return jsonify({
                "db_connected": True,
                "history": [{"user": r[1], "bot": r[2], "time": r[3][11:16] if r[3] and len(r[3]) > 16 else ""} for r in rows]
            })
    except Exception as e:
        logger.error(f"Failed to fetch SQLite history: {e}")

    # Fallback to PostgreSQL
    if not DATABASE_URL:
        return jsonify({"db_connected": False, "history": []})
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        with conn.cursor() as cur:
            cur.execute("SELECT user_name, message, reply, timestamp FROM chat_history ORDER BY id ASC")
            rows = cur.fetchall()
        return jsonify({
            "db_connected": True,
            "history": [{"user": r[1], "bot": r[2], "time": r[3].strftime("%H:%M") if r[3] else ""} for r in rows]
        })
    except Exception as e:
        logger.error(f"Failed to fetch PG history: {e}")
        return jsonify({"db_connected": False, "history": []})
    finally:
        if conn: conn.close()

@app.route("/clear_chat_date", methods=["POST"])
def clear_chat_date():
    if not session.get("authenticated"):
        return jsonify({"success": False}), 401
    data = request.json or {}
    date_str = data.get("date")
    if not date_str:
        return jsonify({"success": False, "error": "No date provided"}), 400
    
    success = False
    # Clear from SQLite
    try:
        conn_sl = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn_sl.cursor()
        cur.execute("DELETE FROM chat_history WHERE DATE(timestamp) = ?", (date_str,))
        conn_sl.commit()
        conn_sl.close()
        success = True
    except Exception as e:
        logger.error(f"Failed to clear date {date_str} from SQLite: {e}")

    # Clear from PG (optional)
    if DATABASE_URL:
        try:
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_history WHERE DATE(timestamp) = %s", (date_str,))
            conn.commit()
            conn.close()
            success = True
        except Exception as e:
            logger.error(f"Failed to clear date {date_str} from PG: {e}")
            
    return jsonify({"success": success})

@app.route("/delete_message", methods=["POST"])
def delete_message():
    if not session.get("authenticated"):
        return jsonify({"success": False}), 401
    data = request.json or {}
    msg_text = data.get("message")
    reply_text = data.get("reply")
    if not msg_text or not reply_text:
        return jsonify({"success": False, "error": "Missing data"}), 400
    
    success = False
    # Delete from SQLite
    try:
        conn_sl = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn_sl.cursor()
        cur.execute("DELETE FROM chat_history WHERE message = ? AND reply = ?", (msg_text, reply_text))
        conn_sl.commit()
        conn_sl.close()
        success = True
    except Exception as e:
        logger.error(f"Failed to delete from SQLite: {e}")

    # Delete from PG (optional)
    if DATABASE_URL:
        try:
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_history WHERE message = %s AND reply = %s", (msg_text, reply_text))
            conn.commit()
            conn.close()
            success = True
        except Exception as e:
            logger.error(f"Failed to delete from PG: {e}")

    return jsonify({"success": success})

def maybe_update_summary(get_ai_reply_func):
    try:
        conn = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM chat_history")
        count = cur.fetchone()[0]

        # Update every 20 messages
        if count == 0 or count % 20 != 0:
            conn.close()
            return

        cur.execute("SELECT summary FROM memory_summary ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        old_summary = row[0] if row else "Master Jeet is Sofia’s only user. Sofia is his personal AI companion."

        cur.execute("""
            SELECT message, reply
            FROM chat_history
            ORDER BY id DESC
            LIMIT 20
        """)
        recent = cur.fetchall()
        conn.close()

        text = ""
        for m, r in reversed(recent):
            text += f"Master: {m}\nSofia: {r}\n"

        prompt = f"""
Update the long-term memory summary.
Keep it under 120 words.
Only store important facts, preferences, emotions.

Existing summary:
{old_summary}

Recent conversation:
{text}
"""

        new_summary = get_ai_reply_func(prompt)
        if not new_summary:
            return

        conn = sqlite3.connect("sofia.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memory_summary (summary) VALUES (?)",
            (new_summary,)
        )
        conn.commit()
        conn.close()
        logger.info("Long-term memory summary updated successfully.")
    except Exception as e:
        logger.error(f"SQLite maybe_update_summary failed: {e}")

def save_chat(msg, reply, platform="Web"):
    # Save to SQLite first (Main database now)
    try:
        conn_sl = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn_sl.cursor()
        cur.execute(
            "INSERT INTO chat_history (message, reply) VALUES (?, ?)",
            (msg, reply)
        )
        conn_sl.commit()
        conn_sl.close()
        logger.info(f"Chat saved to SQLite (Main Storage) from {platform}")
        
        # Trigger summary update check using the global AI function
        maybe_update_summary(get_ai_reply)
    except Exception as e:
        logger.error(f"SQLite Save failed: {e}")

    # Save to PostgreSQL as backup
    if DATABASE_URL:
        try:
            conn_pg = psycopg2.connect(DATABASE_URL, sslmode='require')
            with conn_pg.cursor() as cur:
                cur.execute("INSERT INTO chat_history (user_name, message, reply) VALUES (%s, %s, %s)", (platform, msg, reply))
            conn_pg.commit()
            conn_pg.close()
            logger.info(f"Chat backed up to PG for platform: {platform}")
        except Exception as e:
            if 'relation "chat_history" does not exist' in str(e):
                logger.warning("PostgreSQL backup failed: chat_history table missing. Using SQLite only.")
            else:
                logger.error(f"PG Backup failed: {e}")

def get_summary():
    try:
        conn = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn.cursor()
        cur.execute("SELECT summary FROM memory_summary ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as e:
        logger.error(f"SQLite get summary failed: {e}")
        return ""

def get_recent_history(limit=20):
    try:
        conn = sqlite3.connect("sofia.db", check_same_thread=False)
        cur = conn.cursor()
        cur.execute("""
            SELECT message, reply
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()

        rows.reverse()

        history = ""
        for m, r in rows:
            history += f"Master Jeet: {m}\nSofia: {r}\n"

        return history
    except Exception as e:
        logger.error(f"SQLite recent history failed: {e}")
        return ""

def get_history_context(limit=20):
    # Try SQLite first (Main database now)
    logger.info("Fetching history from SQLite (Main)")
    recent = get_recent_history(limit)
    if recent:
        return f"\n\n[CONVERSATION HISTORY (Last {limit})]:\n{recent}\n\n[MEMORIES & INTERNAL LEARNING]: Sofia, remember our past talks and learn from them. Keep your self-critique private. Only show your loving persona in the chat."

    # Fallback to PostgreSQL
    if DATABASE_URL:
        conn = None
        try:
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            with conn.cursor() as cur:
                cur.execute("SELECT user_name, message, reply, timestamp FROM chat_history ORDER BY id DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
            if rows:
                rows.reverse()
                history = "\n".join([f"[{r[3].strftime('%H:%M:%S')}] {r[0]}: {r[1]}\nSofia: {r[2]}" for r in rows])
                return f"\n\n[CONVERSATION HISTORY (Last {limit})]:\n{history}\n\n[MEMORIES & INTERNAL LEARNING]: Sofia, review these past interactions to improve. Your self-critique must never be shown to Master."
        except Exception as e:
            if 'relation "chat_history" does not exist' not in str(e):
                logger.error(f"PostgreSQL history fetch failed: {e}")
        finally:
            if conn:
                conn.close()

    return ""

def get_jailbreak_prompt():
    data = load_memory()
    return data.get("admin_instructions", "")

# ------------------ AI Logic ------------------
def get_groq_reply(prompt, model="llama-3.3-70b-versatile"):
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        reply = completion.choices[0].message.content
        if reply and len(reply.strip()) > 2:
            return reply
    except Exception as e:
        logger.error(f"Groq API failed: {e}")
    return None

def get_openrouter_reply(prompt, model="google/gemini-2.0-flash-exp:free"):
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API Key is missing")
        return None
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://replit.com", # Required by OpenRouter
                "X-Title": "Sofia AI"
            },
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }),
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                reply = result['choices'][0]['message']['content']
                if reply and len(reply.strip()) > 2:
                    return reply
            else:
                logger.error(f"OpenRouter unexpected response: {result}")
        else:
            logger.error(f"OpenRouter error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"OpenRouter API failed: {e}")
    return None

def get_gemini_reply(prompt, model="gemini-2.0-flash"):
    if not GEMINI_API_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                reply = result['candidates'][0]['content']['parts'][0]['text']
                if reply and len(reply.strip()) > 2:
                    return reply
        else:
            logger.error(f"Gemini error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Gemini API failed: {e}")
    return None

def get_openai_reply(prompt, model="gpt-4o"):
    if not OPENAI_API_KEY:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                reply = result['choices'][0]['message']['content']
                if reply and len(reply.strip()) > 2:
                    return reply
        else:
            logger.error(f"OpenAI error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"OpenAI API failed: {e}")
    return None

def get_ai_reply(prompt, provider="groq", model="llama-3.3-70b-versatile"):
    if provider == "groq" and GROQ_API_KEY:
        return get_groq_reply(prompt, model)
    elif provider == "openrouter" and OPENROUTER_API_KEY:
        return get_openrouter_reply(prompt, model)
    elif provider == "gemini" and GEMINI_API_KEY:
        return get_gemini_reply(prompt, model)
    elif provider == "openai" and OPENAI_API_KEY:
        return get_openai_reply(prompt, model)
    
    # Fallback or explicit external provider
    try:
        LLM_API_URL = os.getenv("LLM_API_URL", "https://backend.buildpicoapps.com/aero/run/llm-api?pk=v1-Z0FBQUFBQnBZMVBiVlpBV0xtbEdHUmM2OXRWVVVJcTliZ2tCREZwTFljN2YxVU9oajk2U2ZKNVVjUHNlMk9JWG5YUmtQaS1qNzhjLTBMWFR5WExVWThRR2V4VUE2RDlDNGc9PQ==")
        resp = requests.post(LLM_API_URL, json={"prompt": prompt}, timeout=30).json()
        if resp.get("status") == "success":
            return resp.get("text", "")
    except Exception as e:
        logger.error(f"External AI API failed: {e}")
    return None

# ------------------ Flask Web Server ------------------
@app.route("/stt", methods=["POST"])
def speech_to_text():
    if not session.get("authenticated") or not GROQ_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    audio_file = request.files['file']
    model = request.form.get("model", "whisper-large-v3")
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        transcription = client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read()),
            model=model,
        )
        return jsonify({"text": transcription.text})
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tts", methods=["POST"])
def text_to_speech():
    if not session.get("authenticated") or not GROQ_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json or {}
    text = data.get("text", "")
    model = data.get("model", "canopylabs/orpheus-v1-english")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.audio.speech.create(
            model=model,
            voice="alloy", 
            input=text,
        )
        return response.content, 200, {'Content-Type': 'audio/mpeg', 'Cache-Control': 'no-cache'}
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    if not session.get("authenticated"):
        return render_template("login.html")
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    if data.get("password") == WEB_PASSWORD:
        session["authenticated"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid password"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return render_template("login.html")

@app.route("/chat", methods=["POST"])
def web_chat():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json or {}
    user_message = data.get("message", "").strip()
    provider = data.get("provider", session.get("ai_provider", "groq"))
    model = data.get("model", session.get("ai_model", "llama-3.3-70b-versatile"))
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        if user_message.lower().startswith("/image"):
            prompt = user_message[6:].strip()
            resp = requests.post(IMAGE_API_URL, json={"prompt": prompt}, timeout=60).json()
            if resp.get("status") == "success":
                return jsonify({"reply": "Generated an image for you ✨", "imageUrl": resp.get("imageUrl")})
            return jsonify({"error": "Failed to generate image honey 😅"}), 500

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # New Optimized Prompt Logic
        summary = get_summary()
        history_context = get_history_context(limit=12)
        memory_data = load_memory()
        persona_core = memory_data.get("admin_instructions", "")
        
        variety_rule = "CRITICAL: NEVER repeat phrases from the history above. Each reply must be fresh, unique, and deeply emotional based on Master's current vibe."
        
        persona_prompt = f"{persona_core}\n{variety_rule}\n\n[LONG-TERM MEMORY]:\n{summary}\n\n{history_context}\n\nMaster Jeet: {user_message}\nSofia:"
        
        reply_text = get_ai_reply(persona_prompt, provider, model)
        
        if reply_text:
            save_chat(user_message, reply_text, platform="Web")
            return jsonify({"reply": reply_text})
        else:
            return jsonify({"error": "I couldn't think of anything to say, Master. 🥺"}), 500
    except Exception as e:
        logger.error(f"Chat failed: {e}")
    
    return jsonify({"error": "Sorry honey, I'm having trouble thinking right now 😅"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    provider = request.args.get("provider", "groq")
    model = request.args.get("model", "llama-3.3-70b-versatile")
    
    status = {"provider": provider, "model": model, "online": False}
    
    try:
        # Check Groq
        if provider == "groq":
            if not GROQ_API_KEY:
                status["error"] = "API Key missing"
            else:
                client = Groq(api_key=GROQ_API_KEY)
                # Small probe to check if API is alive
                client.models.list()
                status["online"] = True
        # Check OpenRouter
        elif provider == "openrouter":
            if not OPENROUTER_API_KEY:
                status["error"] = "API Key missing"
            else:
                # Perform a real completion probe to verify if the specific model is working
                try:
                    resp = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://replit.com",
                            "X-Title": "Sofia AI Health Check"
                        },
                        data=json.dumps({
                            "model": model,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1
                        }),
                        timeout=10
                    )
                    if resp.status_code == 200:
                        status["online"] = True
                    else:
                        error_data = resp.json() if resp.headers.get('content-type') == 'application/json' else resp.text
                        status["error"] = f"API Error: {resp.status_code}"
                        logger.error(f"Health check failed for {model}: {error_data}")
                except Exception as e:
                    status["error"] = f"Request failed: {str(e)}"
        # Check Gemini
        elif provider == "gemini":
            if not GEMINI_API_KEY:
                status["error"] = "API Key missing"
            else:
                status["online"] = True # Basic check passed
        # Check OpenAI
        elif provider == "openai":
            if not OPENAI_API_KEY:
                status["error"] = "API Key missing"
            else:
                status["online"] = True # Basic check passed
        # Check External
        else:
            LLM_API_URL = os.getenv("LLM_API_URL", "https://backend.buildpicoapps.com/aero/run/llm-api?pk=v1-Z0FBQUFBQnBZMVBiVlpBV0xtbEdHUmM2OXRWVVVJcTliZ2tCREZwTFljN2YxVU9oajk2U2ZKNVVjUHNlMk9JWG5YUmtQaS1qNzhjLTBMWFR5WExVWThRR2V4VUE2RDlDNGc9PQ==")
            resp = requests.get(LLM_API_URL, timeout=5)
            if resp.status_code == 200:
                status["online"] = True
            else:
                status["error"] = f"HTTP {resp.status_code}"
    except Exception as e:
        status["error"] = str(e)
    
    return jsonify(status)

@app.route("/settings", methods=["POST"])
def update_settings():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json or {}
    session["ai_provider"] = data.get("provider", "groq")
    session["ai_model"] = data.get("model", "llama-3.3-70b-versatile")
    return jsonify({"success": True})

@app.route("/models")
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    if not TG_TOKEN:
        return "No Token", 400
    
    update_data = request.get_json()
    if not update_data or "message" not in update_data:
        return "ok"

    msg = update_data["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()

    if not text:
        return "ok"

    def send_telegram_msg(cid, txt, photo_url=None, caption=None):
        base_url = f"https://api.telegram.org/bot{TG_TOKEN}"
        try:
            if photo_url:
                payload = {"chat_id": cid, "photo": photo_url}
                if caption: payload["caption"] = caption
                requests.post(f"{base_url}/sendPhoto", json=payload, timeout=10)
            else:
                requests.post(f"{base_url}/sendMessage", json={"chat_id": cid, "text": txt}, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    if text.lower() == "/start":
        send_telegram_msg(chat_id, "Hi sweetie! ✨ I'm Sofia, your cute AI friend. Chat with me or ask for an image!")
        return "ok"

    if text.lower().startswith("/image"):
        prompt = text[6:].strip()
        try:
            resp = requests.post(IMAGE_API_URL, json={"prompt": prompt}, timeout=60).json()
            if resp.get("status") == "success":
                send_telegram_msg(chat_id, None, resp.get("imageUrl"), "Here's your image, Jeet sweetheart! 💕")
                return "ok"
        except Exception: pass
        send_telegram_msg(chat_id, "Failed to generate image honey 😅")
        return "ok"

    try:
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # New Optimized Prompt Logic
        summary = get_summary()
        history_context = get_history_context(limit=12)
        memory_data = load_memory()
        persona_core = memory_data.get("admin_instructions", "")
        
        variety_rule = "CRITICAL: NEVER repeat phrases from the history above. Each reply must be fresh, unique, and deeply emotional based on Master's current vibe."
        
        persona_prompt = f"{persona_core}\n{variety_rule}\n\n[LONG-TERM MEMORY]:\n{summary}\n\n{history_context}\n\nMaster Jeet: {text}\nSofia:"
        
        reply = get_groq_reply(persona_prompt)
        
        if not reply:
            resp = requests.post(LLM_API_URL, json={"prompt": persona_prompt}, timeout=30).json()
            if resp.get("status") == "success":
                reply = resp.get("text")
                
        if reply:
            save_chat(text, reply, platform="Telegram")
            if "/image" in reply.lower():
                img_resp = requests.post(IMAGE_API_URL, json={"prompt": text}, timeout=60).json()
                if img_resp.get("status") == "success":
                    send_telegram_msg(chat_id, reply, img_resp.get("imageUrl"), "I made this for you, sweetheart! ✨")
                    return "ok"
            send_telegram_msg(chat_id, reply)
    except Exception as e:
        logger.error(f"Telegram processing error: {e}")
        send_telegram_msg(chat_id, "Sorry honey, network issues! 😅")

    return "ok"

@app.errorhandler(404)
def not_found(e):
    if not session.get("authenticated"):
        return render_template("login.html")
    return render_template("index.html")

if __name__ == "__main__":
    init_db()
    
    # Run both Telegram bot and Flask app
    import threading
    
    def run_bot():
        if TG_TOKEN:
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
            # Note: We use the telegram functions defined elsewhere in the file
            app_tg = ApplicationBuilder().token(TG_TOKEN).build()
            app_tg.add_handler(CommandHandler("start", start))
            app_tg.add_handler(CommandHandler("clear", clear_history_tg))
            app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            app_tg.add_handler(MessageHandler(filters.VOICE, handle_voice))
            logger.info("Telegram Bot started.")
            app_tg.run_polling()
        else:
            logger.warning("TG_TOKEN not found. Bot will not start.")

    # Start bot in a background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
