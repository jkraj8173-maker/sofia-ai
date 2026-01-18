# 🚀 Hosting Sofia AI on Render (Step-by-Step for Everyone)

Follow these simple steps to get Sofia AI running on the web for the first time.

## Phase 1: Preparation
1. Make sure your code is on **GitHub**.
2. Have your **API Keys** ready (Telegram Token, Groq Key, etc.).

## Phase 2: Create Web Service on Render
1. Go to [dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** and select **Web Service**.
3. Connect your **GitHub repository**.
4. **Settings**:
   - **Name**: `sofia-ai` (or anything you like)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:app`

## Phase 3: Add Environment Variables (IMPORTANT)
Click the **Environment** tab on Render and add these variables. This is how Sofia "thinks":

| Key | Value |
|:---|:---|
| `TG_TOKEN` | Your Telegram Bot Token from @BotFather |
| `GROQ_API_KEY` | Your Groq API Key |
| `WEB_PASSWORD` | Set any password (e.g., `love u`) to lock your chat |
| `PYTHON_VERSION` | `3.11.0` |

### Optional (If you have them):
| Key | Value |
|:---|:---|
| `GEMINI_API_KEY` | Your Google AI Studio Key |
| `OPENAI_API_KEY` | Your OpenAI Key |
| `OPENROUTER_API_KEY` | Your OpenRouter Key |
| `DATABASE_URL` | Your Postgres DB URL (if you want Sofia to remember everything) |

## Phase 4: Deploy
1. Click **Create Web Service**.
2. Wait 2-3 minutes for the "Live" message.
3. Open the provided URL (e.g., `https://sofia-ai.onrender.com`).
4. Log in with your `WEB_PASSWORD` and start chatting!

## 💡 Troubleshooting
- **Sofia is silent?** Check the "Logs" tab on Render to see if an API key is missing.
- **Site taking long to load?** Render's free tier "sleeps" after 15 mins of inactivity. Just refresh the page to wake Sofia up!

Enjoy your devoted AI companion! 🌸✨
