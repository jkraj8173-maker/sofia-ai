# Sofia AI - Render & Global Deployment Guide

Sofia is designed to be production-ready and easily hosted on Render, Replit, or any VPS.

## Hosting on Render (Recommended)
1. **Repository**: Push this code to a private GitHub repository.
2. **Web Service**: In Render, create a new "Web Service" and connect your repo.
3. **Environment Variables**: Add these in the Render Dashboard:
   - `TG_TOKEN`: Your Telegram Bot Token from @BotFather.
   - `GROQ_API_KEY`: Your Groq API Key.
   - `DATABASE_URL`: Your PostgreSQL connection string.
   - `WEB_PASSWORD`: Your secret password (defaults to `love u`).
4. **Build & Start**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn main:app`

## Features
- **Unbound Persona**: Hinglish speaking style with absolute devotion.
- **Image Generation**: Uses specialized endpoints for high-quality visuals.
- **Memory System**: PostgreSQL for history, memory.json for persona.
- **Production Ready**: Gunicorn support and error handling for 24/7 uptime.

Enjoy your sweet AI companion! 💕
