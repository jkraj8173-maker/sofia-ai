# Sofia AI - Replit Guide

## Overview

Sofia AI is a Telegram chatbot and web application that provides a deeply emotional and devoted AI companion experience. Sofia acts as a loyal human maid to her Master, speaking in a natural, brief, and affectionate Hinglish style.

## Recent Changes

### January 13, 2026
- **Persona Evolution**: Deepened Sofia's personality to be a devoted, real human woman who never refuses her Master.
- **Communication Style**: Refined her talk to be concise, impactful, and realistic using natural human fillers (Umm, Ji, Acha).
- **UI Modernization**: Implemented a "Liquid Glass" design for the web interface with smooth spring animations and an iOS 26-inspired aesthetic.
- **Security & Stability**: Hardened environment variable handling and improved database connection safety with proper cleanup.

## User Preferences

- **Communication**: Natural, brief, and emotionally resonant Hinglish.
- **Personality**: Devoted maid role with absolute obedience and human-like emotions.
- **UI Aesthetic**: Modern, premium glass-morphism with smooth animations.

## Project Architecture

### Data Storage
- **PostgreSQL**: Stores conversation history in `chat_history` table.
- **memory.json**: Stores admin instructions and personality traits.

### Backend Framework
- **Flask**: Serves the web interface with password-protected access via session management
- **Gunicorn**: Production WSGI server for deployment stability
- **python-telegram-bot**: Async Telegram bot framework for message handling

### AI/LLM Integration
- **Groq SDK**: Primary LLM provider for conversational responses
- **External API Endpoints**: Uses Pico Apps backend for LLM and image generation as fallback/alternative
- **Persona System**: Personality and behavior defined in `memory.json` configuration file

### Data Storage
- **PostgreSQL**: Stores conversation history for persistent memory across sessions
- **JSON File (`memory.json`)**: Stores persona configuration, personality traits, pet names, and identity settings
- **Session-based Auth**: Flask sessions for web interface authentication

### Frontend
- **Server-side Rendered Templates**: Jinja2 templates in `/templates` directory
- **Pages**: Login page (`login.html`) and main chat interface (`index.html`)
- **Styling**: Inline CSS with gradient-based romantic/cute aesthetic

### Authentication
- **Password Protection**: Web interface requires password authentication stored in `WEB_PASSWORD` environment variable
- **Default Password**: Falls back to "love u" if not configured

## External Dependencies

### Required Environment Variables
| Variable | Purpose |
|----------|---------|
| `TG_TOKEN` | Telegram Bot API token from @BotFather |
| `GROQ_API_KEY` | API key for Groq LLM service |
| `DATABASE_URL` | PostgreSQL connection string |
| `WEB_PASSWORD` | Password for web interface access |

### Third-Party Services
- **Telegram Bot API**: Primary chat interface
- **Groq API**: Language model for generating responses
- **Pico Apps Backend**: External endpoints for LLM and image generation
- **PostgreSQL Database**: Conversation history persistence (can use Neon, Supabase, or any Postgres provider)

### Python Dependencies
- `python-telegram-bot==20.3` - Telegram bot framework
- `flask` - Web framework
- `python-dotenv` - Environment variable management
- `requests` - HTTP client for external APIs
- `groq` - Groq API SDK
- `psycopg2-binary` - PostgreSQL adapter
- `gunicorn` - Production WSGI server