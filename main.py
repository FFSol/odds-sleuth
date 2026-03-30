"""
odds-sleuth — FastAPI application

Routes:
  GET  /              → serve index.html
  POST /api/brief     → stream briefing generation (SSE)
  POST /api/chat      → stream chat response (SSE)
  GET  /api/health    → health check
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from agent.orchestrator import generate_briefing, chat_response  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="odds-sleuth", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    briefing: str
    history: list[dict] = []
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    return {"status": "ok", "openai_key_set": has_key}


@app.post("/api/brief")
async def brief():
    """Stream the briefing generation as Server-Sent Events."""

    async def event_stream():
        try:
            async for chunk in generate_briefing():
                yield chunk
        except ValueError as e:
            # Missing API key etc.
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Agent error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream a chat response as Server-Sent Events."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.briefing.strip():
        raise HTTPException(status_code=400, detail="No briefing provided — generate one first.")

    async def event_stream():
        try:
            async for chunk in chat_response(
                briefing=request.briefing,
                conversation_history=request.history,
                user_message=request.message,
            ):
                yield chunk
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Agent error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
