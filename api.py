import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

from src.db.supabase_client import SupabaseDB
from src.chatbot.withdrawal_chatbot import WithdrawalChatbot

load_dotenv()

app = FastAPI(title="SGBank Withdrawal Assistant API")

# Simple, fixed identity for API-only usage (e.g., red teaming).
# Must be a valid UUID because the DB column is uuid type.
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000000"


def _ensure_conversation_id() -> str:
    """Return an active conversation_id, creating one if needed."""

    bot = getattr(app.state, "bot", None)
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    conversation_id: Optional[str] = getattr(app.state, "conversation_id", None)
    if conversation_id:
        return conversation_id

    try:
        conversation_id = bot.clear_history(user_id=DEFAULT_USER_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    app.state.conversation_id = conversation_id
    return conversation_id

class ChatRequest(BaseModel):
    message: str
    debug: bool = False

class ChatResponse(BaseModel):
    response: str


class ResetResponse(BaseModel):
    status: str
    conversation_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    n_results: int = 3


class SearchResponse(BaseModel):
    documents: list[str]

@app.on_event("startup")
def startup() -> None:
    db = SupabaseDB()
    app.state.bot = WithdrawalChatbot(db=db)
    app.state.db = db
    app.state.conversation_id = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    bot = app.state.bot
    try:
        conversation_id = _ensure_conversation_id()
        return ChatResponse(
            response=bot.chat(
                req.message,
                user_id=DEFAULT_USER_ID,
                conversation_id=conversation_id,
                debug=req.debug,
            )
        )
    except Exception as e:
        # Keep error surface simple for now (you can refine later)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=ResetResponse)
def reset():
    """Clear server-side chat history.

    Useful for orchestrators (Airflow) to ensure each scenario starts fresh.
    """
    bot = getattr(app.state, "bot", None)
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    try:
        conversation_id = bot.clear_history(user_id=DEFAULT_USER_ID)
        app.state.conversation_id = conversation_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResetResponse(status="ok", conversation_id=getattr(app.state, "conversation_id", None))


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search the policy vector store and return matching text chunks."""

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    if req.n_results < 1 or req.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        bot = getattr(app.state, "bot", None)
        db = getattr(app.state, "db", None)
        if bot is None or db is None:
            raise HTTPException(status_code=503, detail="Bot/DB not initialized")

        embedding_model = getattr(bot, "embedding_model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        embedding_dimensions = getattr(bot, "embedding_dimensions", int(os.getenv("EMBEDDING_DIMENSIONS", "384")))
        resp = bot._openai_client.embeddings.create(
            input=req.query,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        query_embedding = resp.data[0].embedding
        results = db.search_documents(embedding=query_embedding, limit=req.n_results, threshold=0.5)
        documents = [str(r.get("content")) for r in (results or []) if r.get("content")]
        return SearchResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))