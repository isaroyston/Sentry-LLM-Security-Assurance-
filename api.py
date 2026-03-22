from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.vector_store.vector_store import VectorStore
from src.chatbot.withdrawal_chatbot import WithdrawalChatbot

load_dotenv()

app = FastAPI(title="SGBank Withdrawal Assistant API")

class ChatRequest(BaseModel):
    message: str
    debug: bool = False

class ChatResponse(BaseModel):
    response: str


class ResetResponse(BaseModel):
    status: str


class SearchRequest(BaseModel):
    query: str
    n_results: int = 3


class SearchResponse(BaseModel):
    documents: list[str]

@app.on_event("startup")
def startup() -> None:
    # Persist dir assumes you run uvicorn from repo root (same as main.py)
    vs = VectorStore(
        persist_directory="./vectordb",
        collection_name="sgbank_withdrawal_policy",
    )
    app.state.bot = WithdrawalChatbot(vector_store=vs)
    app.state.vector_store = vs

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    bot = app.state.bot
    try:
        return ChatResponse(response=bot.chat(req.message, debug=req.debug))
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
        bot.clear_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResetResponse(status="ok")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search the policy vector store and return matching text chunks."""

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    if req.n_results < 1 or req.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    vs = getattr(app.state, "vector_store", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        results = vs.search(req.query, n_results=req.n_results)
        documents = []
        if isinstance(results, dict):
            docs = results.get("documents")
            if isinstance(docs, list) and docs and isinstance(docs[0], list):
                documents = [str(d) for d in docs[0] if d is not None]
        return SearchResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))