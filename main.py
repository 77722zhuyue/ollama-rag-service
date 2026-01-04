import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI(title="Ollama RAG Customer Service")

# 全局变量，稍后在 startup 中初始化
rag: RAGEngine = None


@app.on_event("startup")
async def startup_event():
    global rag
    print("🔧 正在初始化 RAG 引擎...")
    try:
        rag = RAGEngine()
        print("✅ RAG 引擎初始化完成")
    except Exception as e:
        print(f"❌ RAG 引擎初始化失败: {e}")
        raise RuntimeError(f"Failed to initialize RAG engine: {e}")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    latency_ms: int


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG 引擎尚未初始化，请稍后重试")
    
    start = time.time()
    try:
        result = rag.ask_with_cache(request.question)
        latency_ms = int((time.time() - start) * 1000)
        return QueryResponse(answer=result["answer"], latency_ms=latency_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@app.get("/health")
async def health():
    status = "ready" if rag is not None else "initializing"
    return {"status": status, "model": "gemma3:4b + bge-m3"}