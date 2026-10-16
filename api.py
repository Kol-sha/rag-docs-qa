# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import load_resources, ask

app = FastAPI(title="RAG QA System")

# Загружаем один раз при старте
model, index, chunks = load_resources()

class Question(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(q: Question):
    answer, sources = ask(q.query, model, index, chunks)
    return AnswerResponse(answer=answer, sources=sources)