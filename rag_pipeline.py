# rag_pipeline.py
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Настройки ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "./models/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
GROQ_MODEL = "openai/gpt-oss-120b"

# --- Инициализация Groq ---
# Ключ читается из переменной окружения GROQ_API_KEY.
# Если её нет — впишите ключ прямо здесь (только не коммитьте в Git).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


# --- Загрузка ресурсов ---
def load_resources():
    """Загружает модель эмбеддингов, FAISS-индекс и чанки."""
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks


# --- Retrieval ---
def retrieve(query: str, model, index, chunks, k: int = TOP_K):
    """Ищет top-k наиболее релевантных чанков по запросу."""
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results


# --- Prompt ---
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Собирает промпт из вопроса и найденного контекста."""
    context = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО следующий контекст.
Если ответа нет в контексте — скажи "Не знаю".
Отвечай кратко и по делу, на русском языке.

Контекст:
{context}

Вопрос: {query}

Ответ:"""
    return prompt


# --- Генерация через Groq ---
def generate_with_groq(prompt: str, model_name: str = GROQ_MODEL) -> str:
    """Отправляет промпт в Groq и возвращает ответ."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка при вызове Groq: {e}")
        return "Не удалось получить ответ от LLM."


# --- Полный пайплайн ---
def ask(query: str, model, index, chunks):
    """Полный RAG-пайплайн: retrieval + generation."""
    retrieved = retrieve(query, model, index, chunks)
    prompt = build_prompt(query, retrieved)
    answer = generate_with_groq(prompt)
    return answer, retrieved


# --- Точка входа для отладки ---
if __name__ == "__main__":
    model, index, chunks = load_resources()

    query = "Что такое RAG и зачем он нужен?"
    answer, sources = ask(query, model, index, chunks)

    print("Вопрос:", query)
    print("\nОтвет:", answer)
    print("\nИсточники:")
    for s in sources:
        print(f"  - {s['source']} ({s['chunk_id']})")