# RAG-система для ответов на вопросы по документам

Retrieval-Augmented Generation система, которая отвечает на вопросы 
по коллекции PDF-документов, используя векторный поиск и LLM.

## Архитектура
PDF → чанкинг (RecursiveCharacterTextSplitter) → эмбеддинги 
(paraphrase-multilingual-MiniLM-L12-v2) → FAISS → retrieval → LLM (Groq gpt-oss-120b) → ответ

## Стек
- sentence-transformers — эмбеддинги
- faiss-cpu — векторный поиск
- langchain-text-splitters — чанкинг
- Groq API (gpt-oss-120b) — генерация
- FastAPI — API
- pypdf — чтение PDF

## Метрики
Оценка retrieval на 18 вопросах (русский + английский):
- Hit-rate@5: 1.00
- MRR: 0.92
- Precision@5: 0.36

## Запуск
1. pip install -r requirements.txt
2. export GROQ_API_KEY="gsk_..."
3. python3 indexing.py    # построить индекс
4. python3 rag_pipeline.py # тестовый запрос
5. uvicorn api:app --reload # API на http://localhost:8000/docs

## Структура
- documents.py — загрузка .txt/.md/.pdf
- chunking.py — разбиение на чанки
- indexing.py — эмбеддинги + FAISS
- rag_pipeline.py — retrieval + генерация
- api.py — FastAPI
- eval.py — оценка качества
- auto_label.py — помощник для разметки