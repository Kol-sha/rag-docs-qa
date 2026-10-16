# indexing.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from chunking import create_chunks
from documents import load_documents

# Мультиязычная модель, работает с русским
MODEL_NAME = "./models/paraphrase-multilingual-MiniLM-L12-v2"

def build_index(chunks: list[dict]):
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    
    print("Создаю эмбеддинги...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    print(f"Индекс создан: {index.ntotal} векторов, размерность {dimension}")
    return model, index, texts

if __name__ == "__main__":
    docs = load_documents()
    chunks = create_chunks(docs)
    model, index, texts = build_index(chunks)
    
    # Сохраняем для переиспользования
    faiss.write_index(index, "faiss_index.bin")
    import pickle
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Индекс и чанки сохранены.")