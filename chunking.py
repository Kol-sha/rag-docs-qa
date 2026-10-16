# chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents import load_documents

def create_chunks(documents: dict[str, str], chunk_size: int = 500, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc_name, text in documents.items():
        doc_chunks = splitter.split_text(text)
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "text": chunk,
                "source": doc_name,
                "chunk_id": f"{doc_name}_chunk_{i}"
            })
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = create_chunks(docs)
    print(f"Всего чанков: {len(chunks)}")
    print(f"Пример: {chunks[0]['text'][:200]}...")