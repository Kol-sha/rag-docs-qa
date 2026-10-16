# show_chunks.py
import pickle
import sys

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def search(keyword: str, limit: int = 10):
    """Ищет чанки, содержащие keyword."""
    keyword_lower = keyword.lower()
    found = []
    for c in chunks:
        if keyword_lower in c["text"].lower():
            found.append(c)
            if len(found) >= limit:
                break

    print(f"\n=== Найдено {len(found)} чанков по запросу '{keyword}' ===")
    for c in found:
        print(f"\n[{c['chunk_id']}]")
        print(c["text"][:400].replace("\n", " "))
        print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        search(" ".join(sys.argv[1:]))
    else:
        print("Использование: python3 show_chunks.py <ключевое_слово>")
        print("Пример: python3 show_chunks.py RAG")