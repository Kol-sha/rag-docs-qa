# documents.py
import os
from pypdf import PdfReader


def load_documents(docs_dir: str = "docs") -> dict[str, str]:
    """Загружает все .txt, .md и .pdf файлы из папки."""
    documents = {}
    for filename in os.listdir(docs_dir):
        path = os.path.join(docs_dir, filename)

        if filename.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                documents[filename] = f.read()

        elif filename.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                documents[filename] = text
            except Exception as e:
                print(f"Не удалось прочитать {filename}: {e}")

    return documents