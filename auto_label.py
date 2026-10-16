# auto_label.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_pipeline import load_resources, retrieve
from eval import TEST_CASES


def show_candidates(k=10):
    model, index, chunks = load_resources()

    for case in TEST_CASES:
        print("\n" + "=" * 80)
        print(f"ВОПРОС: {case['question']}")
        print(f"Текущая разметка: {case['relevant_ids']}")
        print("=" * 80)

        retrieved = retrieve(case["question"], model, index, chunks, k=k)

        for i, c in enumerate(retrieved, start=1):
            marker = "✅" if c["chunk_id"] in case["relevant_ids"] else "  "
            print(f"\n{marker} [{i}] {c['chunk_id']}")
            print(f"    {c['text'][:400].replace(chr(10), ' ')}")

        input("\nНажмите Enter, чтобы перейти к следующему вопросу...")


if __name__ == "__main__":
    show_candidates(k=10)