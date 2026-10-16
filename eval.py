# eval.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_pipeline import load_resources, retrieve


# ============================================================
# ТЕСТОВЫЙ НАБОР
# ============================================================
# Вопросы составлены по документам из docs/.
# relevant_ids — чанки, содержащие ответ.
# Если HR@5 = 0 для какого-то вопроса — уточните chunk_id
# через: python3 show_chunks.py "ключевое слово"
# ============================================================

TEST_CASES = [
    # --- Русскоязычные документы ---
    {
        "question": "Что такое машинное обучение и как оно применяется в бизнес-аналитике?",
        "relevant_ids": [
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_1",
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_2",
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_13",
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_38",
        ]
    },
    {
        "question": "Какие преимущества даёт машинное обучение для бизнеса?",
        "relevant_ids": [
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_3",
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_4",
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_17",
            "Masinnoe_obucenie_v_biznes-analitike.pdf_chunk_38",
        ]
    },
    {
        "question": "Какие методы машинного обучения используются для анализа текстовых данных?",
        "relevant_ids": [
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_1",
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_3",
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_6",
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_22",
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_23",
        ]
    },
    {
        "question": "Что такое обработка естественного языка (NLP)?",
        "relevant_ids": [
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_1",
            "metody-analiza-tekstovyh-dannyh-s-ispolzovaniem-mashinnogo-obucheniya.pdf_chunk_2",
            "10.pdf_chunk_26",
            "3.pdf_chunk_27",
            "3.pdf_chunk_28",
        ]
    },
    {
        "question": "Как установить Python на Windows?",
        "relevant_ids": [
            "pythonworldru.pdf_chunk_30",
            "pythonworldru.pdf_chunk_32",
            "pythonworldru.pdf_chunk_34",
        ]
    },
    {
        "question": "Какие циклы есть в Python?",
        "relevant_ids": [
            "pythonworldru.pdf_chunk_62",
            "pythonworldru.pdf_chunk_63",
            "pythonworldru.pdf_chunk_64",
        ]
    },
    {
        "question": "Что делают операторы break и continue в Python?",
        "relevant_ids": [
            "pythonworldru.pdf_chunk_64",
            "pythonworldru.pdf_chunk_65",
            "pythonworldru.pdf_chunk_66",
            "pythonworldru.pdf_chunk_67",
        ]
    },
    {
        "question": "Что такое условная инструкция if в Python?",
        "relevant_ids": [
            "pythonworldru.pdf_chunk_57",
            "pythonworldru.pdf_chunk_58",
            "pythonworldru.pdf_chunk_59",
        ]
    },

    # --- Англоязычные документы ---
    {
        "question": "What is LangChain and what is it used for?",
        "relevant_ids": [
            "LangChain.pdf_chunk_0",
            "LangChain.pdf_chunk_1",
            "LangChain.pdf_chunk_8",
            "LangChain.pdf_chunk_12",
        ]
    },
    {
        "question": "How does LangChain support retrieval-augmented generation (RAG)?",
        "relevant_ids": [
            "LangChain.pdf_chunk_1",
            "LangChain.pdf_chunk_25",
            "LangChain.pdf_chunk_26",
            "LangChain.pdf_chunk_27",
        ]
    },
    {
        "question": "What are the core components of LangChain?",
        "relevant_ids": [
            "LangChain.pdf_chunk_2",
            "LangChain.pdf_chunk_31",
            "LangChain.pdf_chunk_10",
            "LangChain.pdf_chunk_13",
            "LangChain.pdf_chunk_60",
        ]
    },
    {
        "question": "What is AI-driven data analytics?",
        "relevant_ids": [
            "10.pdf_chunk_6",
            "10.pdf_chunk_7",
            "3.pdf_chunk_1",
            "10.pdf_chunk_38",
        ]
    },
    {
        "question": "How is AI used in the telecom industry?",
        "relevant_ids": [
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_1",
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_4",
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_5",
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_13",
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_41",
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_99",
            "AI-DrivenDataAnalyticsInsightsforTelecomGrowthStrategies.pdf_chunk_100",
        ]
    },
    {
        "question": "What is the role of AI in business intelligence?",
        "relevant_ids": [
            "BUSINESS_INTELLIGENCE_TRANSFORMATION_THROUGH_AI_AN.pdf_chunk_3",
            "BUSINESS_INTELLIGENCE_TRANSFORMATION_THROUGH_AI_AN.pdf_chunk_18",
            "BUSINESS_INTELLIGENCE_TRANSFORMATION_THROUGH_AI_AN.pdf_chunk_31",
            "BUSINESS_INTELLIGENCE_TRANSFORMATION_THROUGH_AI_AN.pdf_chunk_145",
            "BUSINESS_INTELLIGENCE_TRANSFORMATION_THROUGH_AI_AN.pdf_chunk_167",
        ]
    },
    {
        "question": "How is AI applied in solar energy?",
        "relevant_ids": [
            "Artificial_Intelligence_in_Solar_Energy_Innovation.pdf_chunk_2",
            "Artificial_Intelligence_in_Solar_Energy_Innovation.pdf_chunk_3",
            "Artificial_Intelligence_in_Solar_Energy_Innovation.pdf_chunk_15",
            "Artificial_Intelligence_in_Solar_Energy_Innovation.pdf_chunk_16",
            "Artificial_Intelligence_in_Solar_Energy_Innovation.pdf_chunk_83",
        ]
    },
    {
        "question": "What are the main AI techniques used in data analytics?",
        "relevant_ids": [
            "3.pdf_chunk_1",
            "3.pdf_chunk_2",
            "3.pdf_chunk_40",
            "3.pdf_chunk_41",
            "10.pdf_chunk_22",
        ]
    },
    {
        "question": "What are the challenges of AI in data analytics?",
        "relevant_ids": [
            "10.pdf_chunk_9",
            "10.pdf_chunk_10",
            "10.pdf_chunk_58",
            "10.pdf_chunk_94",
            "3.pdf_chunk_67",
        ]
    },
    {
        "question": "How has artificial intelligence and machine learning grown recently?",
        "relevant_ids": [
            "100-ArticleText-664-1-10-20211219.pdf_chunk_3",
            "100-ArticleText-664-1-10-20211219.pdf_chunk_4",
            "100-ArticleText-664-1-10-20211219.pdf_chunk_26",
            "100-ArticleText-664-1-10-20211219.pdf_chunk_27",
        ]
    },
]


def hit_rate_at_k(retrieved, relevant_ids, k=5):
    top_k = {c["chunk_id"] for c in retrieved[:k]}
    return 1.0 if top_k & set(relevant_ids) else 0.0


def precision_at_k(retrieved, relevant_ids, k=5):
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if c["chunk_id"] in relevant_ids)
    return hits / len(top_k)


def mrr(retrieved, relevant_ids):
    for i, c in enumerate(retrieved, start=1):
        if c["chunk_id"] in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate(k=5):
    model, index, chunks = load_resources()

    hit_rates, precisions, mrrs = [], [], []

    print(f"{'Вопрос':<52} {'HR@5':>6} {'P@5':>6} {'MRR':>6}")
    print("-" * 74)

    for case in TEST_CASES:
        retrieved = retrieve(case["question"], model, index, chunks, k=k)
        relevant = set(case["relevant_ids"])
        
        # Отладка: если провал — показываем, что нашлось
        if not (set(c["chunk_id"] for c in retrieved) & relevant):
            print(f"\n⚠️ ПРОВАЛ: {case['question']}")
            print(f"   Ожидали: {case['relevant_ids']}")
            print(f"   Нашли:   {[c['chunk_id'] for c in retrieved]}")
            print(f"   Топ-1 текст: {retrieved[0]['text'][:200]}")
        
        hr = hit_rate_at_k(retrieved, relevant, k)
        p = precision_at_k(retrieved, relevant, k)
        r = mrr(retrieved, relevant)

        hit_rates.append(hr)
        precisions.append(p)
        mrrs.append(r)

        q = case["question"][:49]
        print(f"{q:<52} {hr:>6.2f} {p:>6.2f} {r:>6.2f}")

    print("-" * 74)
    n = len(TEST_CASES)
    print(f"{'Среднее':<52} "
          f"{sum(hit_rates)/n:>6.2f} "
          f"{sum(precisions)/n:>6.2f} "
          f"{sum(mrrs)/n:>6.2f}")

    print(f"\nИтоговые метрики на {n} вопросах:")
    print(f"  Hit-rate@{k}:    {sum(hit_rates)/n:.2f}")
    print(f"  Precision@{k}:   {sum(precisions)/n:.2f}")
    print(f"  MRR:             {sum(mrrs)/n:.2f}")


if __name__ == "__main__":
    evaluate(k=10)