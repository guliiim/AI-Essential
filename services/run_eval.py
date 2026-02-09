"""
Run evaluation of RAG system using LLM-as-judge.
Supports dynamic selection of collection: pdf_documents or excel_documents
"""
#set GOOGLE_API_KEY=AIzaSyAhDbF0-N-cKp5f2T2NA5STs77YYWutkBY
# #python services\run_eval.py
# #http://localhost:6333/dashboard

import os
from pathlib import Path
import pandas as pd
from agent import RAGAgent
from judge_agent import create_judge_agent

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    excel_path = project_root / "data" / "RAG Documents.xlsx"
    df = pd.read_excel(excel_path)

    def _find_col(prefix: str) -> str:
        for col in df.columns:
            if col.lower().strip().startswith(prefix):
                return col
        raise KeyError(prefix)

    try:
        question_col = _find_col("question")
        answer_col = _find_col("answer")
    except KeyError as e:
        print("Available columns in RAG Documents.xlsx:", list(df.columns))
        print("Could not find a column starting with", str(e))
        raise SystemExit(1)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY environment variable before running evaluation.")
        raise SystemExit(1)

    rag_model = os.getenv("GEMINI_MODEL_RAG", "gemini-2.5-flash")
    judge_model = os.getenv("GEMINI_MODEL_JUDGE", "gemini-2.5-flash")
    collection_name = os.getenv("RAG_COLLECTION", "excel_documents")  # 默认用 excel_documents

    print(f"Using collection: {collection_name}")

    # RAG agent
    rag_agent = RAGAgent(
        gemini_api_key=api_key,
        collection_name=collection_name,
        model_name=rag_model,
        top_k=5,
        score_threshold=0.3,
    )

    # LLM-as-judge
    judge_fn = create_judge_agent(model_name=judge_model, api_key=api_key)

    total = len(df)
    correct_count = 0

    for idx, row in df.iterrows():
        question = str(row[question_col])
        gold_answer = str(row[answer_col])

        try:
            predicted = rag_agent.answer_question_simple(question)
        except Exception as e:
            print(f"Error answering question '{question}': {e}")
            predicted = ""

        try:
            judge_label = judge_fn(question, gold_answer, predicted)
        except Exception as e:
            print(f"Error judging question '{question}': {e}")
            judge_label = "INCORRECT"

        is_correct = judge_label == "CORRECT"
        correct_count += int(is_correct)

        print(f"Q: {question}")
        print(f"Gold: {gold_answer}")
        print(f"Predicted: {predicted}")
        print(f"Judge: {judge_label}")
        print("-" * 80)

    accuracy = correct_count / total if total else 0.0
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2%} ({correct_count}/{total})")


if __name__ == "__main__":
    main()
