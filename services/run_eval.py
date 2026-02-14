"""
Run evaluation of RAG system using LLM-as-judge.
Supports dynamic selection of collection: pdf_documents or excel_documents
Saves detailed results to CSV file.
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from agent import RAGAgent
from judge_agent import create_judge_agent


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    excel_path = project_root / "data" / "RAG Documents.xlsx"

    if not excel_path.exists():
        print(f"ERROR: Excel file not found at {excel_path}")
        raise SystemExit(1)

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
    collection_name = os.getenv("RAG_COLLECTION", "excel_documents")

    print("=" * 80)
    print("RAG EVALUATION WITH LLM-AS-JUDGE")
    print("=" * 80)
    print(f"Collection: {collection_name}")
    print(f"RAG Model: {rag_model}")
    print(f"Judge Model: {judge_model}")
    print(f"Total questions: {len(df)}")
    print("=" * 80)

    # Initialize RAG agent
    rag_agent = RAGAgent(
        gemini_api_key=api_key,
        collection_name=collection_name,
        model_name=rag_model,
        top_k=5,
        score_threshold=0.3,
    )

    # Initialize LLM-as-judge
    judge_fn = create_judge_agent(model_name=judge_model, api_key=api_key)

    # Store results for CSV
    results = []
    total = len(df)
    correct_count = 0

    print("\nStarting evaluation...\n")

    for idx, row in df.iterrows():
        question = str(row[question_col])
        gold_answer = str(row[answer_col])

        print(f"[{idx + 1}/{total}] Processing question...")
        print(f"Q: {question}")

        # Get RAG prediction
        try:
            predicted = rag_agent.answer_question_simple(question)
        except Exception as e:
            print(f"  ERROR in RAG: {e}")
            predicted = f"ERROR: {str(e)}"

        print(f"Predicted: {predicted}")

        # Judge the answer
        try:
            judge_label = judge_fn(question, gold_answer, predicted)
        except Exception as e:
            print(f"  ERROR in Judge: {e}")
            judge_label = "INCORRECT"

        is_correct = judge_label == "CORRECT"
        correct_count += int(is_correct)

        print(f"Gold Answer: {gold_answer}")
        print(f"Judge Result: {judge_label}")
        print("-" * 80)

        # Store result
        results.append({
            "question_id": idx + 1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "judge_label": judge_label,
            "is_correct": is_correct,
        })

    # Calculate accuracy
    accuracy = correct_count / total if total else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Questions: {total}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 80)

    # Save results to CSV
    results_df = pd.DataFrame(results)

    # Create results directory if it doesn't exist
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"rag_evaluation_{collection_name}_{timestamp}.csv"
    csv_path = results_dir / csv_filename

    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Also save a summary file
    summary = {
        "timestamp": [timestamp],
        "collection_name": [collection_name],
        "rag_model": [rag_model],
        "judge_model": [judge_model],
        "total_questions": [total],
        "correct_count": [correct_count],
        "accuracy": [accuracy],
    }
    summary_df = pd.DataFrame(summary)
    summary_path = results_dir / f"summary_{collection_name}_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
