"""
Task 3: Retrieval Accuracy Evaluation
======================================
For each question, checks:
1. Doc Accuracy:     at least 1 chunk from correct source PDF (True/False)
2. Chunk Precision:  how many of the top-5 chunks are from correct PDF (0%-100%)

Requires a 'source_pdf' column in RAG Documents2.xlsx

Usage:
    python services/run_retrieval_eval.py
"""

from pathlib import Path
from datetime import datetime
import os

import pandas as pd
from embedding_manager import Embedder
from qdrant_manager import QdrantManager


def normalize(name: str) -> str:
    """Remove .pdf extension and lowercase for comparison."""
    return name.lower().replace(".pdf", "").strip()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    excel_path   = project_root / "data" / "RAG Documents2.xlsx"

    # ------------------------------------------------------------------ #
    # Load evaluation data
    # ------------------------------------------------------------------ #
    if not excel_path.exists():
        print(f"ERROR: {excel_path} not found")
        raise SystemExit(1)

    df = pd.read_excel(excel_path)

    def _find_col(prefix: str) -> str:
        for col in df.columns:
            if col.lower().strip().startswith(prefix):
                return col
        raise KeyError(prefix)

    try:
        question_col   = _find_col("question")
        source_pdf_col = _find_col("source")
    except KeyError as e:
        print(f"ERROR: Could not find column starting with '{e}'")
        print(f"Available columns: {list(df.columns)}")
        raise SystemExit(1)

    collection_name = os.getenv("RAG_COLLECTION", "pdf_documents")
    top_k           = 5

    print("=" * 80)
    print("TASK 3: RETRIEVAL ACCURACY EVALUATION")
    print("=" * 80)
    print(f"Collection:      {collection_name}")
    print(f"Total questions: {len(df)}")
    print(f"Top-K retrieved: {top_k}")
    print(f"Metrics:")
    print(f"  1. Doc Accuracy    — at least 1 chunk from correct PDF")
    print(f"  2. Chunk Precision — how many of {top_k} chunks from correct PDF")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # Initialize
    # ------------------------------------------------------------------ #
    embedder       = Embedder(model_name="nomic-embed-text")
    qdrant_manager = QdrantManager()

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    results           = []
    correct_count     = 0   # for doc accuracy
    total_precision   = 0.0 # for avg chunk precision
    total             = len(df)

    for idx, row in df.iterrows():
        question        = str(row[question_col])
        expected_source = str(row[source_pdf_col]).strip()

        # Skip rows with no source_pdf
        if expected_source.lower() in ("nan", "", "none"):
            print(f"[{idx + 1}/{total}] SKIPPED — no source_pdf")
            total -= 1
            continue

        # Retrieve top-K chunks
        retrieved = qdrant_manager.search_by_text(
            query_text      = question,
            collection_name = collection_name,
            top_k           = top_k,
            score_threshold = 0.0,
        )

        retrieved_sources = [
            r["metadata"].get("source", "").strip()
            for r in retrieved
        ]

        # ── Metric 1: Doc Accuracy (at least 1 correct) ──
        hit = any(
            normalize(expected_source) in normalize(src)
            for src in retrieved_sources
        )
        correct_count += int(hit)

        # ── Metric 2: Chunk Precision (how many of top-5 are correct) ──
        correct_chunks = sum(
            1 for src in retrieved_sources
            if normalize(expected_source) in normalize(src)
        )
        chunk_precision = correct_chunks / len(retrieved_sources) if retrieved_sources else 0.0
        total_precision += chunk_precision

        print(f"[{idx + 1}/{total}]  doc_hit={hit}  chunk_precision={correct_chunks}/{len(retrieved_sources)} ({chunk_precision:.0%})")
        print(f"  Expected: {expected_source}")
        print(f"  Retrieved: {list(set(retrieved_sources))}")
        print("-" * 80)

        results.append({
            "question_id":       idx + 1,
            "question":          question,
            "expected_source":   expected_source,
            "retrieved_sources": ", ".join(retrieved_sources),
            "correct":           hit,                          # True/False
            "correct_chunks":    f"{correct_chunks}/{top_k}", # e.g. 3/5
            "chunk_precision":   f"{chunk_precision:.0%}",    # e.g. 60%
        })

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    doc_accuracy       = correct_count / total if total else 0.0
    avg_chunk_precision = total_precision / total if total else 0.0

    print("\n" + "=" * 80)
    print("RETRIEVAL ACCURACY RESULTS")
    print("=" * 80)
    print(f"Total Questions:    {total}")
    print(f"Doc Accuracy:       {correct_count}/{total} = {doc_accuracy:.2%}")
    print(f"Avg Chunk Precision:{avg_chunk_precision:.2%}")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # Save single CSV
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results)

    separator = pd.DataFrame(
        [[""] * len(results_df.columns)],
        columns=results_df.columns
    )

    # Two summary rows at the bottom
    summary_rows = pd.DataFrame([
        {
            "question_id":       "DOC ACCURACY",
            "question":          "at least 1 chunk from correct PDF",
            "expected_source":   "",
            "retrieved_sources": f"{correct_count}/{total}",
            "correct":           f"{doc_accuracy:.2%}",
            "correct_chunks":    "",
            "chunk_precision":   "",
        },
        {
            "question_id":       "CHUNK PRECISION",
            "question":          f"avg correct chunks out of top-{top_k}",
            "expected_source":   "",
            "retrieved_sources": "",
            "correct":           "",
            "correct_chunks":    "",
            "chunk_precision":   f"{avg_chunk_precision:.2%}",
        },
    ])

    final_df = pd.concat([results_df, separator, summary_rows], ignore_index=True)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = results_dir / f"retrieval_accuracy_{timestamp}.csv"
    final_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------------------------------------------
# """
# Task 3: Retrieval Accuracy Evaluation
# ======================================
# For each question, checks whether the retrieved chunks contain
# at least one chunk from the correct source PDF.
#
# Requires a 'source_pdf' column in RAG Documents.xlsx mapping
# each question to its ground-truth PDF filename.
#
# Usage:
#     python services/run_retrieval_eval.py
# """
#
# import os
# from pathlib import Path
# from datetime import datetime
#
# import pandas as pd
# from embedding_manager import Embedder
# from qdrant_manager import QdrantManager
#
#
# def main() -> None:
#     project_root = Path(__file__).resolve().parents[1]
#     excel_path   = project_root / "data" / "RAG Documents2.xlsx"
#
#     # ------------------------------------------------------------------ #
#     # Load evaluation data
#     # ------------------------------------------------------------------ #
#     if not excel_path.exists():
#         print(f"ERROR: {excel_path} not found")
#         raise SystemExit(1)
#
#     df = pd.read_excel(excel_path)
#
#     def _find_col(prefix: str) -> str:
#         for col in df.columns:
#             if col.lower().strip().startswith(prefix):
#                 return col
#         raise KeyError(prefix)
#
#     try:
#         question_col   = _find_col("question")
#         source_pdf_col = _find_col("source")   # e.g. "source_pdf" or "source"
#     except KeyError as e:
#         print(f"ERROR: Could not find column starting with '{e}' in Excel.")
#         print(f"Available columns: {list(df.columns)}")
#         print("Please add a 'source_pdf' column mapping each question to its PDF filename.")
#         raise SystemExit(1)
#
#     collection_name = os.getenv("RAG_COLLECTION", "pdf_documents")
#     top_k           = 5    # how many chunks to retrieve per question
#
#     print("=" * 80)
#     print("TASK 3: RETRIEVAL ACCURACY EVALUATION")
#     print("=" * 80)
#     print(f"Collection:      {collection_name}")
#     print(f"Total questions: {len(df)}")
#     print(f"Top-K retrieved: {top_k}")
#     print(f"Pass condition:  at least 1 chunk from correct source PDF")
#     print("=" * 80)
#
#     # ------------------------------------------------------------------ #
#     # Initialize embedder + Qdrant
#     # ------------------------------------------------------------------ #
#     embedder       = Embedder(model_name="nomic-embed-text")
#     qdrant_manager = QdrantManager()
#
#     # ------------------------------------------------------------------ #
#     # Evaluation loop
#     # ------------------------------------------------------------------ #
#     results      = []
#     correct_count = 0
#     total         = len(df)
#
#     for idx, row in df.iterrows():
#         question       = str(row[question_col])
#         expected_source = str(row[source_pdf_col]).strip()
#
#         # Retrieve top-K chunks
#         retrieved = qdrant_manager.search_by_text(
#             query_text       = question,
#             collection_name  = collection_name,
#             top_k            = top_k,
#             score_threshold  = 0.0,   # no threshold — we want exactly top_k
#         )
#
#         # Check if any retrieved chunk comes from the correct PDF
#         retrieved_sources = [
#             r["metadata"].get("source", "").strip()
#             for r in retrieved
#         ]
#
#         # Pass = at least one chunk from the correct source
#         def normalize(name: str) -> str:
#             return name.lower().replace(".pdf", "").strip()
#
#         hit = any(normalize(expected_source) in normalize(src) for src in retrieved_sources)
#         correct_count += int(hit)
#
#         print(f"[{idx + 1}/{total}]  correct={hit}")
#         print(f"  Question:        {question[:70]}...")
#         print(f"  Expected source: {expected_source}")
#         print(f"  Retrieved from:  {list(set(retrieved_sources))}")
#         print("-" * 80)
#
#         results.append({
#             "question_id":      idx + 1,
#             "question":         question,
#             "expected_source":  expected_source,
#             "retrieved_sources": ", ".join(set(retrieved_sources)),
#             "correct":          hit,        # True / False
#         })
#
#     # ------------------------------------------------------------------ #
#     # Accuracy
#     # ------------------------------------------------------------------ #
#     accuracy = correct_count / total if total else 0.0
#
#     print("\n" + "=" * 80)
#     print("RETRIEVAL ACCURACY RESULTS")
#     print("=" * 80)
#     print(f"Correct retrievals: {correct_count}/{total}")
#     print(f"Retrieval Accuracy: {accuracy:.2%}")
#     print("=" * 80)
#
#     # ------------------------------------------------------------------ #
#     # Save single CSV: per-question True/False + accuracy row at bottom
#     # ------------------------------------------------------------------ #
#     results_df = pd.DataFrame(results)
#
#     separator = pd.DataFrame(
#         [[""] * len(results_df.columns)],
#         columns=results_df.columns
#     )
#
#     accuracy_row = pd.DataFrame([{
#         "question_id":       "ACCURACY",
#         "question":          "",
#         "expected_source":   "",
#         "retrieved_sources": f"{correct_count}/{total}",
#         "correct":           f"{accuracy:.2%}",
#     }])
#
#     final_df = pd.concat([results_df, separator, accuracy_row], ignore_index=True)
#
#     results_dir = project_root / "results"
#     results_dir.mkdir(exist_ok=True)
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv_path  = results_dir / f"retrieval_accuracy_{timestamp}.csv"
#     final_df.to_csv(csv_path, index=False)
#
#     print(f"\nResults saved to: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()