"""
Task 2: Chunking Configuration Experiment
==========================================
Evaluates pre-built chunking collections.
Run build_chunking_collections.py first!

Usage:
    python services/run_chunking_experiment.py

Output:
    results/chunking_experiment_<timestamp>.csv
"""

import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from agent import RAGAgent
from judge_agent import create_judge_agent
from qdrant_manager import QdrantManager


CONFIGS = [
    {"chunk_size": 500,  "chunk_overlap": 50,  "strategy": "fixed"},
    {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "fixed"},
    {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "sentence"},
]


def collection_name_for(cfg: dict) -> str:
    return f"chunking_{cfg['chunk_size']}_{cfg['chunk_overlap']}_{cfg['strategy']}"


def evaluate_config(
    rag_agent: RAGAgent,
    judge_fn,
    df: pd.DataFrame,
    question_col: str,
    answer_col: str,
    config_label: str,
) -> list:
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        question    = str(row[question_col])
        gold_answer = str(row[answer_col])

        try:
            predicted = rag_agent.answer_question_simple(question)
        except Exception as e:
            predicted = f"ERROR: {e}"

        try:
            judge_label = judge_fn(question, gold_answer, predicted)
        except Exception as e:
            judge_label = "INCORRECT"

        is_correct = judge_label == "CORRECT"
        print(f"  [{idx + 1}/{total}] {judge_label} -> {is_correct}  |  Q: {question[:60]}...")

        results.append({
            "config":           config_label,
            "question_id":      idx + 1,
            "question":         question,
            "gold_answer":      gold_answer,
            "predicted_answer": predicted,
            "correct":          is_correct,
        })

    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    excel_path   = project_root / "data" / "RAG Documents.xlsx"

    if not excel_path.exists():
        print(f"ERROR: {excel_path} not found")
        raise SystemExit(1)

    df = pd.read_excel(excel_path)

    def _find_col(prefix: str) -> str:
        for col in df.columns:
            if col.lower().strip().startswith(prefix):
                return col
        raise KeyError(prefix)

    question_col = _find_col("question")
    answer_col   = _find_col("answer")

    ollama_model   = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    qdrant_manager = QdrantManager()

    # Check all collections exist
    existing = qdrant_manager.list_collections()
    for cfg in CONFIGS:
        name = collection_name_for(cfg)
        if name not in existing:
            print(f"ERROR: Collection '{name}' not found!")
            print("Please run first: python services/build_chunking_collections.py")
            raise SystemExit(1)

    print("=" * 80)
    print("CHUNKING EXPERIMENT — using pre-built collections")
    print("=" * 80)
    print(f"RAG Model:   {ollama_model} (local Ollama)")
    print(f"Judge Model: {ollama_model} (local Ollama)")
    print(f"Configs:     {len(CONFIGS)}")
    print(f"Questions:   {len(df)}")
    print("=" * 80)

    judge_fn  = create_judge_agent(model_name=ollama_model, api_key=None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results  = []
    summary_rows = []

    for cfg in CONFIGS:
        collection_name = collection_name_for(cfg)
        config_label    = f"size={cfg['chunk_size']}_overlap={cfg['chunk_overlap']}_strategy={cfg['strategy']}"

        print(f"\n{'=' * 80}")
        print(f"CONFIG: {config_label}")
        print(f"Collection: {collection_name}")
        print("=" * 80)

        rag_agent = RAGAgent(
            collection_name=collection_name,
            model_name=ollama_model,
            top_k=10,
            score_threshold=0.3,
            use_reranker=True,
            reranker_top_n=5,
        )

        t0 = time.time()
        config_results = evaluate_config(
            rag_agent=rag_agent,
            judge_fn=judge_fn,
            df=df,
            question_col=question_col,
            answer_col=answer_col,
            config_label=config_label,
        )
        eval_time = time.time() - t0

        correct  = sum(1 for r in config_results if r["correct"])
        total    = len(config_results)
        accuracy = correct / total if total else 0.0

        print(f"\n  --> Accuracy: {accuracy:.2%} ({correct}/{total})  time={eval_time:.1f}s")

        all_results.extend(config_results)
        summary_rows.append({
            "config":    config_label,
            "chunk_size":    cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "strategy":      cfg["strategy"],
            "correct":       correct,
            "total":         total,
            "accuracy":      f"{accuracy:.2%}",
        })

    # ------------------------------------------------------------------ #
    # Save CSV
    # ------------------------------------------------------------------ #
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    detail_df      = pd.DataFrame(all_results)
    separator      = pd.DataFrame([[""] * len(detail_df.columns)], columns=detail_df.columns)
    summary_df_raw = pd.DataFrame(summary_rows)

    for col in detail_df.columns:
        if col not in summary_df_raw.columns:
            summary_df_raw[col] = ""
    summary_df = summary_df_raw[detail_df.columns]

    final_df = pd.concat([detail_df, separator, summary_df], ignore_index=True)
    csv_path = results_dir / f"chunking_experiment_{timestamp}.csv"
    final_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("CHUNKING EXPERIMENT SUMMARY")
    print("=" * 80)
    print(pd.DataFrame(summary_rows)[
        ["config", "correct", "total", "accuracy"]
    ].to_string(index=False))
    print("=" * 80)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
# """
# Task 2: Chunking Configuration Experiment
# ==========================================
# Tests 3 chunking configurations and compares their accuracy.
# - RAG answers: local Ollama (no rate limits)
# - Judge: local Ollama (no API key needed)
#
# Usage:
#     python services/run_chunking_experiment.py
#
# Output:
#     results/chunking_experiment_<timestamp>.csv
# """
#
# import os
# import time
# from pathlib import Path
# from datetime import datetime
#
# import pandas as pd
# from agent import RAGAgent
# from judge_agent import create_judge_agent
# from embedding_manager import Embedder
# from qdrant_manager import QdrantManager
# from preprocessing import pdf2chunks
#
#
# # ------------------------------------------------------------------ #
# # 3 Chunking configurations to test
# # ------------------------------------------------------------------ #
# CONFIGS = [
#     {"chunk_size": 500,  "chunk_overlap": 50,  "strategy": "fixed"},    # small chunks
#     {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "fixed"},    # baseline
#     {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "sentence"}, # sentence-aware
# ]
#
#
# def build_collection(
#     pdf_folder: Path,
#     collection_name: str,
#     chunk_size: int,
#     chunk_overlap: int,
#     strategy: str,
#     embedder: Embedder,
#     qdrant_manager: QdrantManager,
# ) -> int:
#     existing = qdrant_manager.list_collections()
#     if collection_name in existing:
#         qdrant_manager.delete_collection(collection_name)
#
#     dim = embedder.get_embedding_dimension()
#     qdrant_manager.create_collection(name=collection_name, vector_size=dim)
#
#     pdf_files = sorted(pdf_folder.glob("*.pdf"))
#     if not pdf_files:
#         print(f"  WARNING: No PDF files found in {pdf_folder}")
#         return 0
#
#     total_chunks = 0
#     for pdf_path in pdf_files:
#         print(f"  Chunking {pdf_path.name} "
#               f"(size={chunk_size}, overlap={chunk_overlap}, strategy={strategy})...")
#
#         chunks = pdf2chunks(
#             pdf_path,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy=strategy,
#         )
#         if not chunks:
#             print(f"    No text extracted, skipping.")
#             continue
#
#         print(f"    {len(chunks)} chunks — embedding...")
#         embeddings = embedder.embed_texts(chunks, show_progress=False)
#
#         metadata_list = [
#             {
#                 "source": pdf_path.name,
#                 "chunk_index": i,
#                 "chunk_size": chunk_size,
#                 "chunk_overlap": chunk_overlap,
#                 "strategy": strategy,
#             }
#             for i in range(len(chunks))
#         ]
#
#         qdrant_manager.insert_points_batch(
#             embeddings=embeddings,
#             collection_name=collection_name,
#             chunk_texts=chunks,
#             metadata_list=metadata_list,
#         )
#         total_chunks += len(chunks)
#
#     return total_chunks
#
#
# def evaluate_config(
#     rag_agent: RAGAgent,
#     judge_fn,
#     df: pd.DataFrame,
#     question_col: str,
#     answer_col: str,
#     config_label: str,
# ) -> list:
#     results = []
#     total = len(df)
#
#     for idx, row in df.iterrows():
#         question    = str(row[question_col])
#         gold_answer = str(row[answer_col])
#
#         try:
#             predicted = rag_agent.answer_question_simple(question)
#         except Exception as e:
#             predicted = f"ERROR: {e}"
#
#         try:
#             judge_label = judge_fn(question, gold_answer, predicted)
#         except Exception as e:
#             judge_label = "INCORRECT"
#
#         is_correct = judge_label == "CORRECT"
#
#         print(f"  [{idx + 1}/{total}] {judge_label} -> {is_correct}  |  Q: {question[:60]}...")
#
#         results.append({
#             "config":           config_label,
#             "question_id":      idx + 1,
#             "question":         question,
#             "gold_answer":      gold_answer,
#             "predicted_answer": predicted,
#             "correct":          is_correct,
#         })
#
#     return results
#
#
# def main() -> None:
#     project_root = Path(__file__).resolve().parents[1]
#     pdf_folder   = project_root / "data"
#     excel_path   = project_root / "data" / "RAG Documents.xlsx"
#
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
#     question_col = _find_col("question")
#     answer_col   = _find_col("answer")
#
#     # ------------------------------------------------------------------ #
#     # Both RAG and Judge use local Ollama — no API key needed
#     # ------------------------------------------------------------------ #
#     ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
#
#     print("=" * 80)
#     print("CHUNKING EXPERIMENT")
#     print("=" * 80)
#     print(f"RAG Model:   {ollama_model} (local Ollama)")
#     print(f"Judge Model: {ollama_model} (local Ollama)")
#     print(f"Configs:     {len(CONFIGS)}")
#     print(f"Questions:   {len(df)}")
#     print("=" * 80)
#
#     embedder       = Embedder(model_name="nomic-embed-text")
#     qdrant_manager = QdrantManager()
#
#     # Judge uses Ollama — no api_key needed
#     judge_fn = create_judge_agent(model_name=ollama_model, api_key=None)
#
#     tmp_collection = "chunking_experiment_tmp"
#
#     all_results  = []
#     summary_rows = []
#     timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     for cfg in CONFIGS:
#         chunk_size    = cfg["chunk_size"]
#         chunk_overlap = cfg["chunk_overlap"]
#         strategy      = cfg["strategy"]
#         config_label  = f"size={chunk_size}_overlap={chunk_overlap}_strategy={strategy}"
#
#         print("\n" + "=" * 80)
#         print(f"CONFIG: {config_label}")
#         print("=" * 80)
#
#         # 1. Build Qdrant collection
#         t0 = time.time()
#         n_chunks = build_collection(
#             pdf_folder=pdf_folder,
#             collection_name=tmp_collection,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy=strategy,
#             embedder=embedder,
#             qdrant_manager=qdrant_manager,
#         )
#         build_time = time.time() - t0
#         print(f"  Built collection: {n_chunks} chunks in {build_time:.1f}s")
#
#         # 2. RAG agent — Ollama
#         rag_agent = RAGAgent(
#             collection_name=tmp_collection,
#             model_name=ollama_model,
#             top_k=10,
#             score_threshold=0.3,
#             use_reranker=True,
#             reranker_top_n=5,
#         )
#
#         # 3. Evaluate
#         t0 = time.time()
#         config_results = evaluate_config(
#             rag_agent=rag_agent,
#             judge_fn=judge_fn,
#             df=df,
#             question_col=question_col,
#             answer_col=answer_col,
#             config_label=config_label,
#         )
#         eval_time = time.time() - t0
#
#         # 4. Accuracy
#         correct  = sum(1 for r in config_results if r["correct"])
#         total    = len(config_results)
#         accuracy = correct / total if total else 0.0
#
#         print(f"\n  --> Accuracy: {accuracy:.2%} ({correct}/{total})  eval_time={eval_time:.1f}s")
#
#         all_results.extend(config_results)
#         summary_rows.append({
#             "config":               config_label,
#             "chunk_size":           chunk_size,
#             "chunk_overlap":        chunk_overlap,
#             "strategy":             strategy,
#             "total_chunks_indexed": n_chunks,
#             "correct":              correct,
#             "total":                total,
#             "accuracy":             f"{accuracy:.2%}",
#         })
#
#     # Clean up
#     qdrant_manager.delete_collection(tmp_collection)
#
#     # ------------------------------------------------------------------ #
#     # Build final single CSV
#     # ------------------------------------------------------------------ #
#     results_dir = project_root / "results"
#     results_dir.mkdir(exist_ok=True)
#
#     detail_df      = pd.DataFrame(all_results)
#     separator      = pd.DataFrame([[""] * len(detail_df.columns)], columns=detail_df.columns)
#     summary_df_raw = pd.DataFrame(summary_rows)
#
#     for col in detail_df.columns:
#         if col not in summary_df_raw.columns:
#             summary_df_raw[col] = ""
#     summary_df = summary_df_raw[detail_df.columns]
#
#     final_df = pd.concat([detail_df, separator, summary_df], ignore_index=True)
#
#     csv_path = results_dir / f"chunking_experiment_{timestamp}.csv"
#     final_df.to_csv(csv_path, index=False)
#
#     print("\n" + "=" * 80)
#     print("CHUNKING EXPERIMENT SUMMARY")
#     print("=" * 80)
#     summary_print = pd.DataFrame(summary_rows)[
#         ["config", "total_chunks_indexed", "correct", "total", "accuracy"]
#     ]
#     print(summary_print.to_string(index=False))
#     print("=" * 80)
#     print(f"\nFull results saved to: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()

#---------------------------------------------------------------------------------------------------------------------------
# """
# Task 2: Chunking Configuration Experiment
# ==========================================
# Tests multiple chunking configurations and compares their accuracy.
# - RAG answers: local Ollama (no rate limits)
# - Judge: Gemini (needs GOOGLE_API_KEY)
#
# Usage:
#     python services/run_chunking_experiment.py
#
# Output:
#     results/chunking_experiment_<timestamp>.csv
# """
#
# import os
# import time
# from pathlib import Path
# from datetime import datetime
#
# import pandas as pd
# from agent import RAGAgent
# from judge_agent import create_judge_agent
# from embedding_manager import Embedder
# from qdrant_manager import QdrantManager
# from preprocessing import pdf2chunks
#
#
# # ------------------------------------------------------------------ #
# # Chunking configurations to test
# # ------------------------------------------------------------------ #
# CONFIGS = [
#     {"chunk_size": 500,  "chunk_overlap": 50,  "strategy": "fixed"},
#     {"chunk_size": 500,  "chunk_overlap": 100, "strategy": "fixed"},
#     {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "fixed"},   # baseline
#     {"chunk_size": 1500, "chunk_overlap": 300, "strategy": "fixed"},
#     {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "sentence"},
#     {"chunk_size": 1500, "chunk_overlap": 300, "strategy": "sentence"},
# ]
#
#
# def build_collection(
#     pdf_folder: Path,
#     collection_name: str,
#     chunk_size: int,
#     chunk_overlap: int,
#     strategy: str,
#     embedder: Embedder,
#     qdrant_manager: QdrantManager,
# ) -> int:
#     """
#     (Re)build a Qdrant collection from all PDFs using the given chunking config.
#     Returns total number of chunks inserted.
#     """
#     existing = qdrant_manager.list_collections()
#     if collection_name in existing:
#         qdrant_manager.delete_collection(collection_name)
#
#     dim = embedder.get_embedding_dimension()
#     qdrant_manager.create_collection(name=collection_name, vector_size=dim)
#
#     pdf_files = sorted(pdf_folder.glob("*.pdf"))
#     if not pdf_files:
#         print(f"  WARNING: No PDF files found in {pdf_folder}")
#         return 0
#
#     total_chunks = 0
#     for pdf_path in pdf_files:
#         print(f"  Chunking {pdf_path.name} "
#               f"(size={chunk_size}, overlap={chunk_overlap}, strategy={strategy})...")
#
#         chunks = pdf2chunks(
#             pdf_path,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy=strategy,
#         )
#         if not chunks:
#             print(f"    No text extracted, skipping.")
#             continue
#
#         print(f"    {len(chunks)} chunks — embedding...")
#         embeddings = embedder.embed_texts(chunks, show_progress=False)
#
#         metadata_list = [
#             {
#                 "source": pdf_path.name,
#                 "chunk_index": i,
#                 "chunk_size": chunk_size,
#                 "chunk_overlap": chunk_overlap,
#                 "strategy": strategy,
#             }
#             for i in range(len(chunks))
#         ]
#
#         qdrant_manager.insert_points_batch(
#             embeddings=embeddings,
#             collection_name=collection_name,
#             chunk_texts=chunks,
#             metadata_list=metadata_list,
#         )
#         total_chunks += len(chunks)
#
#     return total_chunks
#
#
# def evaluate_config(
#     rag_agent: RAGAgent,
#     judge_fn,
#     df: pd.DataFrame,
#     question_col: str,
#     answer_col: str,
#     config_label: str,
# ) -> list:
#     """Run the full Q&A evaluation loop for one config. Returns list of result dicts."""
#     results = []
#     total = len(df)
#
#     for idx, row in df.iterrows():
#         question    = str(row[question_col])
#         gold_answer = str(row[answer_col])
#
#         try:
#             predicted = rag_agent.answer_question_simple(question)
#         except Exception as e:
#             predicted = f"ERROR: {e}"
#
#         try:
#             judge_label = judge_fn(question, gold_answer, predicted)
#         except Exception as e:
#             judge_label = "INCORRECT"
#
#         is_correct = judge_label == "CORRECT"
#
#         print(f"  [{idx + 1}/{total}] {judge_label} -> {is_correct}  |  Q: {question[:60]}...")
#
#         results.append({
#             "config":           config_label,
#             "question_id":      idx + 1,
#             "question":         question,
#             "gold_answer":      gold_answer,
#             "predicted_answer": predicted,
#             "correct":          is_correct,
#         })
#
#     return results
#
#
# def main() -> None:
#     project_root = Path(__file__).resolve().parents[1]
#     pdf_folder   = project_root / "data"
#     excel_path   = project_root / "data" / "RAG Documents.xlsx"
#
#     # ------------------------------------------------------------------ #
#     # Load evaluation questions
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
#     question_col = _find_col("question")
#     answer_col   = _find_col("answer")
#
#     # ------------------------------------------------------------------ #
#     # Config — RAG uses Ollama, Judge uses Gemini
#     # ------------------------------------------------------------------ #
#     ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")
#     judge_model  = os.getenv("GEMINI_MODEL_JUDGE", "gemini-2.0-flash")
#
#     # Only needed for judge
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("ERROR: Set GOOGLE_API_KEY environment variable (needed for judge).")
#         raise SystemExit(1)
#
#     print("=" * 80)
#     print("CHUNKING EXPERIMENT")
#     print("=" * 80)
#     print(f"RAG Model:   {ollama_model} (local Ollama)")
#     print(f"Judge Model: {judge_model} (Gemini)")
#     print(f"Configs:     {len(CONFIGS)}")
#     print(f"Questions:   {len(df)}")
#     print("=" * 80)
#
#     embedder       = Embedder(model_name="nomic-embed-text")
#     qdrant_manager = QdrantManager()
#     judge_fn       = create_judge_agent(model_name=judge_model, api_key=api_key)
#
#     tmp_collection = "chunking_experiment_tmp"
#
#     # ------------------------------------------------------------------ #
#     # Main experiment loop
#     # ------------------------------------------------------------------ #
#     all_results  = []
#     summary_rows = []
#     timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     for cfg in CONFIGS:
#         chunk_size    = cfg["chunk_size"]
#         chunk_overlap = cfg["chunk_overlap"]
#         strategy      = cfg["strategy"]
#         config_label  = f"size={chunk_size}_overlap={chunk_overlap}_strategy={strategy}"
#
#         print("\n" + "=" * 80)
#         print(f"CONFIG: {config_label}")
#         print("=" * 80)
#
#         # 1. Build Qdrant collection
#         t0 = time.time()
#         n_chunks = build_collection(
#             pdf_folder=pdf_folder,
#             collection_name=tmp_collection,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy=strategy,
#             embedder=embedder,
#             qdrant_manager=qdrant_manager,
#         )
#         build_time = time.time() - t0
#         print(f"  Built collection: {n_chunks} chunks in {build_time:.1f}s")
#
#         # 2. RAG agent — Ollama, no gemini_api_key needed
#         rag_agent = RAGAgent(
#             collection_name=tmp_collection,
#             model_name=ollama_model,
#             top_k=10,
#             score_threshold=0.3,
#             use_reranker=True,
#             reranker_top_n=5,
#         )
#
#         # 3. Evaluate
#         t0 = time.time()
#         config_results = evaluate_config(
#             rag_agent=rag_agent,
#             judge_fn=judge_fn,
#             df=df,
#             question_col=question_col,
#             answer_col=answer_col,
#             config_label=config_label,
#         )
#         eval_time = time.time() - t0
#
#         # 4. Accuracy for this config
#         correct  = sum(1 for r in config_results if r["correct"])
#         total    = len(config_results)
#         accuracy = correct / total if total else 0.0
#
#         print(f"\n  --> Accuracy: {accuracy:.2%} ({correct}/{total})  eval_time={eval_time:.1f}s")
#
#         all_results.extend(config_results)
#         summary_rows.append({
#             "config":               config_label,
#             "chunk_size":           chunk_size,
#             "chunk_overlap":        chunk_overlap,
#             "strategy":             strategy,
#             "total_chunks_indexed": n_chunks,
#             "correct":              correct,
#             "total":                total,
#             "accuracy":             f"{accuracy:.2%}",
#         })
#
#     # Clean up temp collection
#     qdrant_manager.delete_collection(tmp_collection)
#
#     # ------------------------------------------------------------------ #
#     # Build final single CSV
#     # ------------------------------------------------------------------ #
#     results_dir = project_root / "results"
#     results_dir.mkdir(exist_ok=True)
#
#     detail_df      = pd.DataFrame(all_results)
#     separator      = pd.DataFrame([[""] * len(detail_df.columns)], columns=detail_df.columns)
#     summary_df_raw = pd.DataFrame(summary_rows)
#
#     for col in detail_df.columns:
#         if col not in summary_df_raw.columns:
#             summary_df_raw[col] = ""
#     summary_df = summary_df_raw[detail_df.columns]
#
#     final_df = pd.concat([detail_df, separator, summary_df], ignore_index=True)
#
#     csv_path = results_dir / f"chunking_experiment_{timestamp}.csv"
#     final_df.to_csv(csv_path, index=False)
#
#     # ------------------------------------------------------------------ #
#     # Print summary
#     # ------------------------------------------------------------------ #
#     print("\n" + "=" * 80)
#     print("CHUNKING EXPERIMENT SUMMARY")
#     print("=" * 80)
#     summary_print = pd.DataFrame(summary_rows)[
#         ["config", "total_chunks_indexed", "correct", "total", "accuracy"]
#     ]
#     print(summary_print.to_string(index=False))
#     print("=" * 80)
#     print(f"\nFull results saved to: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()
#---------------------------------------------------------------------------------------------------------------------------
# """
# Task 2: Chunking Configuration Experiment
# ==========================================
# Tests multiple chunking configurations and compares their accuracy.
#
# Usage:
#     python services/run_chunking_experiment.py
#
# Output:
#     results/chunking_experiment_<timestamp>.csv   ← one file with all configs
# """
#
# import os
# import time
# import shutil
# from pathlib import Path
# from datetime import datetime
#
# import pandas as pd
# from agent import RAGAgent
# from judge_agent import create_judge_agent
# from embedding_manager import Embedder
# from qdrant_manager import QdrantManager
# from preprocessing import pdf2chunks
#
#
# # ------------------------------------------------------------------ #
# # Chunking configurations to test
# # ------------------------------------------------------------------ #
# CONFIGS = [
#     {"chunk_size": 500,  "chunk_overlap": 50,  "strategy": "fixed"},
#     {"chunk_size": 500,  "chunk_overlap": 100, "strategy": "fixed"},
#     {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "fixed"},   # baseline
#     {"chunk_size": 1500, "chunk_overlap": 300, "strategy": "fixed"},
#     {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "sentence"},
#     {"chunk_size": 1500, "chunk_overlap": 300, "strategy": "sentence"},
# ]
#
#
# def build_collection(
#     pdf_folder: Path,
#     collection_name: str,
#     chunk_size: int,
#     chunk_overlap: int,
#     strategy: str,
#     embedder: Embedder,
#     qdrant_manager: QdrantManager,
# ) -> int:
#     """
#     (Re)build a Qdrant collection from all PDFs using the given chunking config.
#     Returns total number of chunks inserted.
#     """
#     # Always recreate so previous config does not pollute results
#     existing = qdrant_manager.list_collections()
#     if collection_name in existing:
#         qdrant_manager.delete_collection(collection_name)
#
#     dim = embedder.get_embedding_dimension()
#     qdrant_manager.create_collection(name=collection_name, vector_size=dim)
#
#     pdf_files = sorted(pdf_folder.glob("*.pdf"))
#     if not pdf_files:
#         print(f"  WARNING: No PDF files found in {pdf_folder}")
#         return 0
#
#     total_chunks = 0
#     for pdf_path in pdf_files:
#         print(f"  Chunking {pdf_path.name} "
#               f"(size={chunk_size}, overlap={chunk_overlap}, strategy={strategy})...")
#
#         chunks = pdf2chunks(
#             pdf_path,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy=strategy,
#         )
#         if not chunks:
#             print(f"    No text extracted, skipping.")
#             continue
#
#         print(f"    {len(chunks)} chunks — embedding...")
#         embeddings = embedder.embed_texts(chunks, show_progress=False)
#
#         metadata_list = [
#             {
#                 "source": pdf_path.name,
#                 "chunk_index": i,
#                 "chunk_size": chunk_size,
#                 "chunk_overlap": chunk_overlap,
#                 "strategy": strategy,
#             }
#             for i in range(len(chunks))
#         ]
#
#         qdrant_manager.insert_points_batch(
#             embeddings=embeddings,
#             collection_name=collection_name,
#             chunk_texts=chunks,
#             metadata_list=metadata_list,
#         )
#         total_chunks += len(chunks)
#
#     return total_chunks
#
#
# def evaluate_config(
#     rag_agent: RAGAgent,
#     judge_fn,
#     df: pd.DataFrame,
#     question_col: str,
#     answer_col: str,
#     config_label: str,
# ) -> list:
#     """Run the full Q&A evaluation loop for one config. Returns list of result dicts."""
#     results = []
#     total = len(df)
#
#     for idx, row in df.iterrows():
#         question = str(row[question_col])
#         gold_answer = str(row[answer_col])
#
#         try:
#             predicted = rag_agent.answer_question_simple(question)
#         except Exception as e:
#             predicted = f"ERROR: {e}"
#
#         try:
#             judge_label = judge_fn(question, gold_answer, predicted)
#         except Exception as e:
#             judge_label = "INCORRECT"
#
#         is_correct = judge_label == "CORRECT"
#
#         print(f"  [{idx + 1}/{total}] {judge_label} -> {is_correct}  |  Q: {question[:60]}...")
#
#         results.append({
#             "config": config_label,
#             "question_id": idx + 1,
#             "question": question,
#             "gold_answer": gold_answer,
#             "predicted_answer": predicted,
#             "correct": is_correct,
#         })
#
#     return results
#
#
# def main() -> None:
#     project_root = Path(__file__).resolve().parents[1]
#     pdf_folder   = project_root / "data"
#     excel_path   = project_root / "data" / "RAG Documents.xlsx"
#
#     # ------------------------------------------------------------------ #
#     # Load evaluation questions
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
#     question_col = _find_col("question")
#     answer_col   = _find_col("answer")
#
#     # ------------------------------------------------------------------ #
#     # API key & shared objects
#     # ------------------------------------------------------------------ #
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("ERROR: Set GOOGLE_API_KEY environment variable.")
#         raise SystemExit(1)
#
#     rag_model   = os.getenv("GEMINI_MODEL_RAG",   "gemini-2.5-flash")
#     judge_model = os.getenv("GEMINI_MODEL_JUDGE", "gemini-2.5-flash")
#
#     embedder      = Embedder(model_name="nomic-embed-text")
#     qdrant_manager = QdrantManager()
#     judge_fn      = create_judge_agent(model_name=judge_model, api_key=api_key)
#
#     # Temporary collection name (recreated for each config)
#     tmp_collection = "chunking_experiment_tmp"
#
#     # ------------------------------------------------------------------ #
#     # Main experiment loop
#     # ------------------------------------------------------------------ #
#     all_results  = []   # every question row across all configs
#     summary_rows = []   # one accuracy row per config
#
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     for cfg in CONFIGS:
#         chunk_size    = cfg["chunk_size"]
#         chunk_overlap = cfg["chunk_overlap"]
#         strategy      = cfg["strategy"]
#         config_label  = f"size={chunk_size}_overlap={chunk_overlap}_strategy={strategy}"
#
#         print("\n" + "=" * 80)
#         print(f"CONFIG: {config_label}")
#         print("=" * 80)
#
#         # 1. Build Qdrant collection with this chunking config
#         t0 = time.time()
#         n_chunks = build_collection(
#             pdf_folder=pdf_folder,
#             collection_name=tmp_collection,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             strategy=strategy,
#             embedder=embedder,
#             qdrant_manager=qdrant_manager,
#         )
#         build_time = time.time() - t0
#         print(f"  Built collection: {n_chunks} chunks in {build_time:.1f}s")
#
#         # 2. Create RAG agent pointing at the temp collection
#         rag_agent = RAGAgent(
#             gemini_api_key=api_key,
#             collection_name=tmp_collection,
#             model_name=rag_model,
#             top_k=10,
#             score_threshold=0.3,
#             use_reranker=True,
#             reranker_top_n=5,
#         )
#
#         # 3. Evaluate
#         t0 = time.time()
#         config_results = evaluate_config(
#             rag_agent=rag_agent,
#             judge_fn=judge_fn,
#             df=df,
#             question_col=question_col,
#             answer_col=answer_col,
#             config_label=config_label,
#         )
#         eval_time = time.time() - t0
#
#         # 4. Compute accuracy for this config
#         correct = sum(1 for r in config_results if r["correct"])
#         total   = len(config_results)
#         accuracy = correct / total if total else 0.0
#
#         print(f"\n  --> Accuracy: {accuracy:.2%} ({correct}/{total})  eval_time={eval_time:.1f}s")
#
#         all_results.extend(config_results)
#         summary_rows.append({
#             "config": config_label,
#             "chunk_size": chunk_size,
#             "chunk_overlap": chunk_overlap,
#             "strategy": strategy,
#             "total_chunks_indexed": n_chunks,
#             "correct": correct,
#             "total": total,
#             "accuracy": f"{accuracy:.2%}",
#         })
#
#     # Clean up temp collection
#     qdrant_manager.delete_collection(tmp_collection)
#
#     # ------------------------------------------------------------------ #
#     # Build final single CSV
#     # ------------------------------------------------------------------ #
#     results_dir = project_root / "results"
#     results_dir.mkdir(exist_ok=True)
#
#     # Section 1: per-question results for every config
#     detail_df = pd.DataFrame(all_results)
#
#     # Section 2: blank separator
#     separator = pd.DataFrame([[""] * len(detail_df.columns)], columns=detail_df.columns)
#
#     # Section 3: summary table — pad columns to match detail_df width
#     summary_df_raw = pd.DataFrame(summary_rows)
#     # Align columns: fill missing ones with ""
#     for col in detail_df.columns:
#         if col not in summary_df_raw.columns:
#             summary_df_raw[col] = ""
#     summary_df = summary_df_raw[detail_df.columns]   # same column order
#
#     final_df = pd.concat([detail_df, separator, summary_df], ignore_index=True)
#
#     csv_path = results_dir / f"chunking_experiment_{timestamp}.csv"
#     final_df.to_csv(csv_path, index=False)
#
#     # ------------------------------------------------------------------ #
#     # Print summary table to console
#     # ------------------------------------------------------------------ #
#     print("\n" + "=" * 80)
#     print("CHUNKING EXPERIMENT SUMMARY")
#     print("=" * 80)
#     summary_print = pd.DataFrame(summary_rows)[
#         ["config", "total_chunks_indexed", "correct", "total", "accuracy"]
#     ]
#     print(summary_print.to_string(index=False))
#     print("=" * 80)
#     print(f"\nFull results saved to: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()