"""
Pre-build all chunking collections for Task 2 experiment.
Run this ONCE before the demo to save time.

Usage:
    python services/build_chunking_collections.py

Collections created:
    chunking_500_50_fixed
    chunking_1000_200_fixed
    chunking_1000_200_sentence
"""

import time
from pathlib import Path
from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from preprocessing import pdf2chunks


CONFIGS = [
    {"chunk_size": 500,  "chunk_overlap": 50,  "strategy": "fixed"},
    {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "fixed"},
    {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "sentence"},
]


def collection_name_for(cfg: dict) -> str:
    return f"chunking_{cfg['chunk_size']}_{cfg['chunk_overlap']}_{cfg['strategy']}"


def build_collection(
    pdf_folder: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    embedder: Embedder,
    qdrant_manager: QdrantManager,
    force_rebuild: bool = False,
) -> int:
    existing = qdrant_manager.list_collections()

    if collection_name in existing:
        if not force_rebuild:
            print(f"  Collection '{collection_name}' already exists — skipping.")
            info = qdrant_manager.get_collection_info(collection_name)
            try:
                return info.points_count
            except Exception:
                return 0
        else:
            print(f"  Deleting and rebuilding '{collection_name}'...")
            qdrant_manager.delete_collection(collection_name)

    dim = embedder.get_embedding_dimension()
    qdrant_manager.create_collection(name=collection_name, vector_size=dim)

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"  WARNING: No PDF files found in {pdf_folder}")
        return 0

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"    Chunking {pdf_path.name}...")
        chunks = pdf2chunks(
            pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
        if not chunks:
            continue

        embeddings = embedder.embed_texts(chunks, show_progress=False)
        metadata_list = [
            {
                "source": pdf_path.name,
                "chunk_index": i,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "strategy": strategy,
            }
            for i in range(len(chunks))
        ]
        qdrant_manager.insert_points_batch(
            embeddings=embeddings,
            collection_name=collection_name,
            chunk_texts=chunks,
            metadata_list=metadata_list,
        )
        total_chunks += len(chunks)

    return total_chunks


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pdf_folder   = project_root / "data"

    embedder       = Embedder(model_name="nomic-embed-text")
    qdrant_manager = QdrantManager()

    print("=" * 80)
    print("PRE-BUILDING CHUNKING COLLECTIONS")
    print("=" * 80)
    print(f"Configs: {len(CONFIGS)}")
    print("Collections will be reused in run_chunking_experiment.py")
    print("=" * 80)

    t_total = time.time()

    for cfg in CONFIGS:
        name = collection_name_for(cfg)
        print(f"\nConfig: {name}")
        t0 = time.time()
        n = build_collection(
            pdf_folder=pdf_folder,
            collection_name=name,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            strategy=cfg["strategy"],
            embedder=embedder,
            qdrant_manager=qdrant_manager,
            force_rebuild=False,  # skip if already exists
        )
        print(f"  Done: {n} chunks in {time.time() - t0:.1f}s")

    print("\n" + "=" * 80)
    print(f"All collections ready in {time.time() - t_total:.1f}s")
    print("Now run: python services/run_chunking_experiment.py")
    print("=" * 80)


if __name__ == "__main__":
    main()