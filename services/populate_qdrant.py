from pathlib import Path
from typing import List
import time

from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from preprocessing import pdf2chunks


class QdrantPopulator:
    """
    Populate a Qdrant collection with embeddings for PDF chunks using
    an Ollama-based embedding model (`nomic-embed-text`).
    """

    def __init__(
        self,
        data_folder: Path,
        collection_name: str = "pdf_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            data_folder: Folder containing PDF files.
            collection_name: Qdrant collection name.
            chunk_size: Character length of each chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.data_folder = Path(data_folder)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ollama-based embedder (nomic-embed-text)
        self.embedder = Embedder(model_name="nomic-embed-text")
        # Qdrant manager (talks to local Qdrant and stores vectors)
        self.qdrant_manager = QdrantManager()

        print(f"Initialized QdrantPopulator with data folder: {self.data_folder}")
        print(f"Target collection: {self.collection_name}")

    def get_pdf_files(self) -> List[Path]:
        """Return all PDF files under the data folder."""
        if not self.data_folder.exists():
            print(f"Data folder does not exist, creating: {self.data_folder}")
            self.data_folder.mkdir(parents=True, exist_ok=True)
            return []

        pdf_files = sorted(self.data_folder.glob("*.pdf"))
        print(f"\nFound {len(pdf_files)} PDF files:")
        for f in pdf_files:
            print(f"  - {f.name}")
        return pdf_files

    def process_single_pdf(self, pdf_path: Path, verbose: bool = False) -> int:
        """
        Chunk a single PDF, embed all chunks with Ollama, and insert into Qdrant.

        Returns:
            Number of chunks successfully inserted.
        """
        print("\n" + "=" * 80)
        print(f"Processing PDF: {pdf_path.name}")
        print("=" * 80)

        # 1) Chunk PDF
        print("[1/3] Extracting chunks from PDF...")
        try:
            chunks = pdf2chunks(
                pdf_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except Exception as e:
            print(f"  ERROR extracting text from {pdf_path.name}: {e}")
            return 0

        if not chunks:
            print(f"  No text extracted from {pdf_path.name}")
            return 0

        print(f"  Extracted {len(chunks)} chunks")

        if verbose:
            for i, chunk in enumerate(chunks, start=1):
                preview = chunk[:200] + ("..." if len(chunk) > 200 else "")
                print(f"\nChunk {i}/{len(chunks)} ({len(chunk)} chars)")
                print("-" * 80)
                print(preview)

        # 2) Embed chunks via Ollama
        print("\n[2/3] Generating embeddings via Ollama...")
        t0 = time.time()
        try:
            embeddings = self.embedder.embed_texts(chunks, show_progress=True)
        except Exception as e:
            print(f"  ERROR generating embeddings for {pdf_path.name}: {e}")
            return 0

        elapsed = time.time() - t0
        print(f"  Generated {len(embeddings)} embeddings in {elapsed:.2f}s")

        if len(embeddings) != len(chunks):
            print("  ERROR: number of embeddings does not match number of chunks")
            return 0

        # 3) Upsert into Qdrant
        print("\n[3/3] Inserting points into Qdrant...")
        metadata_list = [
            {
                "source": pdf_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]

        try:
            self.qdrant_manager.insert_points_batch(
                embeddings=embeddings,
                collection_name=self.collection_name,
                chunk_texts=chunks,
                metadata_list=metadata_list,
            )
        except Exception as e:
            print(f"  ERROR inserting into Qdrant for {pdf_path.name}: {e}")
            return 0

        print(f"  Successfully inserted {len(chunks)} chunks from {pdf_path.name}")
        return len(chunks)

    def _ensure_collection(self, recreate_collection: bool) -> None:
        """Create or recreate the Qdrant collection with correct vector size."""
        existing = self.qdrant_manager.list_collections()

        if recreate_collection and self.collection_name in existing:
            print(f"\nDeleting existing collection '{self.collection_name}'...")
            self.qdrant_manager.delete_collection(self.collection_name)
            existing = self.qdrant_manager.list_collections()

        if self.collection_name not in existing:
            print(f"\nCreating collection '{self.collection_name}' in Qdrant...")
            # Ask Ollama for one dummy embedding to infer vector size
            dim = self.embedder.get_embedding_dimension()
            self.qdrant_manager.create_collection(name=self.collection_name, vector_size=dim)
        else:
            print(f"\nUsing existing collection '{self.collection_name}'")

    def populate(self, verbose: bool = False, recreate_collection: bool = False) -> None:
        """
        Populate Qdrant with all PDFs under the data folder.

        This is the end-to-end step: PDF → chunks → Ollama embeddings → Qdrant.
        """
        print("\n" + "=" * 80)
        print("STARTING QDRANT POPULATION")
        print("=" * 80)

        pdf_files = self.get_pdf_files()
        if not pdf_files:
            print("\nNo PDF files found. Put PDFs under:")
            print(f"  {self.data_folder}")
            return

        self._ensure_collection(recreate_collection=recreate_collection)

        total_chunks = 0
        processed_files = 0
        t0 = time.time()

        for idx, pdf_path in enumerate(pdf_files, start=1):
            print(f"\n[File {idx}/{len(pdf_files)}] {pdf_path.name}")
            n_chunks = self.process_single_pdf(pdf_path, verbose=verbose)
            if n_chunks > 0:
                total_chunks += n_chunks
                processed_files += 1

        elapsed = time.time() - t0
        print("\n" + "=" * 80)
        print("POPULATION COMPLETE")
        print("=" * 80)
        print(f"Processed files: {processed_files}/{len(pdf_files)}")
        print(f"Total chunks inserted: {total_chunks}")
        print(f"Total time: {elapsed:.2f}s")
        if processed_files:
            print(f"Average time per file: {elapsed / processed_files:.2f}s")

        info = self.qdrant_manager.get_collection_info(self.collection_name)
        if info is not None:
            print(f"\nCollection '{self.collection_name}' stats:")
            try:
                print(f"  Points count: {info.points_count}")
                print(f"  Vector size: {info.config.params.vectors.size}")
            except Exception:
                # Older qdrant-client versions use slightly different fields
                pass


def main() -> None:
    """
    CLI entry point.

    Usage (from project root):
        python -m services.populate_qdrant
    """
    # Resolve ../data relative to this file so it works from any CWD.
    project_root = Path(__file__).resolve().parents[1]
    data_folder = project_root / "data"

    populator = QdrantPopulator(
        data_folder=data_folder,
        collection_name="pdf_documents",
        chunk_size=1000,
        chunk_overlap=200,
    )

    # By default, keep existing collection and append new points.
    populator.populate(verbose=False, recreate_collection=False)


if __name__ == "__main__":
    main()
