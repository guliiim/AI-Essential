from pathlib import Path
import pandas as pd
import time
from embedding_manager import Embedder
from qdrant_manager import QdrantManager

class ExcelQdrantPopulator:
    """
    Populate Qdrant with embeddings from an Excel file containing
    question-answer pairs.
    """
    def __init__(self, excel_path: Path, collection_name: str = "excel_documents"):
        self.excel_path = excel_path
        self.collection_name = collection_name

        # Ollama-based embedder
        self.embedder = Embedder(model_name="nomic-embed-text")
        # Qdrant manager
        self.qdrant_manager = QdrantManager()

        print(f"Initialized ExcelQdrantPopulator")
        print(f"  Excel file: {self.excel_path}")
        print(f"  Target collection: {self.collection_name}")

    def _ensure_collection(self, recreate_collection: bool = False):
        """Create or reuse collection in Qdrant."""
        existing = self.qdrant_manager.list_collections()

        if recreate_collection and self.collection_name in existing:
            print(f"Deleting existing collection '{self.collection_name}'...")
            self.qdrant_manager.delete_collection(self.collection_name)
            existing = self.qdrant_manager.list_collections()

        if self.collection_name not in existing:
            print(f"Creating collection '{self.collection_name}'...")
            dim = self.embedder.get_embedding_dimension()
            self.qdrant_manager.create_collection(name=self.collection_name, vector_size=dim)
        else:
            print(f"Using existing collection '{self.collection_name}'")

    def populate(self, recreate_collection: bool = False):
        """Read Excel, embed each answer, and upsert to Qdrant."""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")

        df = pd.read_excel(self.excel_path)

        # Detect columns
        cols_lower = {c.lower(): c for c in df.columns}

        def _find_col(prefix: str):
            for c in df.columns:
                if c.lower().strip().startswith(prefix):
                    return c
            raise KeyError(prefix)

        try:
            question_col = _find_col("question")
            answer_col = _find_col("answer")
        except KeyError as e:
            print("Available columns in Excel:", list(df.columns))
            raise SystemExit(f"Could not find a column starting with {e}")

        self._ensure_collection(recreate_collection=recreate_collection)

        total_rows = len(df)
        print(f"\nEmbedding and inserting {total_rows} rows from Excel...\n")
        start_time = time.time()
        embeddings = []
        texts = []

        for i, row in df.iterrows():
            text = str(row[answer_col])
            try:
                emb = self.embedder.embed_text(text)
            except Exception as e:
                print(f"Error embedding row {i}: {e}")
                continue
            embeddings.append(emb)
            texts.append(text)

        metadata_list = [
            {"question": str(row[question_col]), "row_index": i}
            for i, row in df.iterrows()
        ]

        self.qdrant_manager.insert_points_batch(
            embeddings=embeddings,
            collection_name=self.collection_name,
            chunk_texts=texts,
            metadata_list=metadata_list,
        )

        elapsed = time.time() - start_time
        print(f"\nFinished inserting {len(texts)} points in {elapsed:.2f}s")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    excel_path = project_root / "data" / "RAG Documents.xlsx"

    populator = ExcelQdrantPopulator(
        excel_path=excel_path,
        collection_name="excel_documents",
    )

    populator.populate(recreate_collection=False)
