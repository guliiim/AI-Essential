"""
RAG Agent with Gemini and Qdrant retrieval + cross-encoder reranking.
Supports multiple collections (pdf_documents, excel_documents)
"""

from __future__ import annotations
import os
import time
from typing import List, Dict, Any
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient

from reranker import CrossEncoderReranker


class QdrantRetriever:
    """
    Retrieval tool for Qdrant.
    Supports dynamic collection_name: pdf_documents or excel_documents
    """
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        qdrant_url: str = "http://localhost:6333",
        ollama_url: str = "http://localhost:11434",
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.embedding_model = "nomic-embed-text:latest"

# query to vector
    def _embed(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

# query embedding sent to qdrant, find top k documents
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self._embed(query)
        search_url = f"{self.qdrant_url}/collections/{self.collection_name}/points/search"

        resp = requests.post(
            search_url,
            json={
                "vector": query_embedding,
                "limit": self.top_k,
                "with_payload": True,
                "score_threshold": self.score_threshold,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", [])

        docs: List[Dict[str, Any]] = []
        for hit in results:
            payload = hit.get("payload", {}) if isinstance(hit, dict) else getattr(hit, "payload", {})
            score = hit.get("score") if isinstance(hit, dict) else getattr(hit, "score", 0.0)
            docs.append(
                {
                    "text": payload.get("text", ""),
                    "score": score,
                    "source": payload.get("source", "Unknown"),
                    "chunk_index": payload.get("chunk_index", 0),
                }
            )
        return docs


class RAGAgent:
    """
    Retrieval-Augmented Generation agent with cross-encoder reranking.

    Pipeline:
        1. Embed query with Ollama (nomic-embed-text)
        2. Retrieve top-K candidates from Qdrant (vector similarity)
        3. Rerank candidates with a cross-encoder (true relevance scoring)
        4. Pass top-N reranked docs as context to Gemini for answer generation

    The two-stage retrieve-then-rerank approach improves answer quality
    because vector similarity is fast but approximate, while the cross-encoder
    scores each (query, passage) pair jointly and is far more accurate.
    """
    def __init__(
        self,
        gemini_api_key: str,
        collection_name: str = "pdf_documents",
        model_name: str = "gemini-2.5-flash",
        # Retrieval config
        top_k: int = 10,           # fetch more candidates for reranker to work with
        score_threshold: float = 0.3,
        # Reranker config
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_n: int = 5,   # how many docs to keep after reranking
    ) -> None:
        model_name = os.getenv("GEMINI_MODEL_RAG", model_name)
        self.use_reranker = use_reranker
        self.reranker_top_n = reranker_top_n

        self.retriever = QdrantRetriever(
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.0,
            convert_system_message_to_human=True,
        )

        # Load reranker (downloads model on first run, cached afterwards)
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(model_name=reranker_model)
        else:
            self.reranker = None

        print(f"RAGAgent initialized with:")
        print(f"  - Gemini model:   {model_name}")
        print(f"  - Collection:     {collection_name}")
        print(f"  - top_k:          {top_k}")
        print(f"  - threshold:      {score_threshold}")
        print(f"  - reranker:       {'ON' if use_reranker else 'OFF'}")
        if use_reranker:
            print(f"  - reranker_top_n: {reranker_top_n}")
            print(f"  - reranker_model: {reranker_model}")

    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve and optionally rerank documents for a query."""
        docs = self.retriever.retrieve(query)
        if self.use_reranker and self.reranker and docs:
            docs = self.reranker.rerank(query, docs, top_k=self.reranker_top_n)
        return docs

    def answer_question(self, question: str, verbose: bool = False) -> str:
        if verbose:
            print(f"\n[Retrieval] Question: {question}")

        # Stage 1: Vector retrieval
        raw_docs = self.retriever.retrieve(question)
        if verbose:
            print(f"[Retrieval] Retrieved {len(raw_docs)} candidates from Qdrant")
            for i, d in enumerate(raw_docs, start=1):
                print(f"  {i}. {d['source']} (vector_score={d['score']:.3f}, chunk={d['chunk_index']})")

        # Stage 2: Cross-encoder reranking
        if self.use_reranker and self.reranker and raw_docs:
            docs = self.reranker.rerank(question, raw_docs, top_k=self.reranker_top_n)
            if verbose:
                print(f"\n[Reranker] Top {len(docs)} after reranking:")
                for i, d in enumerate(docs, start=1):
                    print(
                        f"  {i}. {d['source']} "
                        f"(rerank_score={d['rerank_score']:.3f}, "
                        f"vector_score={d['score']:.3f})"
                    )
        else:
            docs = raw_docs

        # Stage 3: Build context and generate answer
        if not docs:
            context = "No relevant information was found in the indexed documents."
        else:
            pieces = [
                f"[Document {i} | source={d['source']} | score={d.get('rerank_score', d['score']):.3f}]\n{d['text']}"
                for i, d in enumerate(docs, start=1)
            ]
            context = "\n\n".join(pieces)

        prompt = f"""You are a helpful assistant that answers questions using the provided context only.

                Context:
                {context}

                Question: {question}

                Instructions:
                - Use only the information in the context when answering.
                - If the context is insufficient, say exactly:
                  "I don't have enough information in the provided documents to answer this question."
                - Do NOT hallucinate or invent facts not supported by the context.

                Answer:"""

        try:
            for attempt in range(5):
                try:
                    resp = self.llm.invoke(prompt)
                    answer = resp.content if hasattr(resp, "content") else str(resp)
                    return answer.strip()
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        time.sleep(2 ** attempt)
                        continue
                    raise
        except Exception as e:
            print(f"Error generating answer with Gemini: {e}")
            return f"Error generating answer: {e}"

    def answer_question_simple(self, question: str) -> str:
        return self.answer_question(question, verbose=False)
