"""
Task 4: Live Demo — Interactive RAG Chatbot (local Ollama, no API limits)
Routes general questions directly to Ollama,
RAG questions through the full pipeline (Qdrant + reranker + Ollama).

Run:
    python services/demo.py
"""

import os
from agent import RAGAgent
from langchain_ollama import ChatOllama


# Keywords that suggest the question is about the indexed PDF documents
RAG_KEYWORDS = [
    "rag", "retrieval", "augmented", "generation", "chunk", "embedding",
    "transformer", "attention", "llm", "language model", "paper", "research",
    "document", "pdf", "rerank", "vector", "qdrant", "hallucination",
    "cottonbot", "graphrag", "treeqa", "modular", "naive", "advanced",
    "bias", "privacy", "misinformation", "knowledge", "graph", "ontology",
    "according", "summarize", "explain", "what does", "how does",
]


def is_rag_question(text: str) -> bool:
    """Return True if the question is likely about the indexed PDF documents."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in RAG_KEYWORDS)


def main() -> None:
    MODEL_NAME = "llama3.2:3b"          # change to "llama3.2:3b" if you have it
    COLLECTION = os.getenv("RAG_COLLECTION", "pdf_documents")

    print("=" * 60)
    print("       RAG CHATBOT - LIVE DEMO")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME} (local Ollama)")
    print(f"  Collection : {COLLECTION}")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    # Full RAG pipeline for document questions
    rag_agent = RAGAgent(
        collection_name=COLLECTION,
        model_name=MODEL_NAME,
        use_reranker=True,
        top_k=10,
        score_threshold=0.3,
        reranker_top_n=5,
    )

    # Direct Ollama for general questions (faster, no Qdrant lookup)
    direct_llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.7,
        timeout=300,
    )

    print("\nChatbot is ready!\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if is_rag_question(user_input):
            # Full RAG pipeline
            print("[RAG] Searching documents...")
            answer = rag_agent.answer_question(user_input, verbose=False)
            if not answer or answer.strip() in ("None", ""):
                answer = "Sorry, I could not find a relevant answer in the documents."
        else:
            # Direct Ollama — fast, no Qdrant
            print("[Direct] Answering...")
            try:
                resp = direct_llm.invoke(user_input)
                answer = resp.content if hasattr(resp, "content") else str(resp)
                answer = answer.strip()
            except Exception as e:
                answer = f"Error: {e}"

        print(f"Assistant: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
# """
# Task 4: Live Demo — Interactive RAG Chatbot
# Routes general questions directly to Gemini,
# RAG-specific questions through the full pipeline.
# """
#
# import os
# from agent import RAGAgent
# from langchain_google_genai import ChatGoogleGenerativeAI
#
#
# # Keywords that suggest the question is about the indexed documents
# RAG_KEYWORDS = [
#     "rag", "retrieval", "augmented", "generation", "chunk", "embedding",
#     "transformer", "attention", "llm", "language model", "paper", "research",
#     "document", "pdf", "rerank", "vector", "qdrant", "hallucination",
#     "cottonbot", "graphrag", "treeqa", "modular", "naive", "advanced",
#     "bias", "privacy", "misinformation", "knowledge", "graph", "ontology",
# ]
#
#
# def is_rag_question(text: str) -> bool:
#     """Return True if the question is likely about the indexed PDF documents."""
#     text_lower = text.lower()
#     return any(keyword in text_lower for keyword in RAG_KEYWORDS)
#
#
# def main() -> None:
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("ERROR: Set GOOGLE_API_KEY environment variable.")
#         raise SystemExit(1)
#
#     collection_name = os.getenv("RAG_COLLECTION", "pdf_documents")
#     model_name = "gemini-2.5-flash"
#
#     print("=" * 60)
#     print("       RAG CHATBOT - LIVE DEMO")
#     print("=" * 60)
#     print(f"Collection : {collection_name}")
#     print(f"Model      : {model_name}")
#     print("Type your question and press Enter.")
#     print("Type 'quit' or 'exit' to stop.")
#     print("=" * 60)
#
#     # Full RAG pipeline (used for document-related questions)
#     rag_agent = RAGAgent(
#         gemini_api_key=api_key,
#         collection_name=collection_name,
#         model_name=model_name,
#         use_reranker=True,
#         top_k=10,
#         score_threshold=0.3,
#         reranker_top_n=5,
#     )
#
#     # Direct Gemini (used for general questions — much faster)
#     direct_llm = ChatGoogleGenerativeAI(
#         model=model_name,
#         google_api_key=api_key,
#         temperature=0.7,
#         convert_system_message_to_human=True,
#     )
#
#     print("\nChatbot is ready!\n")
#
#     while True:
#         try:
#             user_input = input("You: ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print("\n\nGoodbye!")
#             break
#
#         if user_input.lower() in ("quit", "exit", "q", "bye"):
#             print("\nGoodbye!")
#             break
#
#         if not user_input:
#             continue
#
#         # Route the question
#         if is_rag_question(user_input):
#             # Use full RAG pipeline
#             print("[RAG] Retrieving from documents...")
#             answer = rag_agent.answer_question(user_input, verbose=False)
#             if not answer or answer.strip() in ("None", ""):
#                 answer = "Sorry, I could not find a relevant answer in the documents."
#             print(f"Assistant: {answer}")
#         else:
#             # Answer directly with Gemini — no Qdrant, much faster
#             print("[Direct] Answering directly...")
#             try:
#                 resp = direct_llm.invoke(user_input)
#                 answer = resp.content if hasattr(resp, "content") else str(resp)
#                 answer = answer.strip()
#             except Exception as e:
#                 answer = f"Error: {e}"
#             print(f"Assistant: {answer}")
#
#         print("-" * 60)
#
#
# if __name__ == "__main__":
#     main()
#--------------------------------------------------
# import os
# from agent import RAGAgent
#
#
# def main() -> None:
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("ERROR: Set GOOGLE_API_KEY environment variable.")
#         raise SystemExit(1)
#
#     collection_name = os.getenv("RAG_COLLECTION", "pdf_documents")
#
#     print("=" * 60)
#     print("       RAG CHATBOT - LIVE DEMO")
#     print("=" * 60)
#     print(f"Collection: {collection_name}")
#     print("Type your question and press Enter.")
#     print("Type 'quit' or 'exit' to stop.")
#     print("=" * 60)
#
#     rag_agent = RAGAgent(
#         gemini_api_key=api_key,
#         collection_name=collection_name,
#         model_name="gemini-2.0-flash",   # faster than 2.5-flash
#         use_reranker=True,
#         top_k=10,
#         score_threshold=0.3,
#         reranker_top_n=5,
#     )
#
#     print("\nChatbot is ready!\n")
#
#     while True:
#         try:
#             user_input = input("You: ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print("\n\nGoodbye!")
#             break
#
#         if user_input.lower() in ("quit", "exit", "q", "bye"):
#             print("\nGoodbye!")
#             break
#
#         if not user_input:
#             continue
#
#         print("Assistant: thinking...")
#         answer = rag_agent.answer_question(user_input, verbose=False)
#
#         # Fix None response
#         if not answer or answer.strip() == "None":
#             answer = "Sorry, I could not generate an answer. Please try again."
#
#         print(f"Assistant: {answer}")
#         print("-" * 60)
#
#
# if __name__ == "__main__":
#     main()
#-------------------------------------------------------------------
# """
# Task 4: Live Demo — Interactive RAG Chatbot
# ============================================
# Run in terminal:
#     python services/demo.py
# """
#
# import os
# from agent import RAGAgent
#
#
# def main() -> None:
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("ERROR: Set GOOGLE_API_KEY environment variable.")
#         raise SystemExit(1)
#
#     collection_name = os.getenv("RAG_COLLECTION", "pdf_documents")
#
#     print("=" * 60)
#     print("       RAG CHATBOT - LIVE DEMO")
#     print("=" * 60)
#     print(f"Collection: {collection_name}")
#     print("Type your question and press Enter.")
#     print("Type 'quit' or 'exit' to stop.")
#     print("=" * 60)
#
#     # Initialize RAG agent
#     rag_agent = RAGAgent(
#         gemini_api_key=api_key,
#         collection_name=collection_name,
#         use_reranker=True,
#         top_k=10,
#         score_threshold=0.3,
#         reranker_top_n=5,
#     )
#
#     print("\nChatbot is ready!\n")
#
#     while True:
#         # Get user input
#         try:
#             user_input = input("You: ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print("\n\nGoodbye!")
#             break
#
#         # Exit commands
#         if user_input.lower() in ("quit", "exit", "q", "bye"):
#             print("\nGoodbye!")
#             break
#
#         # Skip empty input
#         if not user_input:
#             continue
#
#         # Get answer
#         print("\nAssistant: thinking...", end="\r")
#         answer = rag_agent.answer_question(user_input, verbose=False)
#         print(f"Assistant: {answer}")
#         print("-" * 60)
#
#
# if __name__ == "__main__":
#     main()