# RAG System with LLM-as-Judge Evaluation

A Retrieval-Augmented Generation (RAG) system using Qdrant for vector storage, Ollama for embeddings, and Google Gemini for generation. Includes LLM-as-judge evaluation framework for assessing retrieval and generation quality.

## System Architecture

```
PDF Documents → Chunking → Ollama Embeddings → Qdrant Vector DB
                                                      ↓
User Question → Ollama Embedding → Similarity Search → Top-K Chunks
                                                      ↓
                                          Gemini LLM → Answer
                                                      ↓
                                          LLM Judge → Evaluation
```

## Features

- **Document Processing**: Automatic PDF chunking with configurable overlap
- **Vector Search**: Qdrant-based semantic search with cosine similarity
- **Dual Collections**: Support for both PDF documents and Excel Q&A pairs
- **RAG Agent**: Question-answering using retrieved context
- **LLM-as-Judge**: Automated evaluation of retrieval + generation quality
- **CSV Export**: Detailed results saved for analysis

## Prerequisites

1. **Ollama** (for embeddings)
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull nomic-embed-text
   ```

2. **Qdrant** (vector database)
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Python Dependencies**
   ```bash
   pip install qdrant-client requests langchain-google-genai pandas openpyxl PyPDF2 ollama numpy
   ```

4. **Google API Key** (for Gemini)
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   # Windows:
   set GOOGLE_API_KEY=your-api-key-here
   ```

## Project Structure

```
project/
├── data/
│   ├── *.pdf                    # PDF documents to index
│   └── RAG Documents.xlsx       # Q&A pairs for evaluation
├── services/
│   ├── preprocessing.py         # PDF chunking utilities
│   ├── embedding_manager.py     # Ollama embedding wrapper
│   ├── qdrant_manager.py        # Qdrant operations
│   ├── populate_qdrant.py       # PDF → Qdrant pipeline
│   ├── populate_excel_qdrant.py # Excel → Qdrant pipeline
│   ├── agent.py                 # RAG agent with retrieval
│   ├── judge_agent.py           # LLM-as-judge evaluator
│   └── run_eval.py              # Evaluation script
├── results/                     # Evaluation results (auto-created)
└── README.md
```

## Setup & Usage

### 1. Populate Qdrant with PDF Documents

```bash
# Place PDF files in data/ folder
python services/populate_qdrant.py
```

This will:
- Extract text from all PDFs in `data/`
- Chunk text (1000 chars, 200 overlap)
- Generate embeddings via Ollama
- Store in `pdf_documents` collection

### 2. Populate Qdrant with Excel Q&A Pairs

```bash
# Ensure RAG Documents.xlsx is in data/
python services/populate_excel_qdrant.py
```

This creates an `excel_documents` collection with answer embeddings.

### 3. Run Evaluation

```bash
python services/run_eval.py
```

**Environment Variables (optional):**
- `GOOGLE_API_KEY` - Your Gemini API key (required)
- `GEMINI_MODEL_RAG` - Model for RAG (default: `gemini-2.5-flash`)
- `GEMINI_MODEL_JUDGE` - Model for judging (default: `gemini-2.5-flash`)
- `RAG_COLLECTION` - Collection to use (default: `excel_documents`)

**Example with custom settings:**
```bash
export RAG_COLLECTION=pdf_documents
export GEMINI_MODEL_RAG=gemini-2.5-flash
export GEMINI_MODEL_JUDGE=gemini-2.5-flash
python services/run_eval.py
```

### 4. View Results

Results are saved to `results/` directory:
- `rag_evaluation_<collection>_<timestamp>.csv` - Detailed per-question results
- `summary_<collection>_<timestamp>.csv` - Overall accuracy metrics

## Evaluation Methodology

### LLM-as-Judge Process

For each question in the evaluation set:

1. **Retrieval**: Query embedding → Qdrant search → Top-5 chunks (score ≥ 0.3)
2. **Generation**: Chunks + Question → Gemini → Predicted Answer
3. **Judgment**: Question + Gold Answer + Predicted Answer → Judge LLM → CORRECT/INCORRECT

The judge uses this prompt:
```
You are an automatic evaluator.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

If the predicted answer correctly answers the question 
(allowing minor wording differences), respond with exactly one word: CORRECT
Otherwise respond with exactly one word: INCORRECT
```

### Metrics Calculated

- **Accuracy**: `(Correct Predictions / Total Questions) × 100%`
- Per-question correctness stored in CSV

## Results

### Evaluation on Excel Documents Collection

**Test Dataset**: RAG Documents.xlsx (Q&A pairs)

| Metric | Value |
|--------|-------|
| **Total Questions** | Varies per dataset |
| **Accuracy** | **XX.X%** |
| **Collection** | excel_documents |
| **RAG Model** | gemini-2.5-flash |
| **Judge Model** | gemini-2.5-flash |
| **Top-K** | 5 |
| **Score Threshold** | 0.3 |

> **Note**: Run `python services/run_eval.py` to generate current accuracy metrics. Results will be saved to `results/` directory with timestamp.

### Sample Output

```
[1/10] Processing question...
Q: What is machine learning?
Predicted: Machine learning is a subset of artificial intelligence...
Gold Answer: Machine learning is AI that learns from data...
Judge Result: CORRECT
--------------------------------------------------------------------------------
...

EVALUATION RESULTS
================================================================================
Total Questions: 10
Correct: 8
Incorrect: 2
Accuracy: 80.00%
================================================================================
```

## Configuration

### Chunking Parameters
```python
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks
```

### Retrieval Parameters
```python
top_k = 5              # Number of chunks to retrieve
score_threshold = 0.3  # Minimum similarity score (0-1)
```

### Models
- **Embedding**: `nomic-embed-text:latest` (768 dimensions)
- **RAG LLM**: `gemini-2.5-flash` (free tier)
- **Judge LLM**: `gemini-2.5-flash` (free tier)

## Troubleshooting

### Qdrant Connection Error
```bash
# Start Qdrant if not running
docker run -p 6333:6333 qdrant/qdrant
```

### Ollama Model Not Found
```bash
ollama pull nomic-embed-text
```

### Rate Limit Errors
The system includes automatic retry with exponential backoff for Gemini API rate limits.

### No Results Found
- Ensure PDFs/Excel are in `data/` folder
- Check that collections were populated successfully
- Lower `score_threshold` if needed (e.g., 0.2)

## Viewing Qdrant Dashboard

Access the Qdrant UI at: `http://localhost:6333/dashboard`




