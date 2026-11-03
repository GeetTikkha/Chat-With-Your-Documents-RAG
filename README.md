# Chat With Your Documents — RAG Application

A document Q&A app using Retrieval-Augmented Generation (RAG).

## Features
- Upload PDF, DOCX and TXT files
- Extract text and split it into overlapping chunks
- Generate embeddings with Sentence Transformers
- Store and search vectors using FAISS
- Retrieve relevant passages and send them as context to an LLM
- Display answers and retrieved passages

## Architecture
Document → Text extraction → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer

## Run locally
1. Install Python 3.10+ and clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Add your own API key to `.env` (never commit this file).
5. Run: `streamlit run app.py`

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (runs locally after download).
Answer generation: OpenAI API, model configured with `OPENAI_MODEL` (default `gpt-4o-mini`).
API usage may incur charges. The embedding and retrieval demonstration in the notebook can run without an API key.

## Deploy
Deploy this repo on Streamlit Community Cloud. Add `OPENAI_API_KEY` and `OPENAI_MODEL` in app Secrets. Never put secrets in GitHub.

## Limitations and responsible use
Scanned PDFs may require OCR. Similarity scores are ranking signals, not probabilities. RAG can still miss relevant text or generate incorrect answers. Verify answers against source documents and do not upload confidential files to unapproved services.
