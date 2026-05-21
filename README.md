# Multi-Document RAG Q&A System with Citations

An internship-style AI/ML project where a user uploads multiple PDF files, asks questions, and receives answers grounded only in the uploaded documents. The app uses sentence-transformer embeddings, ChromaDB retrieval, and either Gemini or Groq for final answer generation.

The goal is practical: a clean local RAG system that is easy to run, easy to inspect, and honest about uncertainty.

## What This Project Does

- Upload and store multiple PDF documents locally.
- Extract text from PDFs page by page.
- Split text into overlapping chunks.
- Embed chunks using `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors in a local ChromaDB collection.
- Retrieve the most relevant chunks for a question.
- Ask an LLM to answer only from retrieved context.
- Show citations with source file, page number, chunk id, and snippet.
- Refuse to answer when retrieved evidence is weak.
- Support multilingual questions by instructing the LLM to answer in the same language.
- Compare two documents for agreement, contradiction, or partial conflict.

## Architecture Overview

```text
PDF Uploads
    |
    v
uploaded_docs/
    |
    v
PyPDFLoader extracts page text
    |
    v
RecursiveCharacterTextSplitter
chunk size = 500, overlap = 100
    |
    v
SentenceTransformer embeddings
    |
    v
ChromaDB local vector store
    |
    v
Question -> similarity search -> top-k chunks
    |
    v
Gemini/Groq LLM answers from retrieved context only
    |
    v
Answer + citations + retrieved chunk debug view
```

## Project Structure

```text
project/
├── app.py
├── api.py
├── ingest.py
├── rag.py
├── conflict_checker.py
├── utils.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   └── sample_prompts.md
├── chroma_db/
└── uploaded_docs/
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
copy .env.example .env
```

Then edit `.env` and add either a Gemini key or a Groq key.

For Gemini:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

For Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Optional API server:

```bash
uvicorn api:app --reload
```

## How To Use

1. Open the app in the browser.
2. Upload one or more PDF files from the sidebar.
3. Click **Save and index PDFs**.
4. Ask a question in the Q&A tab.
5. Review the answer, citations, and retrieved chunks.
6. Use the Conflict Checker tab to compare two indexed documents on a claim or topic.

## How Retrieval Works

During ingestion, every PDF page is loaded with metadata:

- source file name
- local file path
- page number

The text is split into chunks and embedded. At question time, the user question is also embedded, and ChromaDB returns the top-k most similar chunks.

The retrieved chunks are passed to the LLM as the only allowed context. The prompt explicitly tells the model not to use outside knowledge. The app also shows retrieved chunks in the UI so the user can inspect what evidence was used.

## Chunking Strategy

This project uses:

- chunk size: `500`
- chunk overlap: `100`

A 500-character chunk is small enough to retrieve focused evidence, but large enough to keep nearby sentences together. The 100-character overlap reduces the chance that an important sentence gets split awkwardly between two chunks.

This is a simple, student-friendly strategy. It is not perfect. Some PDFs have tables, multi-column layouts, or scanned pages, and basic text extraction may not preserve those well.

## Hallucination Prevention

The app uses three layers of protection:

1. Retrieval threshold: if the best similarity score is below `MIN_RELEVANCE_SCORE`, the app refuses to answer.
2. Grounded prompt: the LLM is instructed to answer only from retrieved context.
3. Fallback response: when evidence is missing, the app returns:

```text
The uploaded documents do not contain enough information to answer this question.
```

This does not make hallucination impossible, but it makes the system more conservative and easier to debug.

## Citations

Every answer includes a citations section. Each citation contains:

- PDF file name
- page number
- chunk id
- short snippet

Example:

```text
[Source: attention_paper.pdf | Page 4 | Chunk 12]
```

The snippets are intentionally short. They help the user verify the answer without flooding the UI.

## Multilingual Support

The embedding model supports many semantic matches reasonably well, and the prompt tells the LLM to answer in the same language as the question.

What works well:

- Asking English documents questions in English.
- Asking simple multilingual questions when the relevant evidence is clear.
- Getting answers in Hindi, Spanish, French, and similar common languages if the LLM supports them.

What is imperfect:

- Cross-lingual retrieval can be weaker than same-language retrieval.
- For a stronger production version, I would use a multilingual embedding model such as `intfloat/multilingual-e5-base`.

## Document Conflict Checker

The conflict checker is a stretch feature. It lets the user select two indexed documents and enter a claim or topic.

The system retrieves evidence from each document separately, then asks the LLM to classify the relationship as:

- Agreement
- Contradiction
- Partial conflict
- Not enough evidence

This is useful for quick document comparison, but it should be treated as an assistant, not a final legal or scientific judgment.

## Example Queries

- What are the main findings of these documents?
- What limitations does the paper mention?
- Which document discusses privacy risks?
- Summarize the evaluation methodology.
- इस दस्तावेज़ में मुख्य निष्कर्ष क्या हैं?
- Do the two documents agree about deployment risks?

More examples are in `data/sample_prompts.md`.

## Screenshots

Add screenshots here after running locally:

```text
screenshots/upload_and_index.png
screenshots/question_answer_with_citations.png
screenshots/retrieved_chunks_debug.png
screenshots/conflict_checker.png
```

## Tradeoffs

I kept this project intentionally simple.

What works well:

- Clear local workflow.
- Easy-to-read code.
- Persistent vector store.
- Debug mode for retrieval inspection.
- Conservative fallback behavior.

What is imperfect:

- No OCR for scanned PDFs.
- No user authentication.
- No advanced citation verification after generation.
- Chroma collections are appended to when indexing; duplicate uploads can create duplicate chunks.
- PDF table extraction is basic.

These tradeoffs are acceptable for an internship evaluation because the core RAG flow is visible and understandable.

## Future Improvements

- Add OCR with Tesseract or a document parsing API.
- Add duplicate document detection.
- Add a reset-index button.
- Add reranking with a cross-encoder.
- Add per-document filters in Q&A.
- Add evaluation scripts with expected answers.
- Add a lightweight FastAPI endpoint for programmatic Q&A.

## Optional FastAPI Layer

The main deliverable is Streamlit because it gives a complete upload-and-query experience quickly. A small FastAPI layer is also included for programmatic use.

Available endpoints:

```text
POST /ingest
POST /ask
POST /compare
GET /sources
GET /health
```

This is intentionally lightweight. It reuses the same ingestion, retrieval, and conflict-checking functions as the UI.
