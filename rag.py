import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import COLLECTION_NAME
from utils import CHROMA_DIR, EMBEDDING_MODEL_NAME, clean_snippet


FALLBACK_RESPONSE = (
    "The uploaded documents do not contain enough information to answer this question."
)
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever(top_k: int = 4) -> dict[str, Any]:
    return {"vector_store": get_vector_store(), "top_k": top_k}


def get_llm():
    if DEFAULT_LLM_PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            api_key=api_key,
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        temperature=0,
        google_api_key=api_key,
    )


def get_available_sources() -> list[str]:
    vector_store = get_vector_store()
    raw = vector_store.get(include=["metadatas"])
    sources = {
        metadata.get("source")
        for metadata in raw.get("metadatas", [])
        if metadata and metadata.get("source")
    }
    return sorted(sources)


def retrieve_chunks(question: str, retriever: dict[str, Any]) -> list[tuple[Document, float]]:
    vector_store: Chroma = retriever["vector_store"]
    top_k = retriever["top_k"]
    return vector_store.similarity_search_with_relevance_scores(question, k=top_k)


def format_context(results: list[tuple[Document, float]]) -> str:
    blocks = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        blocks.append(
            f"[Source: {source} | Page {page} | Chunk {chunk_id} | Score {score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a careful document question-answering assistant.

Rules:
- Answer only from the retrieved context.
- If the context does not answer the question, say exactly:
  "{FALLBACK_RESPONSE}"
- Keep the answer in the same language as the user's question.
- Include short inline source references where useful, using this format:
  [Source: file.pdf | Page 2]
- Do not use outside knowledge.

Retrieved context:
{context}

Question:
{question}

Answer:
""".strip()


def build_citations(results: list[tuple[Document, float]]) -> list[dict]:
    citations = []
    seen = set()
    for doc, score in results:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "score": score,
                "snippet": clean_snippet(doc.page_content, max_chars=320),
            }
        )
    return citations


def serialize_chunks(results: list[tuple[Document, float]]) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "score": score,
            "text": doc.page_content,
        }
        for doc, score in results
    ]


def answer_question(question: str, retriever: dict[str, Any]) -> dict:
    results = retrieve_chunks(question, retriever)
    if not results:
        return {
            "answer": FALLBACK_RESPONSE,
            "blocked": True,
            "citations": [],
            "retrieved_chunks": [],
        }

    best_score = results[0][1]
    retrieved_chunks = serialize_chunks(results)
    if best_score < MIN_RELEVANCE_SCORE:
        return {
            "answer": FALLBACK_RESPONSE,
            "blocked": True,
            "citations": build_citations(results),
            "retrieved_chunks": retrieved_chunks,
        }

    context = format_context(results)
    llm = get_llm()
    response = llm.invoke(build_prompt(question, context))
    answer = response.content.strip()

    if FALLBACK_RESPONSE.lower() in answer.lower():
        blocked = True
        answer = FALLBACK_RESPONSE
    else:
        blocked = False

    return {
        "answer": answer,
        "blocked": blocked,
        "citations": build_citations(results),
        "retrieved_chunks": retrieved_chunks,
    }

