from langchain_core.documents import Document

from rag import FALLBACK_RESPONSE, get_llm, get_vector_store
from utils import clean_snippet


def _retrieve_for_document(source: str, topic: str, k: int = 4) -> list[tuple[Document, float]]:
    vector_store = get_vector_store()
    return vector_store.similarity_search_with_relevance_scores(
        topic,
        k=k,
        filter={"source": source},
    )


def _format_evidence(label: str, results: list[tuple[Document, float]]) -> str:
    parts = [f"{label} evidence:"]
    for doc, score in results:
        parts.append(
            f"[Source: {doc.metadata.get('source')} | Page {doc.metadata.get('page')} | "
            f"Chunk {doc.metadata.get('chunk_id')} | Score {score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _serialize_evidence(results: list[tuple[Document, float]]) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "score": score,
            "text": clean_snippet(doc.page_content, max_chars=900),
        }
        for doc, score in results
    ]


def compare_documents(doc_a: str, doc_b: str, topic: str) -> dict:
    results_a = _retrieve_for_document(doc_a, topic)
    results_b = _retrieve_for_document(doc_b, topic)

    if not results_a or not results_b:
        return {
            "label": "Not enough evidence",
            "reasoning": FALLBACK_RESPONSE,
            "evidence": _serialize_evidence(results_a + results_b),
        }

    prompt = f"""
You are comparing two uploaded documents for factual agreement.

Topic or claim:
{topic}

Decide one label only:
- Agreement
- Contradiction
- Partial conflict
- Not enough evidence

Use only the evidence below. Be conservative. Explain the reasoning briefly and cite pages.

{_format_evidence("Document A", results_a)}

---

{_format_evidence("Document B", results_b)}

Return this format:
Label: <one label>
Reasoning: <brief explanation with citations>
""".strip()

    response = get_llm().invoke(prompt)
    text = response.content.strip()

    label = "Not enough evidence"
    reasoning = text
    for line in text.splitlines():
        if line.lower().startswith("label:"):
            label = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    return {
        "label": label,
        "reasoning": reasoning,
        "evidence": _serialize_evidence(results_a + results_b),
    }

