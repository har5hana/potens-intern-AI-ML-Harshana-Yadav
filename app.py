import html

import streamlit as st
from dotenv import load_dotenv

from conflict_checker import compare_documents
from ingest import ingest_pdfs
from rag import answer_question, get_available_sources, get_retriever
from utils import (
    CHROMA_DIR,
    DATA_DIR,
    UPLOAD_DIR,
    ensure_directories,
    save_uploaded_files,
)


load_dotenv()
ensure_directories()

st.set_page_config(
    page_title="Multi-Document RAG Q&A",
    page_icon="📚",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1180px; }
    .stButton > button { border-radius: 6px; }
    .citation-box {
        padding: 0.8rem;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        background: #fbfbfb;
        margin-bottom: 0.7rem;
    }
    .small-muted { color: #666; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def sidebar_upload() -> None:
    st.sidebar.header("Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.sidebar.button("Save and index PDFs", type="primary"):
            with st.spinner("Extracting text, chunking, and building embeddings..."):
                saved_paths = save_uploaded_files(uploaded_files)
                stats = ingest_pdfs(saved_paths)
            st.sidebar.success(
                f"Indexed {stats['files']} file(s), {stats['pages']} page(s), "
                f"and {stats['chunks']} chunk(s)."
            )

    st.sidebar.divider()
    st.sidebar.caption(f"Uploads: `{UPLOAD_DIR}`")
    st.sidebar.caption(f"Chroma DB: `{CHROMA_DIR}`")
    st.sidebar.caption(f"Data cache: `{DATA_DIR}`")


def render_answer(result: dict) -> None:
    if result.get("blocked"):
        st.warning(result["answer"])
    else:
        st.subheader("Answer")
        st.write(result["answer"])

    st.subheader("Citations")
    citations = result.get("citations", [])
    if not citations:
        st.info("No citations available.")
        return

    for citation in citations:
        source = html.escape(str(citation["source"]))
        page = html.escape(str(citation["page"]))
        chunk_id = html.escape(str(citation["chunk_id"]))
        snippet = html.escape(citation["snippet"])
        st.markdown(
            f"""
            <div class="citation-box">
                <strong>[Source: {source} | Page {page} | Chunk {chunk_id}]</strong>
                <div class="small-muted">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def qa_tab() -> None:
    st.title("Multi-Document RAG Q&A System with Citations")
    st.caption(
        "Upload PDFs, ask questions, and get grounded answers with source references."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        question = st.text_input(
            "Ask a question about the uploaded documents",
            placeholder="Example: What does the report say about model limitations?",
        )
    with col_b:
        top_k = st.slider("Top-k chunks", min_value=2, max_value=8, value=4)

    debug_mode = st.toggle("Show retrieved chunks", value=True)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            retriever = get_retriever(top_k=top_k)
            with st.spinner("Retrieving evidence and generating answer..."):
                result = answer_question(question, retriever)
            render_answer(result)

            if debug_mode:
                st.subheader("Retrieved Chunks")
                for idx, chunk in enumerate(result.get("retrieved_chunks", []), start=1):
                    label = (
                        f"{idx}. {chunk['source']} | Page {chunk['page']} | "
                        f"Chunk {chunk['chunk_id']} | Score {chunk['score']:.3f}"
                    )
                    with st.expander(label):
                        st.write(chunk["text"])
        except Exception as exc:
            st.error(str(exc))


def conflict_tab() -> None:
    st.header("Document Conflict Checker")
    st.caption(
        "Select two indexed documents and compare their claims on a specific topic."
    )

    sources = get_available_sources()
    if len(sources) < 2:
        st.info("Index at least two PDF documents to use the conflict checker.")
        return

    left, right = st.columns(2)
    with left:
        doc_a = st.selectbox("First document", sources, key="doc_a")
    with right:
        doc_b = st.selectbox("Second document", sources, key="doc_b")

    topic = st.text_input(
        "Claim or topic to compare",
        placeholder="Example: What do the documents say about privacy risks?",
    )

    if st.button("Compare documents", disabled=not topic.strip() or doc_a == doc_b):
        try:
            with st.spinner("Retrieving claims and checking for conflicts..."):
                result = compare_documents(doc_a, doc_b, topic)
            st.subheader(result["label"])
            st.write(result["reasoning"])

            st.subheader("Evidence")
            for item in result["evidence"]:
                with st.expander(
                    f"{item['source']} | Page {item['page']} | Chunk {item['chunk_id']}"
                ):
                    st.write(item["text"])
        except Exception as exc:
            st.error(str(exc))


sidebar_upload()
tab_qa, tab_conflict = st.tabs(["Q&A", "Conflict Checker"])
with tab_qa:
    qa_tab()
with tab_conflict:
    conflict_tab()
