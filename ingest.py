from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import CHROMA_DIR, EMBEDDING_MODEL_NAME, ensure_directories


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "uploaded_pdf_chunks"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_pdf_pages(pdf_paths: Iterable[Path]) -> list[Document]:
    pages: list[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        loaded_pages = loader.load()
        for page in loaded_pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["file_path"] = str(pdf_path)
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
        pages.extend(loaded_pages)
    return pages


def chunk_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def ingest_pdfs(pdf_paths: Iterable[Path]) -> dict:
    ensure_directories()
    pdf_paths = [Path(path) for path in pdf_paths]
    if not pdf_paths:
        raise ValueError("No PDF files were provided for ingestion.")

    pages = load_pdf_pages(pdf_paths)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.add_documents(chunks)

    return {
        "files": len(pdf_paths),
        "pages": len(pages),
        "chunks": len(chunks),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index PDF files into ChromaDB.")
    parser.add_argument("pdfs", nargs="+", help="PDF file paths to ingest")
    args = parser.parse_args()
    result = ingest_pdfs([Path(path) for path in args.pdfs])
    print(result)

