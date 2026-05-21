import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_directories() -> None:
    for directory in [UPLOAD_DIR, CHROMA_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9_. -]", "_", name).strip()


def save_uploaded_files(uploaded_files) -> list[Path]:
    ensure_directories()
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target_path = UPLOAD_DIR / safe_filename(uploaded_file.name)
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def clean_snippet(text: str, max_chars: int = 300) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."

