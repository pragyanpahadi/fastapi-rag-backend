import fitz
import tiktoken
from fastapi import HTTPException
from typing import List


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    if filename.endswith('.txt'):
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid text encoding. Must be UTF-8.")

    elif filename.endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    else:
        raise HTTPException(
            status_code=400, detail="Unsupported file extension. Only .txt and .pdf allowed.")


def chunk_by_tokens(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))

        if i + chunk_size >= len(tokens):
            break

    return chunks


def chunk_by_recursive_characters(text: str, max_chunk_len: int = 1500) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) > max_chunk_len:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
