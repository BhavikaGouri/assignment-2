from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(slots=True)
class Document:
    source_type: str
    source_name: str
    text: str
    metadata: dict[str, str]


def ingest_documents(input_file: Path | None, urls: Iterable[str], logger) -> list[Document]:
    documents: list[Document] = []

    if input_file is not None:
        try:
            documents.append(_ingest_file(input_file))
        except Exception:
            logger.exception('Skipping file input %s because it could not be ingested.', input_file)

    for url in urls:
        try:
            documents.append(_ingest_url(url))
        except Exception:
            logger.exception('Skipping URL input %s because it could not be ingested.', url)

    return documents


def _ingest_file(path: Path) -> Document:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == '.txt':
        text = path.read_text(encoding='utf-8', errors='replace')
    elif suffix == '.pdf':
        text = _read_pdf(path)
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}')

    return Document(source_type='file', source_name=path.name, text=text, metadata={'path': str(path)})


def _read_pdf(path: Path) -> str:
    parts: list[str] = []
    with path.open('rb') as handle:
        reader = PdfReader(handle)
        for page in reader.pages:
            parts.append(page.extract_text() or '')
    return '\n'.join(parts)


def _ingest_url(url: str) -> Document:
    with httpx.Client(timeout=30, follow_redirects=True, headers={'User-Agent': 'llm-pipeline/1.0'}) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            text = _read_pdf_bytes(response.content)
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
                tag.decompose()
            text = soup.get_text('\n')

    return Document(source_type='url', source_name=url, text=text, metadata={'url': url})


def _read_pdf_bytes(content: bytes) -> str:
    parts: list[str] = []
    reader = PdfReader(io.BytesIO(content))
    for page in reader.pages:
        parts.append(page.extract_text() or '')
    return '\n'.join(parts)
