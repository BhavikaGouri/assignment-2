from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

from pipeline.ingestion import Document


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    token_count: int


class TextPreprocessor:
    def __init__(self, max_tokens: int = 900, overlap_tokens: int = 120) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = min(max(0, overlap_tokens), max_tokens - 1 if max_tokens > 1 else 0)

    def chunk_document(self, document: Document) -> list[Chunk]:
        cleaned = self.clean_text(document.text)
        if not cleaned:
            return []

        paragraphs = [piece.strip() for piece in re.split(r'\n\s*\n+', cleaned) if piece.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            if paragraph_tokens > self.max_tokens:
                if current:
                    chunks.append('\n\n'.join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_long_paragraph(paragraph))
                continue

            if current_tokens + paragraph_tokens > self.max_tokens and current:
                chunks.append('\n\n'.join(current))
                current = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current.append(paragraph)
                current_tokens += paragraph_tokens

        if current:
            chunks.append('\n\n'.join(current))

        if self.overlap_tokens <= 0 or len(chunks) <= 1:
            return [Chunk(index=index, text=text, token_count=self.count_tokens(text)) for index, text in enumerate(chunks)]

        overlapped: list[Chunk] = []
        previous_tail = ''
        for index, text in enumerate(chunks):
            merged = previous_tail + '\n\n' + text if previous_tail else text
            overlapped.append(Chunk(index=index, text=merged, token_count=self.count_tokens(merged)))
            previous_tail = self._tail_text(text)
        return overlapped

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines: list[str] = []
        last_line = ''
        for raw_line in text.split('\n'):
            line = re.sub(r'\s+', ' ', raw_line).strip()
            if not line:
                lines.append('')
                last_line = ''
                continue
            if self._looks_like_boilerplate(line):
                continue
            if line == last_line:
                continue
            lines.append(line)
            last_line = line
        cleaned = '\n'.join(lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        return cleaned.strip()

    def count_tokens(self, text: str) -> int:
        encoder = self._encoder()
        if encoder is not None:
            return len(encoder.encode(text))
        return max(1, int(len(text.split()) * 1.3))

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = self.count_tokens(sentence)
            if sentence_tokens > self.max_tokens:
                if current:
                    chunks.append(' '.join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(self._hard_split(sentence))
                continue
            if current_tokens + sentence_tokens > self.max_tokens and current:
                chunks.append(' '.join(current))
                current = [sentence]
                current_tokens = sentence_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_tokens
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        approx_words = max(1, int(self.max_tokens * 0.75))
        chunks: list[str] = []
        for start in range(0, len(words), approx_words):
            pieces = words[start:start + approx_words]
            chunks.append(' '.join(pieces))
        return chunks

    def _tail_text(self, text: str) -> str:
        words = text.split()
        if not words:
            return ''
        approx_words = max(1, int(self.overlap_tokens * 0.75))
        return ' '.join(words[-approx_words:])

    def _encoder(self):
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model('gpt-4o-mini')
        except Exception:
            try:
                return tiktoken.get_encoding('o200k_base')
            except Exception:
                return None

    def _looks_like_boilerplate(self, line: str) -> bool:
        if len(line) <= 2:
            return True
        if re.fullmatch(r'[\W_]+', line):
            return True
        lowered = line.lower()
        return any(marker in lowered for marker in ('copyright', 'all rights reserved', 'privacy policy', 'terms of service'))
