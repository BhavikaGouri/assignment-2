from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class LLMRequestError(RuntimeError):
    pass


class LLMParseError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMClient:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int = 60

    def __post_init__(self) -> None:
        self.api_key = self._resolve_api_key(self.api_key)
        self.base_url = (self.base_url or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')
        self.model = self.model or os.getenv('LLM_MODEL') or os.getenv('OPENAI_MODEL') or 'gpt-4o-mini'
        if not self.api_key:
            raise LLMRequestError('No API key was provided. Set LLM_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY.')

    def analyze_chunk(self, text: str, source_name: str, source_type: str, chunk_index: int, chunk_tokens: int) -> dict[str, Any]:
        system_prompt = 'You extract structured information from messy input. Return a single JSON object only. No markdown, no explanation, no code fences.'
        user_prompt = {
            'source_name': source_name,
            'source_type': source_type,
            'chunk_index': chunk_index,
            'chunk_text': text,
            'output_schema': {
                'summary': '2 to 3 sentence summary',
                'entities': {'people': ['names'], 'places': ['locations'], 'organizations': ['organizations']},
                'sentiment': {'label': 'positive|neutral|negative', 'confidence': 'number between 0 and 1'},
                'questions': ['three important questions raised by the text'],
            },
        }
        raw_content = self._call_model(system_prompt, json.dumps(user_prompt, ensure_ascii=False))
        parsed = self._parse_json(raw_content)
        if parsed is None:
            repaired = self._repair_json(raw_content)
            parsed = self._parse_json(repaired)
        if parsed is None:
            raise LLMParseError(f'Could not parse JSON response for {source_name} chunk {chunk_index}.')
        return {
            'source_name': source_name,
            'source_type': source_type,
            'chunk_index': chunk_index,
            'chunk_tokens': chunk_tokens,
            'summary': parsed.get('summary', ''),
            'entities': self._normalize_entities(parsed.get('entities', {})),
            'sentiment': self._normalize_sentiment(parsed.get('sentiment', {})),
            'questions': self._normalize_questions(parsed.get('questions', [])),
            'raw_model_output': raw_content,
        }

    @retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), retry=retry_if_exception_type((httpx.HTTPError, LLMRequestError)))
    def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            'temperature': 0.2,
            'response_format': {'type': 'json_object'},
        }
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f'{self.base_url}/chat/completions', json=payload, headers=headers)
        if response.status_code == 429:
            raise LLMRequestError('Rate limited by the LLM API.')
        if response.status_code >= 500:
            raise LLMRequestError(f'LLM API server error: {response.status_code}')
        if response.is_error:
            raise LLMRequestError(f'LLM API request failed: {response.status_code} {response.text[:500]}')
        body = response.json()
        try:
            return body['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMParseError('Unexpected response shape from the LLM API.') from exc

    def _repair_json(self, raw_content: str) -> str:
        prompt = 'The previous assistant response was malformed JSON. Return only a corrected JSON object with keys summary, entities, sentiment, and questions.\n\nBroken response:\n' + raw_content
        return self._call_model('You repair invalid JSON. Return JSON only.', prompt)

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        candidate = self._extract_json_object(text)
        if candidate is None:
            return None
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _extract_json_object(self, text: str) -> str | None:
        text = text.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end + 1]

    def _normalize_entities(self, entities: Any) -> dict[str, list[str]]:
        if not isinstance(entities, dict):
            return {'people': [], 'places': [], 'organizations': []}
        return {'people': self._as_string_list(entities.get('people', [])), 'places': self._as_string_list(entities.get('places', [])), 'organizations': self._as_string_list(entities.get('organizations', []))}

    def _normalize_sentiment(self, sentiment: Any) -> dict[str, Any]:
        if not isinstance(sentiment, dict):
            return {'label': 'neutral', 'confidence': 0.0}
        label = str(sentiment.get('label', 'neutral')).strip().lower()
        if label not in {'positive', 'neutral', 'negative'}:
            label = 'neutral'
        confidence = sentiment.get('confidence', 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {'label': label, 'confidence': confidence}

    def _normalize_questions(self, questions: Any) -> list[str]:
        return self._as_string_list(questions)[:3]

    def _as_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in re.split(r'\n|;|\d+\.', value) if part.strip()]
        return []

    def _resolve_api_key(self, provided: str | None) -> str | None:
        if provided:
            return provided
        return os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('GROQ_API_KEY')
