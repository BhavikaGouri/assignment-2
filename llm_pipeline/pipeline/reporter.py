from __future__ import annotations

from collections import Counter
from typing import Any


def build_summary_report(results: list[dict[str, Any]]) -> str:
    if not results:
        return 'No successful chunk analyses were produced.\n'

    sentiment_counts = Counter(result.get('sentiment', {}).get('label', 'unknown') for result in results)
    people = Counter()
    places = Counter()
    organizations = Counter()
    questions = Counter()

    for result in results:
        entities = result.get('entities', {})
        people.update(entities.get('people', []))
        places.update(entities.get('places', []))
        organizations.update(entities.get('organizations', []))
        questions.update(result.get('questions', []))

    lines = ['LLM Pipeline Summary Report', '============================', f'Total analyzed chunks: {len(results)}', '', 'Sentiment distribution:']
    for label, count in sentiment_counts.most_common():
        lines.append(f'- {label}: {count}')

    lines.extend(['', 'Top people:'])
    lines.extend(_format_counter(people))
    lines.extend(['', 'Top places:'])
    lines.extend(_format_counter(places))
    lines.extend(['', 'Top organizations:'])
    lines.extend(_format_counter(organizations))
    lines.extend(['', 'Most common questions raised:'])
    lines.extend(_format_counter(questions))
    return '\n'.join(lines).rstrip() + '\n'


def _format_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return ['- None found']
    return [f'- {value}: {count}' for value, count in counter.most_common(10)]
