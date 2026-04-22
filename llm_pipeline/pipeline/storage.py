from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_outputs(results: list[dict[str, Any]], report_text: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_csv(results, output_dir / 'results.csv')
    (output_dir / 'summary_report.txt').write_text(report_text, encoding='utf-8')


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for result in results:
        rows.append({
            'source_name': result.get('source_name', ''),
            'source_type': result.get('source_type', ''),
            'chunk_index': result.get('chunk_index', ''),
            'chunk_tokens': result.get('chunk_tokens', ''),
            'summary': result.get('summary', ''),
            'people': '; '.join(result.get('entities', {}).get('people', [])),
            'places': '; '.join(result.get('entities', {}).get('places', [])),
            'organizations': '; '.join(result.get('entities', {}).get('organizations', [])),
            'sentiment_label': result.get('sentiment', {}).get('label', ''),
            'sentiment_confidence': result.get('sentiment', {}).get('confidence', ''),
            'questions': ' | '.join(result.get('questions', [])),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
