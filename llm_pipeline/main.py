from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.ingestion import ingest_documents
from pipeline.llm_client import LLMClient
from pipeline.logger import setup_logger
from pipeline.preprocessor import TextPreprocessor
from pipeline.reporter import build_summary_report
from pipeline.storage import write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ingest a file and URLs, analyze them with an LLM, and write structured outputs.')
    parser.add_argument('--input-file', type=Path, help='Path to a .txt or .pdf file to ingest.')
    parser.add_argument('--urls', nargs='*', default=[], help='List of URLs to ingest in the same run.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'), help='Directory for generated outputs.')
    parser.add_argument('--model', default=None, help='LLM model name. Defaults to LLM_MODEL or gpt-4o-mini.')
    parser.add_argument('--base-url', default=None, help='OpenAI-compatible base URL. Defaults to LLM_BASE_URL or OpenAI.')
    parser.add_argument('--api-key', default=None, help='LLM API key. Defaults to LLM_API_KEY or provider-specific env vars.')
    parser.add_argument('--chunk-tokens', type=int, default=900, help='Approximate maximum tokens per chunk.')
    parser.add_argument('--chunk-overlap', type=int, default=120, help='Approximate token overlap between chunks.')
    parser.add_argument('--timeout', type=int, default=60, help='HTTP timeout for the LLM API in seconds.')
    parser.add_argument('--log-level', default='INFO', help='Logging level.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.output_dir, level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    sources = []
    if args.input_file:
        sources.append(str(args.input_file))
    sources.extend(args.urls)

    if not sources:
        logger.error('No input file or URLs were provided.')
        return 1

    preprocessor = TextPreprocessor(max_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap)
    client = LLMClient(api_key=args.api_key, base_url=args.base_url, model=args.model, timeout=args.timeout)
    documents = ingest_documents(args.input_file, args.urls, logger)

    if not documents:
        logger.error('No valid inputs could be ingested.')
        return 1

    results = []
    for document in documents:
        chunks = preprocessor.chunk_document(document)
        logger.info('Prepared %s chunk(s) from %s', len(chunks), document.source_name)
        for chunk in chunks:
            try:
                analysis = client.analyze_chunk(
                    chunk.text,
                    source_name=document.source_name,
                    source_type=document.source_type,
                    chunk_index=chunk.index,
                    chunk_tokens=chunk.token_count,
                )
            except Exception:
                logger.exception('Skipping chunk %s from %s because analysis failed.', chunk.index, document.source_name)
                continue
            results.append(analysis)

    report_text = build_summary_report(results)
    write_outputs(results, report_text, args.output_dir)
    logger.info('Wrote %s analyzed chunk(s) to %s', len(results), args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
