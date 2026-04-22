# llm_pipeline

A modular Python data pipeline that ingests a text or PDF file plus a list of URLs, cleans and chunks the content, sends each chunk to an LLM API for structured extraction, and writes JSON, CSV, and text summary outputs.

## What it uses

This implementation uses direct OpenAI-compatible chat completions over HTTP. You can point it at OpenAI or another OpenAI-compatible provider by changing the base URL and API key environment variables.

## Features

- Accepts one `.txt` or `.pdf` input file and any number of URLs in the same run
- Cleans noisy text, removes basic boilerplate, and chunks content by approximate token count
- Calls an LLM with retries and exponential backoff
- Parses structured JSON output and attempts JSON repair if the model returns malformed JSON
- Skips bad inputs and logs failures instead of crashing the whole run
- Writes `results.json`, `results.csv`, and `summary_report.txt`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key and optional overrides:

```bash
set LLM_API_KEY=your_key_here
set LLM_MODEL=gpt-4o-mini
set LLM_BASE_URL=https://api.openai.com/v1
```

## Run

From the `llm_pipeline` folder:

```bash
python main.py --input-file sample_inputs/healthcare_ai.txt --urls https://example.com --output-dir outputs
```

You can also pass a PDF:

```bash
python main.py --input-file path/to/file.pdf --urls https://example.com https://example.org
```

## Design decisions

- `ingestion.py` handles files and URLs separately so one bad source does not stop the run.
- `preprocessor.py` keeps chunking deterministic and avoids a large abstraction layer.
- `llm_client.py` uses direct HTTP calls, explicit retries, and a JSON repair path for malformed model output.
- `storage.py` keeps output formatting separate from analysis logic.
- `reporter.py` builds a simple cross-input aggregate summary from successful chunks.

## Tested inputs

- Sample `.txt` input in `sample_inputs/healthcare_ai.txt`
- A public webpage URL
- Empty or unsupported inputs are handled by logging and skipping

## Known limitations

- The pipeline expects an OpenAI-compatible chat-completions endpoint.
- PDF extraction quality depends on how much text the PDF contains.
- HTML cleaning is intentionally simple and may not capture every edge case.
- The sample outputs in `outputs/` are illustrative until you run the pipeline with a valid API key.
