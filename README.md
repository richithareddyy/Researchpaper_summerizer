# Research Paper Summarizer

A Streamlit app that summarizes and analyzes academic papers with Google's Gemini API.
Load a paper as a PDF, as pasted text, or by DOI/arXiv ID. The app can then generate
summaries and analyses, extract figures, tables and references, and export the results.

## Features

- **Input methods**
  - Upload a single PDF
  - Paste text
  - Upload several PDFs and compare them
  - Look up a paper by DOI or arXiv ID/URL. arXiv papers are downloaded in full. For other
    DOIs the app uses the open-access PDF when Crossref lists one, and otherwise the abstract.
- **Summaries**: comprehensive, executive, technical, critique, or "explain like I'm 5"
- **Detailed analysis**: methodology, literature context, future research, and practical applications
- **Follow-up questions**: research questions the paper raises
- **Metadata**: title, authors, year, venue, and DOI (from the PDF, Gemini, Crossref, or arXiv)
- **Extraction**: references, embedded figures, and tables (via PyMuPDF table detection)
- **Visualizations**: TF-IDF keywords, citations by year, and a figure/table gallery
- **Paper comparison**: full, methodology-focused, or results-focused
- **Export**: a report with the summary, analyses and questions, as Markdown, PDF, or JSON
- **History**: reopen the last 10 papers from this session, including their results

Use **Advanced Analysis** in the sidebar to choose which analyses and visualizations appear.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key from https://aistudio.google.com/apikey

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your key in .env
```

## Run

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 and reads your key from `.env` automatically. If no key is
found, the sidebar explains how to add one; you can still load papers and browse their figures,
tables, and references.

To try the app without your own PDF, click **Try a sample paper**. It downloads
*Attention Is All You Need* (Vaswani et al., 2017) from arXiv. Then click **Generate Summary**.

## Configuration

Set these in `.env`. Only `GOOGLE_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (`GEMINI_API_KEY` also works) |
| `GEMINI_MODEL` | `gemini-flash-latest` | Default model. The sidebar also lists every Gemini text model your key can use. |
| `MAX_INPUT_CHARS` | `400000` | Maximum characters of paper text sent to the model |
| `MAX_DOWNLOAD_MB` | `50` | Size limit for PDFs downloaded during DOI/arXiv lookup |
| `CROSSREF_MAILTO` | — | Optional contact email sent to Crossref/arXiv, as their API etiquette asks |

For Streamlit Community Cloud, put `GOOGLE_API_KEY` in the app's secrets instead of `.env`.

## Limitations

- Scanned PDFs without selectable text are not supported (no OCR).
- Reference parsing, table detection, and citation years use heuristics. Results depend on
  the paper's layout.
- Many publisher DOIs have no open full text. For those, download the PDF and upload it.
- Every AI feature needs a valid Gemini API key and uses your API quota.
