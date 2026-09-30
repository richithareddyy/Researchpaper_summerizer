import csv
import hashlib
import io
import itertools
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote

import markdown
import pandas as pd
import plotly.express as px
import pymupdf
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Load environment variables
load_dotenv()

APP_TITLE = "Advanced Research Paper Summarizer"

# Configuration (override any of these in .env)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# Quick background tasks such as reading a paper's title and authors
FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-flash-lite-latest")
# Tried in order when the selected model is out of quota or overloaded
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
# Hidden from the model picker: previews often have no free quota, the rest aren't text models
HIDDEN_MODEL_TAGS = ("preview", "exp", "tts", "image", "live", "audio", "embedding", "robotics", "computer-use")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "400000"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "50"))
REQUEST_TIMEOUT = 30
HISTORY_LIMIT = 10
MIN_TEXT_CHARS = 200
MIN_FIGURE_SIZE = 80  # px; smaller images are usually logos or icons
MAX_FIGURES = 60

SUMMARY_TYPES = {
    "comprehensive": "Comprehensive",
    "executive": "Executive",
    "technical": "Technical",
    "critique": "Critique",
    "eli5": "Explain Like I'm 5",
}

ANALYSIS_TYPES = {
    "methodology": "Methodology",
    "literature": "Literature Context",
    "future_research": "Future Research",
    "practical_applications": "Practical Applications",
}

# Sidebar options; the ones mapped to an analysis type unlock it in the Analysis tab
ANALYSIS_OPTIONS = {
    "Extract Keywords": None,
    "Analyze Methodology": "methodology",
    "Analyze Literature Context": "literature",
    "Identify Future Research": "future_research",
    "Find Practical Applications": "practical_applications",
    "Extract Figures & Tables": None,
    "Generate Citation Graph": None,
}

CITATION_STYLES = ["APA", "MLA", "BibTeX"]
NAME_PARTICLES = {"van", "von", "de", "der", "den", "da", "di", "del", "la", "le", "du", "dos", "das", "bin", "al"}

CHAT_HISTORY_TURNS = 6  # earlier exchanges sent with each question for context
CHAT_STARTERS = [
    "What problem does this paper solve?",
    "What are the main results?",
    "What are the limitations?",
]

SAMPLE_PAPER = {"arxiv_id": "1706.03762", "label": "Attention Is All You Need (2017)"}

COMPARISON_FOCUS = {
    "Full Comparison": "",
    "Methodology Comparison": "Concentrate on how the research designs, data, and methods differ and which is more rigorous.",
    "Results Comparison": "Concentrate on the findings: where results agree, where they conflict, and how strong the evidence is.",
}

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
SESSION_DEFAULTS = {
    "history": [],
    "extracted_text": "",
    "processed_papers": [],
    "current_summary": "",
    "summary_info": "",
    "paper_metadata": {},
    "figures": [],
    "tables": [],
    "references": [],
    "paper_id": None,
    "paper_source": "",
    "analyses": {},
    "follow_up_questions": "",
    "comparison": "",
    "chat_messages": [],
    "study_aids": None,
    "last_upload_id": None,
    "paper_cache": {},
}
for _key, _value in SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value.copy() if isinstance(_value, (list, dict)) else _value

# Custom CSS
st.markdown("""
<style>
    .main-header {color: #1E88E5; font-size: 40px; font-weight: bold; margin-bottom: 20px; text-align: center;}
</style>
""", unsafe_allow_html=True)


EMPTY_RESPONSE_MESSAGE = ("The model returned an empty response. It may have been blocked by safety filters; "
                          "try another summary type or model.")


class GeminiError(Exception):
    """Raised when a Gemini request cannot be completed."""


class PaperLookupError(Exception):
    """Raised when a DOI / arXiv lookup fails."""


# Define enhanced functions for text extraction
def extract_text_from_pdf(pdf_bytes):
    """Extract text, figures, tables and metadata from PDF bytes."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"The file could not be read as a PDF ({e}).") from e

    with doc:
        if doc.needs_pass:
            raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")

        metadata = {}
        pdf_meta = doc.metadata or {}
        if pdf_meta.get("title", "").strip():
            metadata["title"] = pdf_meta["title"].strip()
        if pdf_meta.get("author", "").strip():
            metadata["authors"] = [a.strip() for a in re.split(r";|\band\b", pdf_meta["author"]) if a.strip()]

        text = "\n".join(page.get_text() for page in doc)

        # Extract images
        figures = []
        seen_xrefs = set()
        for page_num, page in enumerate(doc):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs or len(figures) >= MAX_FIGURES:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.width < MIN_FIGURE_SIZE or pix.height < MIN_FIGURE_SIZE:
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK and similar -> RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    figures.append({
                        "page": page_num + 1,
                        "data": pix.tobytes("png"),
                        "width": pix.width,
                        "height": pix.height
                    })
                except Exception:
                    continue  # skip images in unsupported formats

        # Detect tables
        tables = []
        for page_num, page in enumerate(doc):
            try:
                found = page.find_tables()
            except Exception:
                continue
            for table in found.tables:
                rows = [["" if cell is None else str(cell) for cell in row] for row in table.extract()]
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    tables.append({"page": page_num + 1, "rows": rows})

    return text, metadata, figures, tables


def extract_references(text):
    """Extract individual entries from the references section of the paper"""
    heading = re.compile(
        r'^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    matches = list(heading.finditer(text))
    if not matches:
        matches = list(re.finditer(r'references|bibliography|works cited', text, re.IGNORECASE))
    if not matches:
        return []

    # The last match is most likely the actual references section
    ref_text = text[matches[-1].end():]

    if re.search(r'^\s*\[\d+\]', ref_text, re.MULTILINE):
        entries = re.split(r'^\s*\[\d+\]\s*', ref_text, flags=re.MULTILINE)
    elif re.search(r'^\s*\d+\.\s+\S', ref_text, re.MULTILINE):
        entries = re.split(r'^\s*\d+\.\s+', ref_text, flags=re.MULTILINE)
    else:
        # Author-year style: a new entry starts with "Surname, X"
        entries = re.split(r"\n(?=[A-Z][A-Za-z'\-]+,\s+[A-Z])", ref_text)

    references = []
    for entry in entries:
        clean_entry = re.sub(r'\s+', ' ', entry).strip()
        if len(clean_entry) > 20:  # Minimum length to be a valid reference
            references.append(clean_entry)
    return references[:500]


def prepare_text_for_model(text):
    """Normalise whitespace and trim the text to the configured input limit.

    Returns the prepared text and whether it was truncated."""
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # re-join words hyphenated across lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    if len(text) > MAX_INPUT_CHARS:
        return text[:MAX_INPUT_CHARS], True
    return text, False


def paper_id_for(content):
    """Stable identifier for a paper's content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def safe_filename(title, default="research_paper"):
    safe = re.sub(r'[^\w\s-]', '', title or "").strip()
    safe = re.sub(r'\s+', '_', safe)[:80]
    return safe or default


def format_authors(metadata):
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        return authors
    return ", ".join(authors) if authors else "Unknown"


# Advanced analysis functions
@st.cache_data(show_spinner=False)
def extract_keywords(text, top_n=10):
    """Extract important keywords from the paper using TF-IDF.

    The paper is split into passages so that terms concentrated in a few
    passages score higher than terms that appear everywhere."""
    if not text or len(text) < 100:
        return []

    words = re.sub(r'[^A-Za-z\s-]', ' ', text).split()
    passages = [" ".join(words[i:i + 80]) for i in range(0, len(words), 80)]
    if not passages:
        return []

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2 if len(passages) >= 4 else 1,
        max_df=0.9 if len(passages) >= 4 else 1.0,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(passages)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [(word, float(score)) for word, score in keyword_scores[:top_n]]


def create_word_cloud_data(text):
    """Prepare data for word cloud visualization"""
    keywords = extract_keywords(text, top_n=50)
    if not keywords:
        return []

    # Normalize scores for visualization
    max_score = max(score for _, score in keywords)
    return [(word, int((score / max_score) * 100)) for word, score in keywords]


def generate_citation_graph(references):
    """Count cited works by publication year"""
    if not references or len(references) < 3:
        return None

    current_year = datetime.now().year
    year_counts = {}
    for ref in references:
        for match in re.finditer(r'\b(19[5-9]\d|20\d{2})\b', ref):
            year = int(match.group(1))
            if year <= current_year:
                year_counts[year] = year_counts.get(year, 0) + 1
                break

    if not year_counts:
        return None
    return sorted(year_counts.items())


# Gemini API helpers
def get_api_key():
    """API key from the environment (.env), then Streamlit secrets."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    # Retry temporary server failures with exponential backoff; quota errors (429)
    # are not retried here because call_gemini switches to a fallback model instead
    retry = types.HttpRetryOptions(attempts=2, initial_delay=2, max_delay=5,
                                   http_status_codes=[500, 502, 503, 504])
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))


@st.cache_data(ttl=3600, show_spinner=False)
def list_available_models(api_key):
    """Gemini text models available to this API key."""
    names = set()
    for model in get_client(api_key).models.list():
        name = (model.name or "").removeprefix("models/")
        actions = model.supported_actions or []
        if (name.startswith("gemini") and "generateContent" in actions
                and not any(tag in name for tag in HIDDEN_MODEL_TAGS)):
            names.add(name)
    # "-latest" aliases always point at a current model, so list them first
    return sorted(names, key=lambda n: (not n.endswith("-latest"), [-ord(c) for c in n]))


def _describe_api_error(error, model_name):
    code = getattr(error, "code", None)
    message = getattr(error, "message", None) or str(error)
    if code in (401, 403) or "API key not valid" in message:
        return "The Google API key was rejected. Check GOOGLE_API_KEY in your .env file (or the app's Secrets when deployed)."
    if code == 404:
        return f"The model {model_name} is not available: {message}. Choose a different model in the sidebar."
    if code == 429:
        if re.search(r"limit:\s*0\b", message):
            return (f"Your API key has no free quota for {model_name}. Pick another model in the sidebar, "
                    "or enable billing in Google AI Studio.")
        if re.search(r"per ?day", message, re.IGNORECASE):
            return (f"Your daily free quota for {model_name} is used up. It resets at midnight Pacific time. "
                    "Each model has its own quota, so another model in the sidebar may still work.")
        wait = re.search(r"retry in ([\d.]+)s", message, re.IGNORECASE)
        wait_text = f"about {round(float(wait.group(1)))} seconds" if wait else "a minute"
        return (f"Too many requests to {model_name} in a short time (free-tier rate limit). "
                f"Wait {wait_text} and try again.")
    if code in (500, 502, 503, 504):
        return ("The selected Gemini model is busy right now (a temporary problem on Google's side). "
                "Try again in a minute, or pick a different model in the sidebar.")
    return f"Gemini API error ({code}): {message}"


def _with_fallback(model_name, send):
    """Run send(client, model), switching to a fallback model on quota or availability errors."""
    api_key = get_api_key()
    if not api_key:
        raise GeminiError("No Google API key configured. Add GOOGLE_API_KEY to your .env file (or the app's Secrets when deployed).")

    model_name = model_name.removeprefix("models/")
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    first_error = None
    for candidate in candidates:
        try:
            result = send(get_client(api_key), candidate)
        except genai_errors.APIError as e:
            # Out of quota, overloaded, or unavailable: try the next model
            if getattr(e, "code", None) in (404, 429, 500, 502, 503, 504):
                first_error = first_error or e
                continue
            raise GeminiError(_describe_api_error(e, candidate)) from e
        except Exception as e:
            raise GeminiError(f"Could not reach the Gemini API: {e}") from e
        if candidate != model_name:
            st.toast(f"{model_name} was unavailable, so {candidate} was used instead.")
        return result

    message = _describe_api_error(first_error, model_name)
    if len(candidates) > 1:
        message += (f" The backup models ({', '.join(candidates[1:])}) were unavailable too, so this is "
                    "most likely a limit on your API key rather than on one model.")
    raise GeminiError(message) from first_error


def call_gemini(prompt, model_name, json_output=False):
    """Send a prompt to Gemini and return the whole response text."""
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_output else None
    response = _with_fallback(
        model_name, lambda client, model: client.models.generate_content(model=model, contents=prompt, config=config)
    )
    text = response.text
    if not text or not text.strip():
        raise GeminiError(EMPTY_RESPONSE_MESSAGE)
    return text.strip()


def stream_gemini(prompt, model_name):
    """Send a prompt to Gemini and return a generator of text chunks as they arrive."""
    def start(client, model):
        chunks = iter(client.models.generate_content_stream(model=model, contents=prompt))
        return chunks, next(chunks, None)  # the request is sent here, so errors can trigger a fallback

    chunks, first = _with_fallback(model_name, start)

    def text_chunks():
        try:
            for chunk in itertools.chain([first] if first else [], chunks):
                if chunk.text:
                    yield chunk.text
        except genai_errors.APIError as e:
            raise GeminiError(_describe_api_error(e, model_name)) from e
        except Exception as e:
            raise GeminiError(f"The response was interrupted: {e}") from e

    return text_chunks()


# Define functions for generating summaries using Gemini API
def generate_summary(text, model_name=DEFAULT_MODEL, summary_type="comprehensive", stream=False):
    """Generate paper summary using Gemini API"""
    text, _ = prepare_text_for_model(text)

    prompts = {
        "comprehensive": f"""
        You are a research assistant specialized in summarizing academic papers.
        Provide a comprehensive summary of the following research paper:

        <paper>
        {text}
        </paper>

        Structure your summary as follows:
        # Paper Summary
        ## Title and Authors (if available)
        ## Research Question & Objectives
        ## Methodology
        ## Key Findings
        ## Conclusions & Implications

        Make sure to highlight the most important contributions and innovations.
        Format your response in Markdown.
        """,

        "executive": f"""
        Provide a concise executive summary (250-350 words) of the following research paper,
        focusing on the problem addressed, key findings, and practical implications:

        <paper>
        {text}
        </paper>

        Format your response in Markdown.
        """,

        "technical": f"""
        Provide a technical summary of the following research paper, focusing on the methodology,
        technical innovations, algorithms, and experimental results:

        <paper>
        {text}
        </paper>

        Structure your summary as follows:
        # Technical Summary
        ## Problem Statement
        ## Technical Approach
        ## Algorithms & Models
        ## Implementation Details
        ## Evaluation Metrics
        ## Results

        Format your response in Markdown.
        """,

        "critique": f"""
        Provide a critical analysis of the following research paper, evaluating its strengths,
        weaknesses, methodological rigor, and contributions to the field:

        <paper>
        {text}
        </paper>

        Structure your critique as follows:
        # Critical Analysis
        ## Overview
        ## Strengths
        ## Limitations & Weaknesses
        ## Methodological Assessment
        ## Significance & Impact
        ## Suggestions for Improvement

        Format your response in Markdown.
        """,

        "eli5": f"""
        Explain the following research paper as if you were explaining it to a 5th grader.
        Use simple language, analogies, and focus on the big picture ideas:

        <paper>
        {text}
        </paper>

        Keep your explanation under 500 words and make it engaging and easy to understand.
        Format your response in Markdown.
        """
    }

    return (stream_gemini if stream else call_gemini)(prompts[summary_type], model_name)


def generate_detailed_analysis(text, model_name=DEFAULT_MODEL, analysis_type="methodology", stream=False):
    """Generate detailed analysis of specific aspects of the paper"""
    text, _ = prepare_text_for_model(text)

    prompts = {
        "methodology": f"""
        Provide a detailed analysis of the methodology used in this research paper:

        <paper>
        {text}
        </paper>

        Focus on:
        1. Research design and approach
        2. Data collection methods
        3. Analytical techniques
        4. Validity and reliability considerations
        5. Methodological innovations
        6. Limitations of the methodology

        Format your response in Markdown.
        """,

        "literature": f"""
        Analyze how this paper relates to existing literature in the field:

        <paper>
        {text}
        </paper>

        Focus on:
        1. Key works cited and their importance
        2. How this paper builds on previous research
        3. Gaps in literature this paper addresses
        4. Alternative perspectives not considered
        5. Where this paper fits in the broader research landscape

        Format your response in Markdown.
        """,

        "future_research": f"""
        Based on this research paper, identify promising directions for future research:

        <paper>
        {text}
        </paper>

        Consider:
        1. Unanswered questions raised by this paper
        2. Limitations that could be addressed in future work
        3. Potential extensions of the methodology
        4. New hypotheses suggested by the findings
        5. Interdisciplinary connections that could be explored

        Format your response in Markdown.
        """,

        "practical_applications": f"""
        Identify and elaborate on the practical applications of this research:

        <paper>
        {text}
        </paper>

        Focus on:
        1. Industry applications
        2. Policy implications
        3. Practical tools or frameworks that could be developed
        4. Potential beneficiaries of this research
        5. Steps needed to translate this research into practice

        Format your response in Markdown.
        """
    }

    return (stream_gemini if stream else call_gemini)(prompts[analysis_type], model_name)


def generate_follow_up_questions(text, model_name=DEFAULT_MODEL, stream=False):
    """Generate insightful follow-up questions about the paper"""
    text, _ = prepare_text_for_model(text)

    prompt = f"""
    Read the following research paper and generate 5 insightful follow-up questions that a researcher
    might ask after reading this paper. These questions should probe deeper into the methodology,
    explore limitations, suggest extensions, or connect to broader research themes.

    <paper>
    {text}
    </paper>

    Format your response as a numbered list in Markdown.
    """

    return (stream_gemini if stream else call_gemini)(prompt, model_name)


def _parse_json_object(raw, what):
    """Parse a JSON object from a model response, tolerating code fences."""
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiError(f"The model did not return valid {what} JSON. Please try again.") from e
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise GeminiError(f"The model did not return a {what} object. Please try again.")
    return data


def extract_paper_details(text, model_name=DEFAULT_MODEL):
    """Extract structured metadata from the beginning of the paper using Gemini"""
    prompt = f"""
    Extract the following metadata from the beginning of this research paper:

    <paper>
    {text[:6000]}
    </paper>

    Return a JSON object with these fields:
    - title (string): The paper's title
    - authors (array of strings): List of authors
    - publication_year (integer or null): Year of publication
    - journal_or_conference (string or null): Publication venue
    - keywords (array of strings): Author-provided keywords
    - doi (string or null): DOI if present

    Use null or an empty array for anything that is not stated in the text.
    """

    data = _parse_json_object(call_gemini(prompt, model_name, json_output=True), "metadata")

    year = data.get("publication_year")
    try:
        year = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year = None

    def as_list(value):
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(v) for v in value or [] if str(v).strip()]

    return {
        "title": (data.get("title") or "").strip() or None,
        "authors": as_list(data.get("authors")),
        "publication_year": year,
        "journal_or_conference": data.get("journal_or_conference") or None,
        "keywords": as_list(data.get("keywords")),
        "doi": data.get("doi") or None,
    }


def compare_papers(papers_text, model_name=DEFAULT_MODEL, comparison_type="Full Comparison"):
    """Compare multiple research papers"""
    if len(papers_text) < 2:
        raise ValueError("Need at least two papers to compare")

    # Summarize each paper first (in parallel) to keep the comparison prompt small
    ctx = get_script_run_ctx()

    def summarize(text):
        add_script_run_ctx(threading.current_thread(), ctx)  # lets worker threads show notices
        return generate_summary(text, model_name, "executive")

    with ThreadPoolExecutor(max_workers=min(4, len(papers_text))) as pool:
        briefs = list(pool.map(summarize, papers_text))
    summaries = [f"Paper {i + 1}:\n{brief}" for i, brief in enumerate(briefs)]

    combined_summaries = "\n\n".join(summaries)
    focus = COMPARISON_FOCUS.get(comparison_type, "")

    prompt = f"""
    Compare and contrast the following research papers based on these summaries:

    {combined_summaries}

    {focus}

    Structure your comparison as follows:
    # Comparative Analysis
    ## Research Focus & Objectives
    ## Methodological Approaches
    ## Key Findings
    ## Strengths & Weaknesses
    ## Complementary Insights
    ## Contradictory Claims (if any)
    ## Integration Possibilities

    Format your response in Markdown.
    """

    return call_gemini(prompt, model_name)


def answer_question(text, question, chat_history, model_name=DEFAULT_MODEL, stream=False):
    """Answer a question about the paper, using recent chat turns as context"""
    text, _ = prepare_text_for_model(text)
    recent = chat_history[-CHAT_HISTORY_TURNS * 2:]
    transcript = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent
    ) or "(no earlier messages)"

    prompt = f"""
    You are a research assistant answering questions about the research paper below.
    Base your answer on the paper. If the paper does not contain the answer, say so clearly;
    you may then add brief general context, labelled as coming from outside the paper.
    Be concise and use Markdown where it helps.

    <paper>
    {text}
    </paper>

    <conversation>
    {transcript}
    </conversation>

    Question: {question}
    """

    return (stream_gemini if stream else call_gemini)(prompt, model_name)


def generate_study_aids(text, model_name=DEFAULT_MODEL):
    """Create a glossary of key terms and flashcards from the paper"""
    text, _ = prepare_text_for_model(text)

    prompt = f"""
    Create study aids for a student reading the following research paper:

    <paper>
    {text}
    </paper>

    Return a JSON object with:
    - glossary: 10 to 12 objects with "term" and "definition". Pick the terms a reader must
      understand; define each in one or two plain sentences, as it is used in this paper.
    - flashcards: 8 to 10 objects with "question" and "answer" that test understanding of the
      paper's problem, method, results, and limitations. Keep answers to one to three sentences.
    """

    data = _parse_json_object(call_gemini(prompt, model_name, json_output=True), "study aids")

    def clean(items, first, second):
        pairs = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                a, b = str(item.get(first) or "").strip(), str(item.get(second) or "").strip()
                if a and b:
                    pairs.append({first: a, second: b})
        return pairs

    aids = {
        "glossary": clean(data.get("glossary"), "term", "definition"),
        "flashcards": clean(data.get("flashcards"), "question", "answer"),
    }
    if not aids["glossary"] and not aids["flashcards"]:
        raise GeminiError("The model did not return usable study aids. Please try again.")
    return aids


def flashcards_to_csv(flashcards):
    """Question,answer rows (no header) that flashcard apps such as Anki can import."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for card in flashcards:
        writer.writerow([card["question"], card["answer"]])
    return buffer.getvalue()


# Citations
def _split_name(name):
    """Return (last, first) for 'First Last' or 'Last, First'."""
    name = re.sub(r'\s+', ' ', name).strip()
    if "," in name:
        last, first = (part.strip() for part in name.split(",", 1))
        return last, first
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in NAME_PARTICLES:  # keep "van der" with the surname
        i -= 1
    return " ".join(parts[i:]), " ".join(parts[:i])


def _initials(first):
    initials = []
    for part in first.replace(".", ". ").split():
        pieces = [p for p in part.split("-") if p and p[0].isalpha()]
        if pieces:
            initials.append("-".join(f"{p[0].upper()}." for p in pieces))
    return " ".join(initials)


def format_citation(metadata, style):
    """Format the paper's metadata as an APA, MLA, or BibTeX citation."""
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    names = [_split_name(a) for a in authors if a and a.strip()]
    title = (metadata.get("title") or "Untitled").strip().rstrip(".")
    year = metadata.get("publication_year")
    venue = metadata.get("journal_or_conference")
    doi = metadata.get("doi")
    doi_url = f"https://doi.org/{doi}" if doi else ""

    if style == "APA":
        formatted = [f"{last}, {_initials(first)}" if first else last for last, first in names]
        if len(formatted) > 20:
            author_text = ", ".join(formatted[:19]) + ", ... " + formatted[-1]
        elif len(formatted) > 1:
            author_text = ", ".join(formatted[:-1]) + ", & " + formatted[-1]
        else:
            author_text = formatted[0] if formatted else ""
        year_text = f"({year})" if year else "(n.d.)"
        parts = [f"{author_text} {year_text}. {title}." if author_text else f"{title}. {year_text}."]
        if venue:
            parts.append(f"{venue}.")
        if doi_url:
            parts.append(doi_url)
        return " ".join(parts)

    if style == "MLA":
        if len(names) == 1:
            author_text = f"{names[0][0]}, {names[0][1]}".rstrip(", ")
        elif len(names) == 2:
            author_text = f"{names[0][0]}, {names[0][1]}, and {names[1][1]} {names[1][0]}".replace(" ,", ",").strip()
        elif names:
            author_text = f"{names[0][0]}, {names[0][1]}, et al".rstrip(", ")
        else:
            author_text = ""
        details = ", ".join(str(v) for v in (venue, year, doi_url) if v)
        citation = f'{author_text.rstrip(".")}. "{title}."' if author_text else f'"{title}."'
        return f"{citation} {details}." if details else citation

    # BibTeX
    first_author = re.sub(r'[^A-Za-z]', '', names[0][0]).lower() if names else "paper"
    title_word = next((w.lower() for w in re.findall(r'[A-Za-z]+', title) if len(w) > 3), "")
    key = f"{first_author}{year or ''}{title_word}"
    entry_type = "article" if venue and "preprint" not in venue.lower() else "misc"
    fields = [("title", f"{{{title}}}")]
    if names:
        fields.append(("author", " and ".join(f"{last}, {first}" if first else last for last, first in names)))
    if year:
        fields.append(("year", str(year)))
    if venue:
        fields.append(("journal" if entry_type == "article" else "howpublished", venue))
    if doi:
        fields.append(("doi", doi))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


# Paper loading
def build_paper(text, metadata, figures=None, tables=None, source=""):
    return {
        "id": paper_id_for(text),
        "text": text,
        "metadata": metadata,
        "figures": figures or [],
        "tables": tables or [],
        "references": extract_references(text),
        "filename": source,
    }


def enrich_metadata(paper):
    """Fill in title/authors/etc. with Gemini's fast model. Failures are non-fatal."""
    if not get_api_key():
        return
    try:
        ai_metadata = extract_paper_details(paper["text"], FAST_MODEL)
    except GeminiError as e:
        st.caption(f"Could not extract paper details automatically: {e}")
        return
    for key, value in ai_metadata.items():
        if value:
            paper["metadata"][key] = value


def process_pdf(pdf_bytes, filename):
    """Extract a PDF once per session; later calls reuse the cached result."""
    pid = paper_id_for(pdf_bytes)
    cache = st.session_state.paper_cache
    if pid in cache:
        return cache[pid]

    text, metadata, figures, tables = extract_text_from_pdf(pdf_bytes)
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ValueError("No selectable text was found. The PDF may be a scanned image; OCR is not supported.")

    paper = build_paper(text, metadata, figures, tables, filename)
    enrich_metadata(paper)
    cache[pid] = paper
    return paper


def load_paper(paper):
    """Make a paper the active one and clear results from the previous paper."""
    st.session_state.paper_id = paper["id"]
    st.session_state.paper_source = paper.get("filename", "")
    st.session_state.extracted_text = paper["text"]
    st.session_state.paper_metadata = dict(paper["metadata"])
    st.session_state.figures = paper["figures"]
    st.session_state.tables = paper["tables"]
    st.session_state.references = paper["references"]
    st.session_state.current_summary = paper.get("summary", "")
    st.session_state.summary_info = paper.get("summary_info", "")
    st.session_state.analyses = dict(paper.get("analyses", {}))
    st.session_state.follow_up_questions = paper.get("follow_up_questions", "")
    st.session_state.chat_messages = list(paper.get("chat_messages", []))
    st.session_state.study_aids = paper.get("study_aids")


def _http_headers(include_contact=True):
    """Request headers; the optional contact email is only sent to Crossref and arXiv."""
    headers = {"User-Agent": "ResearchPaperSummarizer/1.0"}
    mailto = os.getenv("CROSSREF_MAILTO")
    if mailto and include_contact:
        headers["User-Agent"] += f" (mailto:{mailto})"
    return headers


def _download(url, accept_pdf_only=False, include_contact=False):
    """Download a URL with a timeout and size limit."""
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    with requests.get(url, headers=_http_headers(include_contact), timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > limit:
                raise PaperLookupError(f"The file is larger than {MAX_DOWNLOAD_MB} MB.")
            chunks.append(chunk)
    data = b"".join(chunks)
    if accept_pdf_only and not data.startswith(b"%PDF"):
        raise PaperLookupError("The link did not return a PDF.")
    return data


def fetch_crossref_record(doi):
    """Return (metadata, raw record) for a DOI from Crossref."""
    try:
        response = requests.get(f"https://api.crossref.org/works/{quote(doi)}",
                                headers=_http_headers(), timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        raise PaperLookupError(f"Could not reach Crossref: {e}") from e
    if response.status_code == 404:
        raise PaperLookupError(f"DOI {doi} was not found in Crossref.")
    if not response.ok:
        raise PaperLookupError(f"Crossref returned an error ({response.status_code}).")

    record = response.json().get("message", {})
    date_parts = (record.get("issued") or {}).get("date-parts") or [[None]]
    metadata = {
        "title": " ".join(record.get("title") or []) or None,
        "authors": [
            " ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name", "")
            for a in record.get("author", [])
        ],
        "publication_year": date_parts[0][0] if date_parts and date_parts[0] else None,
        "journal_or_conference": " ".join(record.get("container-title") or []) or None,
        "keywords": record.get("subject") or [],
        "doi": doi,
    }
    return metadata, record


def fetch_arxiv_metadata(arxiv_id):
    """Title, authors and year from the arXiv API."""
    response = requests.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id},
                            headers=_http_headers(), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entry = ET.fromstring(response.content).find("atom:entry", ns)
    if entry is None:
        return {}

    def find_text(path):
        node = entry.find(path, ns)
        return re.sub(r'\s+', ' ', node.text).strip() if node is not None and node.text else None

    published = find_text("atom:published")
    return {
        "title": find_text("atom:title"),
        "authors": [re.sub(r'\s+', ' ', n.text).strip() for n in entry.findall("atom:author/atom:name", ns) if n.text],
        "publication_year": int(published[:4]) if published else None,
        "journal_or_conference": find_text("arxiv:journal_ref") or "arXiv preprint",
        "doi": find_text("arxiv:doi") or f"10.48550/arXiv.{re.sub(r'v[0-9]+$', '', arxiv_id)}",
    }


ARXIV_PATTERN = re.compile(
    r'^(?:https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/|arxiv:|10\.48550/arxiv\.)?'
    r'(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)(?:\.pdf)?/?$',
    re.IGNORECASE
)
DOI_PATTERN = re.compile(r'(10\.\d{4,9}/\S+)')


def lookup_paper(identifier):
    """Fetch a paper by arXiv ID/URL or DOI."""
    identifier = identifier.strip()
    if not identifier:
        raise PaperLookupError("Enter a DOI or arXiv ID.")

    arxiv_match = ARXIV_PATTERN.match(identifier)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        try:
            pdf_bytes = _download(f"https://arxiv.org/pdf/{arxiv_id}", accept_pdf_only=True, include_contact=True)
        except requests.exceptions.RequestException as e:
            raise PaperLookupError(f"Could not download arXiv paper {arxiv_id}: {e}") from e
        paper = process_pdf(pdf_bytes, f"arXiv:{arxiv_id}")
        try:
            metadata = fetch_arxiv_metadata(arxiv_id)
            paper["metadata"].update({k: v for k, v in metadata.items() if v})
        except (requests.exceptions.RequestException, ET.ParseError):
            pass  # metadata is optional; the full text is already loaded
        return paper

    doi_match = DOI_PATTERN.search(identifier)
    if not doi_match:
        raise PaperLookupError("That does not look like a DOI (10.xxxx/...) or an arXiv ID (e.g. 1706.03762).")
    doi = doi_match.group(1).rstrip(".,;)")
    metadata, record = fetch_crossref_record(doi)

    # Prefer an openly available full-text PDF
    for link in record.get("link", []):
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            try:
                pdf_bytes = _download(link["URL"], accept_pdf_only=True)
                paper = process_pdf(pdf_bytes, f"doi:{doi}")
                paper["metadata"].update({k: v for k, v in metadata.items() if v})
                return paper
            except (requests.exceptions.RequestException, PaperLookupError, ValueError):
                continue

    abstract = re.sub(r'<[^>]+>', ' ', record.get("abstract") or "")
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    if not abstract:
        raise PaperLookupError(
            "Crossref has no open full text or abstract for this DOI. "
            "Download the PDF from the publisher and use 'Upload PDF' instead."
        )
    text = f"{metadata['title'] or ''}\n\n{', '.join(metadata['authors'])}\n\nAbstract\n{abstract}"
    paper = build_paper(text, metadata, source=f"doi:{doi}")
    paper["abstract_only"] = True
    return paper


# Report building and export
def _demote_headings(text):
    return re.sub(r'^(#{1,5})(\s)', r'#\1\2', text, flags=re.MULTILINE)


def build_report_markdown():
    """Assemble all results for the current paper into one Markdown document."""
    metadata = st.session_state.paper_metadata
    sections = [
        f"# {metadata.get('title') or 'Research Paper Summary'}",
        f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "## Paper Details",
        f"- **Authors**: {format_authors(metadata)}\n"
        f"- **Year**: {metadata.get('publication_year') or 'Unknown'}\n"
        f"- **Journal/Conference**: {metadata.get('journal_or_conference') or 'Unknown'}\n"
        f"- **DOI**: {metadata.get('doi') or 'Unknown'}",
    ]
    if st.session_state.current_summary:
        sections += ["## Summary", _demote_headings(st.session_state.current_summary)]
    keywords = extract_keywords(st.session_state.extracted_text)
    if keywords:
        sections += ["## Keywords", ", ".join(word for word, _ in keywords)]
    for analysis_type, result in st.session_state.analyses.items():
        sections += [f"## {ANALYSIS_TYPES[analysis_type]} Analysis", _demote_headings(result)]
    if st.session_state.follow_up_questions:
        sections += ["## Follow-up Questions", _demote_headings(st.session_state.follow_up_questions)]
    aids = st.session_state.study_aids
    if aids and aids["glossary"]:
        sections += ["## Glossary", "\n".join(f"- **{g['term']}**: {g['definition']}" for g in aids["glossary"])]
    if aids and aids["flashcards"]:
        sections += ["## Flashcards", "\n\n".join(
            f"**Q{i}. {c['question']}**\n\n{c['answer']}" for i, c in enumerate(aids["flashcards"], 1))]
    if st.session_state.chat_messages:
        sections += ["## Questions & Answers", "\n\n".join(
            f"**Q: {m['content']}**" if m["role"] == "user" else _demote_headings(m["content"])
            for m in st.session_state.chat_messages)]
    sections += ["## Citation", format_citation(metadata, "APA"),
                 "```bibtex\n" + format_citation(metadata, "BibTeX") + "\n```"]
    return "\n\n".join(sections) + "\n"


def has_results():
    return bool(st.session_state.current_summary or st.session_state.analyses
                or st.session_state.follow_up_questions or st.session_state.chat_messages
                or st.session_state.study_aids)


def markdown_to_pdf(markdown_text, title):
    """Render Markdown to a PDF using PyMuPDF (no external tools needed)."""
    html = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "sane_lists"])
    css = (
        "body {font-family: sans-serif; font-size: 10.5pt; line-height: 1.4;} "
        "h1 {font-size: 18pt;} h2 {font-size: 14pt; margin-top: 12pt;} h3 {font-size: 12pt;} "
        "table {border-collapse: collapse;} td, th {border: 1px solid #999; padding: 3pt;} "
        "code, pre {font-family: monospace; font-size: 9pt;}"
    )
    story = pymupdf.Story(html=html, user_css=css)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page_rect = pymupdf.paper_rect("letter")
    content_rect = page_rect + (54, 54, -54, -54)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()

    with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as doc:
        doc.set_metadata({"title": title, "author": "", "creator": APP_TITLE, "producer": ""})
        return doc.tobytes(garbage=3, deflate=True)


def build_export_json():
    export_data = {
        "metadata": st.session_state.paper_metadata,
        "summary": st.session_state.current_summary,
        "summary_info": st.session_state.summary_info,
        "keywords": [
            {"keyword": word, "score": round(score, 4)}
            for word, score in extract_keywords(st.session_state.extracted_text)
        ],
        "analyses": st.session_state.analyses,
        "follow_up_questions": st.session_state.follow_up_questions,
        "chat": st.session_state.chat_messages,
        "study_aids": st.session_state.study_aids,
        "citations": {style: format_citation(st.session_state.paper_metadata, style) for style in CITATION_STYLES},
        "references": st.session_state.references,
        "tables": st.session_state.tables,
        "export_date": datetime.now().isoformat()
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)


# App UI Components
def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.header("Configuration")

    api_key = get_api_key()
    if not api_key:
        st.sidebar.warning(
            "No Gemini API key found. Add `GOOGLE_API_KEY=your-key` to a `.env` file "
            "(or the app's Secrets when deployed) and restart the app."
        )

    # Model selection
    models = []
    if api_key:
        try:
            models = list_available_models(api_key)
        except Exception:
            st.sidebar.caption("Could not load the model list; using the default model.")
    model_options = [DEFAULT_MODEL] + [m for m in models if m != DEFAULT_MODEL]
    model_option = st.sidebar.selectbox("Select Gemini Model", model_options, index=0)

    # Summary type selection
    summary_type = st.sidebar.radio(
        "Summary Type",
        list(SUMMARY_TYPES),
        format_func=SUMMARY_TYPES.get,
        index=0
    )

    # Analysis options
    st.sidebar.subheader("Advanced Analysis")
    analysis_options = st.sidebar.multiselect(
        "Select additional analyses",
        list(ANALYSIS_OPTIONS),
        default=list(ANALYSIS_OPTIONS)
    )

    # History management
    st.sidebar.subheader("History")
    if st.session_state.history:
        history = st.session_state.history
        selected_idx = st.sidebar.selectbox(
            "Previously processed papers",
            range(len(history)),
            format_func=lambda i: f"{i + 1}. {history[i]['metadata'].get('title') or history[i].get('filename') or 'Untitled Paper'}"
        )
        if st.sidebar.button("Load Selected Paper"):
            load_paper(history[selected_idx])
            st.rerun()
        if st.sidebar.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.sidebar.caption("Papers you summarize will appear here.")

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses Google's Gemini API to analyze and summarize research papers. "
        "It extracts key information, generates summaries, and provides visualizations to help "
        "understand research papers more effectively."
    )

    return model_option, summary_type, analysis_options


def render_input_panel(model_option):
    """Render the input section and return the selected input method"""
    st.header("Paper Input")

    if st.button(f"Try a sample paper: {SAMPLE_PAPER['label']}", icon=":material/science:",
                 help="Downloads the paper from arXiv so you can try the app without your own PDF."):
        with st.spinner("Downloading sample paper from arXiv..."):
            try:
                paper = lookup_paper(SAMPLE_PAPER["arxiv_id"])
            except (PaperLookupError, ValueError) as e:
                st.error(f"Could not load the sample paper: {e}")
                paper = None
        if paper:
            load_paper(paper)
            st.success("Sample paper loaded. Click Generate Summary to try it out.")

    upload_option = st.radio("Choose input method:",
                             ["Upload PDF", "Paste Text", "Upload Multiple PDFs", "DOI / arXiv Lookup"])

    if upload_option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
        if uploaded_file is None:
            st.session_state.last_upload_id = None  # so re-uploading the same file loads it again
        else:
            pdf_bytes = uploaded_file.getvalue()
            upload_id = paper_id_for(pdf_bytes)
            # Streamlit reruns the script on every interaction; only load a new upload once
            if upload_id != st.session_state.last_upload_id:
                with st.spinner("Extracting text and content from PDF..."):
                    try:
                        paper = process_pdf(pdf_bytes, uploaded_file.name)
                    except ValueError as e:
                        st.error(f"Error processing PDF: {e}")
                        return upload_option
                st.session_state.last_upload_id = upload_id
                load_paper(paper)
                st.success(
                    f"PDF processed successfully. Extracted {len(paper['text']):,} characters, "
                    f"{len(paper['figures'])} figures, and {len(paper['tables'])} tables.")

    elif upload_option == "Paste Text":
        with st.form("paste_form"):
            pasted = st.text_area("Paste the research paper text here:", height=400)
            submitted = st.form_submit_button("Analyze Text")
        if submitted:
            if len(pasted.strip()) < MIN_TEXT_CHARS:
                st.warning(f"Please paste at least {MIN_TEXT_CHARS} characters of text.")
            else:
                paper = build_paper(pasted.strip(), {}, source="Pasted text")
                with st.spinner("Reading paper details..."):
                    enrich_metadata(paper)
                load_paper(paper)
                st.success("Text loaded.")

    elif upload_option == "Upload Multiple PDFs":
        uploaded_files = st.file_uploader("Upload multiple research papers (PDF)", type=["pdf"],
                                          accept_multiple_files=True)
        papers = []
        for i, file in enumerate(uploaded_files or []):
            with st.spinner(f"Processing file {i + 1}/{len(uploaded_files)}..."):
                try:
                    papers.append(process_pdf(file.getvalue(), file.name))
                except ValueError as e:
                    st.error(f"Error processing {file.name}: {e}")
        st.session_state.processed_papers = papers

        if papers:
            st.write(f"Successfully processed {len(papers)} papers")
            chosen = st.selectbox(
                "Open a paper in the analysis panel",
                range(len(papers)),
                format_func=lambda i: f"{i + 1}. {papers[i]['metadata'].get('title') or papers[i]['filename']}"
            )
            if st.button("Open Paper"):
                load_paper(papers[chosen])
                st.rerun()
            if len(papers) >= 2:
                st.caption("Compare papers in the Paper Comparison section below.")

    elif upload_option == "DOI / arXiv Lookup":
        with st.form("lookup_form"):
            identifier = st.text_input("DOI or arXiv ID/URL", placeholder="10.48550/arXiv.1706.03762 or 1706.03762")
            submitted = st.form_submit_button("Look Up Paper")
        st.caption("arXiv papers are downloaded in full. For other DOIs the open-access PDF is used when "
                   "Crossref lists one; otherwise only the abstract is available.")
        if submitted:
            with st.spinner("Looking up paper..."):
                try:
                    paper = lookup_paper(identifier)
                except (PaperLookupError, ValueError) as e:
                    st.error(str(e))
                    return upload_option
            load_paper(paper)
            if paper.get("abstract_only"):
                st.warning("Only the abstract was available, so results will be limited to it.")
            else:
                st.success(f"Loaded full text ({len(paper['text']):,} characters).")

    if st.session_state.extracted_text:
        text = st.session_state.extracted_text
        with st.expander("View extracted content"):
            st.text_area("Text content (sample)", text[:1000] + ("..." if len(text) > 1000 else ""),
                         height=200, disabled=True)
            st.write(f"{len(st.session_state.figures)} figures, {len(st.session_state.tables)} tables, "
                     f"{len(st.session_state.references)} references")
        if prepare_text_for_model(text)[1]:
            st.caption(f"This paper is long; only the first {MAX_INPUT_CHARS:,} characters are sent to the model "
                       "(set MAX_INPUT_CHARS in .env to change this).")

    return upload_option


def run_ai_stream(label, func, *args):
    """Show a Gemini response as it is written; return the full text, or None on failure."""
    try:
        with st.spinner(label):
            chunks = func(*args, stream=True)
        text = st.write_stream(chunks)
    except GeminiError as e:
        st.error(str(e))
        return None
    if not isinstance(text, str):
        text = "".join(str(part) for part in text or [])
    if not text.strip():
        st.error(EMPTY_RESPONSE_MESSAGE)
        return None
    return text.strip()


def run_ai_task(label, func, *args):
    """Run a Gemini-backed task with a spinner; show an error and return None on failure."""
    with st.spinner(label):
        try:
            return func(*args)
        except GeminiError as e:
            st.error(str(e))
            return None


def render_output_panel(model_option, summary_type, analysis_options):
    """Render the output section"""
    st.header("Analysis Output")

    if not st.session_state.extracted_text:
        if st.session_state.processed_papers:
            st.info("Open one of the uploaded papers to analyze it, or compare papers below.")
        else:
            st.info("Please upload or paste a research paper to analyze, or try the sample paper")
        return

    # Tab-based interface for different outputs
    tabs = st.tabs(["Summary", "Chat", "Analysis", "Study", "Visualization", "Export"])

    # Summary Tab
    with tabs[0]:
        label = "Generate New Summary" if st.session_state.current_summary else "Generate Summary"
        if st.button(label, type="primary"):
            summary = run_ai_stream("Generating summary with Gemini AI...", generate_summary,
                                  st.session_state.extracted_text, model_option, summary_type)
            if summary:
                st.session_state.current_summary = summary
                st.session_state.summary_info = f"{SUMMARY_TYPES[summary_type]} summary · {model_option}"
                _update_history()
                st.rerun()  # refresh the button label and export tab

        if st.session_state.current_summary:
            st.caption(st.session_state.summary_info)
            st.markdown(st.session_state.current_summary)

    # Chat Tab
    with tabs[1]:
        render_chat(model_option)

    # Analysis Tab
    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metadata")
            metadata = st.session_state.paper_metadata
            st.write(f"**Title:** {metadata.get('title') or 'Unknown'}")
            st.write(f"**Authors:** {format_authors(metadata)}")
            st.write(f"**Year:** {metadata.get('publication_year') or 'Unknown'}")
            st.write(f"**Journal/Conference:** {metadata.get('journal_or_conference') or 'Unknown'}")
            if metadata.get("doi"):
                st.write(f"**DOI:** {metadata['doi']}")

            # References
            if st.session_state.references:
                with st.expander(f"References ({len(st.session_state.references)})"):
                    for i, ref in enumerate(st.session_state.references):
                        st.write(f"{i + 1}. {ref}")

        with col2:
            if "Extract Keywords" in analysis_options:
                st.subheader("Keywords")
                keywords = extract_keywords(st.session_state.extracted_text)
                if keywords:
                    df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
                    df["Score"] = df["Score"].round(4)
                    st.dataframe(df, hide_index=True)
                else:
                    st.write("No keywords extracted")

        # Citation
        st.subheader("Cite This Paper")
        style = st.radio("Citation style", CITATION_STYLES, horizontal=True, label_visibility="collapsed")
        st.code(format_citation(st.session_state.paper_metadata, style), language=None, wrap_lines=True)
        st.caption("Built from the extracted paper details. Use the copy icon, and double-check it before submitting.")

        # In-depth analyses
        available = [ANALYSIS_OPTIONS[o] for o in analysis_options if ANALYSIS_OPTIONS[o]]
        st.subheader("Detailed Analysis")
        if available:
            analysis_type = st.selectbox("Select analysis type", available, format_func=ANALYSIS_TYPES.get)
            if st.button("Generate Analysis"):
                result = run_ai_stream("Generating detailed analysis...", generate_detailed_analysis,
                                       st.session_state.extracted_text, model_option, analysis_type)
                if result:
                    st.session_state.analyses[analysis_type] = result
                    _update_history()
                    st.rerun()
            for key, result in st.session_state.analyses.items():
                with st.expander(f"{ANALYSIS_TYPES[key]} Analysis", expanded=(key == analysis_type)):
                    st.markdown(result)
        else:
            st.caption("Enable an analysis type under Advanced Analysis in the sidebar.")

        # Follow-up questions
        st.subheader("Research Questions")
        if st.button("Generate Follow-up Questions"):
            questions = run_ai_stream("Generating questions...", generate_follow_up_questions,
                                      st.session_state.extracted_text, model_option)
            if questions:
                st.session_state.follow_up_questions = questions
                _update_history()
                st.rerun()
        if st.session_state.follow_up_questions:
            st.markdown(st.session_state.follow_up_questions)

    # Study Tab
    with tabs[3]:
        render_study_aids(model_option)

    # Visualization Tab
    with tabs[4]:
        viz_options = [name for name, option in (("Keyword Cloud", "Extract Keywords"),
                                                 ("Citations by Year", "Generate Citation Graph"),
                                                 ("Figures & Tables", "Extract Figures & Tables"))
                       if option in analysis_options]
        if not viz_options:
            st.caption("Enable keywords, citation graph, or figures & tables under Advanced Analysis in the sidebar.")
            viz_type = None
        else:
            viz_type = st.radio("Select visualization type", viz_options)

        if viz_type == "Keyword Cloud":
            st.subheader("Keyword Cloud")
            word_cloud_data = create_word_cloud_data(st.session_state.extracted_text)

            if word_cloud_data:
                top = word_cloud_data[:15]
                fig = px.bar(
                    x=[value for _, value in top],
                    y=[word for word, _ in top],
                    orientation='h',
                    title="Top Keywords by TF-IDF Score",
                    labels={"x": "Relative Importance", "y": ""}
                )
                fig.update_layout(height=500, yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough text to generate keyword visualization")

        elif viz_type == "Citations by Year":
            st.subheader("Citations by Year")
            citation_data = generate_citation_graph(st.session_state.references)

            if citation_data:
                fig = px.bar(
                    x=[year for year, _ in citation_data],
                    y=[count for _, count in citation_data],
                    title="Citations by Publication Year",
                    labels={"x": "Year", "y": "Number of Citations"}
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough references to generate citation graph")

        elif viz_type == "Figures & Tables":
            st.subheader("Extracted Figures & Tables")

            if st.session_state.figures:
                st.write(f"Displaying {len(st.session_state.figures)} extracted figures")
                cols = st.columns(2)
                for i, figure in enumerate(st.session_state.figures):
                    with cols[i % 2]:
                        st.image(figure["data"], caption=f"Figure from page {figure['page']}", width="stretch")
            else:
                st.info("No figures extracted from this document")

            if st.session_state.tables:
                st.write(f"Displaying {len(st.session_state.tables)} detected tables")
                for table in st.session_state.tables:
                    with st.expander(f"Table from page {table['page']}"):
                        st.dataframe(pd.DataFrame(table["rows"]), hide_index=True)
            else:
                st.info("No tables detected in this document")

    # Export Tab
    with tabs[5]:
        st.subheader("Export Options")

        if not has_results():
            st.info("Generate a summary, analysis, chat answer, or study aids first; everything you create is included.")
            return

        title = st.session_state.paper_metadata.get("title") or "Research Paper Summary"
        base_name = f"{safe_filename(title)}_summary"
        report = build_report_markdown()

        export_format = st.radio("Select export format", ["Markdown", "PDF", "JSON"], horizontal=True)
        if export_format == "Markdown":
            st.download_button("Download Markdown", report, file_name=f"{base_name}.md",
                               mime="text/markdown", on_click="ignore")
        elif export_format == "PDF":
            try:
                pdf_bytes = markdown_to_pdf(report, title)
            except Exception as e:
                st.error(f"Could not create the PDF: {e}")
            else:
                st.download_button("Download PDF", pdf_bytes, file_name=f"{base_name}.pdf",
                                   mime="application/pdf", on_click="ignore")
        else:
            st.download_button("Download JSON", build_export_json(), file_name=f"{base_name}.json",
                               mime="application/json", on_click="ignore")

        with st.expander("Preview"):
            st.markdown(report)


def _queue_question(question):
    st.session_state.pending_question = question


def _clear_chat():
    st.session_state.chat_messages = []
    _update_history()


def render_chat(model_option):
    """Ask questions about the paper in a chat"""
    messages = st.session_state.chat_messages

    if not messages:
        st.caption("Ask anything about this paper. Answers are based on the paper's text.")
        for question in CHAT_STARTERS:
            st.button(question, on_click=_queue_question, args=(question,), key=f"starter_{question}")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this paper")
    question = question or st.session_state.pop("pending_question", None)
    if question and question.strip():
        question = question.strip()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer = run_ai_stream("Reading the paper...", answer_question,
                                   st.session_state.extracted_text, question, messages, model_option)
        if answer:
            st.session_state.chat_messages = messages + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            _update_history()
            st.rerun()

    if messages:
        st.button("Clear Chat", on_click=_clear_chat)


def render_study_aids(model_option):
    """Glossary and flashcards generated from the paper"""
    aids = st.session_state.study_aids
    label = "Regenerate Study Aids" if aids else "Generate Study Aids"
    if st.button(label, type="secondary" if aids else "primary"):
        result = run_ai_task("Creating a glossary and flashcards...", generate_study_aids,
                             st.session_state.extracted_text, model_option)
        if result:
            st.session_state.study_aids = result
            _update_history()
            st.rerun()

    if not aids:
        st.caption("Creates a glossary of the paper's key terms and flashcards to test your understanding.")
        return

    if aids["glossary"]:
        st.subheader(f"Glossary ({len(aids['glossary'])} terms)")
        for item in aids["glossary"]:
            st.markdown(f"**{item['term']}**: {item['definition']}")

    if aids["flashcards"]:
        st.subheader(f"Flashcards ({len(aids['flashcards'])})")
        st.caption("Click a card to reveal the answer.")
        for i, card in enumerate(aids["flashcards"], 1):
            with st.expander(f"{i}. {card['question']}"):
                st.markdown(card["answer"])
        title = st.session_state.paper_metadata.get("title") or "paper"
        st.download_button("Download Flashcards (CSV)", flashcards_to_csv(aids["flashcards"]),
                           file_name=f"{safe_filename(title)}_flashcards.csv", mime="text/csv",
                           on_click="ignore", help="Question and answer columns; imports into Anki and Quizlet.")


def _update_history():
    """Save the current paper and its results to the session history"""
    entry = {
        "id": st.session_state.paper_id,
        "text": st.session_state.extracted_text,
        "metadata": dict(st.session_state.paper_metadata),
        "figures": st.session_state.figures,
        "tables": st.session_state.tables,
        "references": st.session_state.references,
        "filename": st.session_state.paper_source,
        "summary": st.session_state.current_summary,
        "summary_info": st.session_state.summary_info,
        "analyses": dict(st.session_state.analyses),
        "follow_up_questions": st.session_state.follow_up_questions,
        "chat_messages": list(st.session_state.chat_messages),
        "study_aids": st.session_state.study_aids,
        "timestamp": datetime.now().isoformat()
    }

    history = [h for h in st.session_state.history if h["id"] != entry["id"]]
    history.append(entry)

    # Keep only the most recent items
    st.session_state.history = history[-HISTORY_LIMIT:]


def handle_paper_comparison(model_option):
    """Handle comparing multiple papers"""
    papers = st.session_state.processed_papers
    if len(papers) < 2:
        return

    st.header("Paper Comparison")

    selected = st.multiselect(
        "Select papers to compare",
        range(len(papers)),
        format_func=lambda i: f"{i + 1}. {papers[i]['metadata'].get('title') or papers[i]['filename']}"
    )
    comparison_type = st.radio("Select comparison type", list(COMPARISON_FOCUS), horizontal=True)

    if len(selected) < 2:
        st.info("Please select at least two papers to compare")
    elif st.button("Compare Selected Papers", type="primary"):
        comparison = run_ai_task(
            f"Comparing {len(selected)} papers (each is summarized first, in parallel)...",
            compare_papers, [papers[i]["text"] for i in selected], model_option, comparison_type
        )
        if comparison:
            st.session_state.comparison = comparison

    if st.session_state.comparison:
        st.markdown(st.session_state.comparison)
        st.download_button(
            "Download Comparison",
            st.session_state.comparison,
            file_name="paper_comparison.md",
            mime="text/markdown",
            on_click="ignore"
        )


# Main application function
def main():
    # Display header
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload academic papers to generate summaries, extract key information, and visualize content using Gemini AI."
    )

    # Render sidebar and get options
    model_option, summary_type, analysis_options = render_sidebar()

    # Main content area with columns
    col1, col2 = st.columns([1, 1])

    with col1:
        input_method = render_input_panel(model_option)

    with col2:
        render_output_panel(model_option, summary_type, analysis_options)

    # Handle paper comparison if multiple papers uploaded
    if input_method == "Upload Multiple PDFs":
        handle_paper_comparison(model_option)

    # Footer
    st.divider()


if __name__ == "__main__":
    main()
