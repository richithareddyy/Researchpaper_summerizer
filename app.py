import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
import re
import requests
import base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from io import BytesIO
import tempfile
from PIL import Image
import fitz  # PyMuPDF
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Advanced Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'processed_papers' not in st.session_state:
    st.session_state.processed_papers = []
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ""
if 'paper_metadata' not in st.session_state:
    st.session_state.paper_metadata = {}
if 'figures' not in st.session_state:
    st.session_state.figures = []
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'references' not in st.session_state:
    st.session_state.references = []

# Custom CSS
st.markdown("""
<style>
    .main-header {color: #1E88E5; font-size: 40px; font-weight: bold; margin-bottom: 20px; text-align: center;}
    .section-header {color: #0D47A1; font-size: 24px; font-weight: bold; margin-top: 20px; margin-bottom: 10px;}
    .subsection-header {color: #1565C0; font-size: 18px; font-weight: bold; margin-top: 10px;}
    .info-box {background-color: #E3F2FD; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
    .success-box {background-color: #E8F5E9; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
    .warning-box {background-color: #FFF8E1; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
    .error-box {background-color: #FFEBEE; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
</style>
""", unsafe_allow_html=True)


# Define enhanced functions for text extraction
def extract_text_from_pdf(pdf_file):
    """Extract text, figures, tables and metadata from PDF."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    # Extract text using PyPDF2
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    metadata = {}

    # Get metadata
    if pdf_reader.metadata:
        for key in pdf_reader.metadata:
            if key and pdf_reader.metadata[key]:
                clean_key = key.strip('/').lower()
                metadata[clean_key] = pdf_reader.metadata[key]

    # Extract text
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    # Use PyMuPDF to extract images and tables
    figures = []
    tables = []

    try:
        doc = fitz.open(tmp_path)

        # Extract images
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Create a PIL image
                image = Image.open(BytesIO(image_bytes))

                # Convert to base64 for displaying
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Save figure info
                figures.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "data": img_str,
                    "width": image.width,
                    "height": image.height
                })

        # Try to identify tables through heuristics
        for page_num, page in enumerate(doc):
            # Look for table-like structures using text blocks
            blocks = page.get_text("blocks")
            for block_num, block in enumerate(blocks):
                # Simple heuristic: blocks with many spaces and few newlines might be tables
                text_block = block[4]
                spaces = text_block.count(' ')
                newlines = text_block.count('\n')

                if spaces > 20 and newlines > 3 and spaces / len(text_block) > 0.15:
                    tables.append({
                        "page": page_num + 1,
                        "index": block_num,
                        "text": text_block
                    })

    except Exception as e:
        st.warning(f"Error extracting images/tables: {e}")

    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

    return text, metadata, figures, tables


def extract_references(text):
    """Extract references section from the paper"""
    references = []

    # Try to find references section
    references_pattern = re.compile(r'references|bibliography|works cited', re.IGNORECASE)
    matches = list(references_pattern.finditer(text))

    if matches:
        # Get the last match which is likely the actual references section
        ref_start = matches[-1].start()
        ref_text = text[ref_start:]

        # Split by common reference patterns
        ref_entries = re.split(r'\[\d+\]|\n\d+\.|\n[A-Z][a-z]+,', ref_text)

        # Clean up entries
        if len(ref_entries) > 1:
            for entry in ref_entries[1:]:  # Skip the header
                clean_entry = entry.strip()
                if len(clean_entry) > 20:  # Minimum length to be a valid reference
                    references.append(clean_entry)

    return references


def clean_text(text):
    """Clean extracted text for better processing"""
    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove headers and footers (common patterns)
    text = re.sub(r'\b(?:Page \d+|^\d+$)', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Fix broken words
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)

    return text.strip()


def split_into_sections(text):
    """Split the paper into sections based on common headings"""
    # Define common section headings
    section_patterns = [
        r'\n\d+\.?\s+[A-Z][A-Za-z\s]+\n',  # Numbered sections: "1. Introduction"
        r'\n[A-Z][A-Z\s]+\n',  # ALL CAPS sections: "INTRODUCTION"
        r'\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\n'  # Title Case sections: "Introduction" or "Related Work"
    ]

    # Combine patterns
    combined_pattern = '|'.join(section_patterns)

    # Find all section headings
    headings = re.finditer(combined_pattern, text)

    sections = []
    last_pos = 0

    for match in headings:
        if last_pos > 0:
            section_text = text[last_pos:match.start()].strip()
            if section_text:
                sections.append({
                    "heading": current_heading,
                    "text": section_text
                })

        current_heading = match.group().strip()
        last_pos = match.end()

    if last_pos < len(text):
        section_text = text[last_pos:].strip()
        if section_text:
            sections.append({
                "heading": current_heading if 'current_heading' in locals() else "Content",
                "text": section_text
            })

    return sections


# Advanced analysis functions
def extract_keywords(text, top_n=10):
    """Extract important keywords from the paper using TF-IDF"""
    if not text or len(text) < 100:
        return []

    # Remove common symbols and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    try:
        # Generate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform([text])

        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # Sort keywords by score
        keyword_scores = sorted(
            [(feature_names[i], scores[i]) for i in range(len(feature_names))],
            key=lambda x: x[1],
            reverse=True
        )

        return keyword_scores[:top_n]
    except:
        return []


def create_word_cloud_data(text):
    """Prepare data for word cloud visualization"""
    # Extract keywords with scores
    keywords = extract_keywords(text, top_n=50)

    if not keywords:
        return []

    # Normalize scores for visualization
    max_score = max([score for _, score in keywords])
    normalized = [(word, int((score / max_score) * 100)) for word, score in keywords]

    return normalized


def generate_citation_graph(references):
    """Generate a simple citation network visualization data"""
    if not references or len(references) < 3:
        return None

    # Extract years from references (simple heuristic)
    years = []
    for ref in references:
        year_match = re.search(r'(19|20)\d{2}', ref)
        if year_match:
            years.append(int(year_match.group(0)))

    if not years:
        return None

    # Count citations by year
    year_counts = {}
    for year in years:
        if year in year_counts:
            year_counts[year] += 1
        else:
            year_counts[year] = 1

    # Convert to list of (year, count) tuples
    data = [(year, count) for year, count in year_counts.items()]
    data.sort(key=lambda x: x[0])  # Sort by year

    return data


# Define functions for generating summaries using Gemini API
def generate_summary(text, model_name="models/gemini-1.5-pro", summary_type="comprehensive"):
    """Generate paper summary using Gemini API"""
    model = genai.GenerativeModel(model_name)

    # Truncate text if too long (Gemini has input limits)
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    prompts = {
        "comprehensive": f"""
        You are a research assistant specialized in summarizing academic papers.
        Provide a comprehensive summary of the following research paper:

        {text}

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

        {text}

        Format your response in Markdown.
        """,

        "technical": f"""
        Provide a technical summary of the following research paper, focusing on the methodology, 
        technical innovations, algorithms, and experimental results:

        {text}

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

        {text}

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

        {text}

        Keep your explanation under 500 words and make it engaging and easy to understand.
        Format your response in Markdown.
        """
    }

    try:
        response = model.generate_content(prompts[summary_type])
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"


def generate_detailed_analysis(text, model_name="models/gemini-1.5-pro", analysis_type="methodology"):
    """Generate detailed analysis of specific aspects of the paper"""
    model = genai.GenerativeModel(model_name)

    # Truncate text if too long
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    prompts = {
        "methodology": f"""
        Provide a detailed analysis of the methodology used in this research paper:

        {text}

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

        {text}

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

        {text}

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

        {text}

        Focus on:
        1. Industry applications
        2. Policy implications
        3. Practical tools or frameworks that could be developed
        4. Potential beneficiaries of this research
        5. Steps needed to translate this research into practice

        Format your response in Markdown.
        """
    }

    try:
        response = model.generate_content(prompts[analysis_type])
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"


def generate_follow_up_questions(text, model_name="models/gemini-1.5-pro"):
    """Generate insightful follow-up questions about the paper"""
    model = genai.GenerativeModel(model_name)

    prompt = f"""
    Read the following research paper and generate 5 insightful follow-up questions that a researcher
    might ask after reading this paper. These questions should probe deeper into the methodology,
    explore limitations, suggest extensions, or connect to broader research themes.

    {text[:20000]}  # Limit text size

    Format your response as a numbered list in Markdown.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating questions: {str(e)}"


def extract_paper_details(text, model_name="models/gemini-1.5-pro"):
    """Extract structured metadata from the paper using Gemini"""
    model = genai.GenerativeModel(model_name)

    prompt = f"""
    Extract the following metadata from this research paper:

    {text[:5000]}  # Use just the beginning where metadata is typically found

    Return the results in JSON format with these fields:
    1. title (string): The paper's title
    2. authors (array of strings): List of authors
    3. publication_year (number or null): Year of publication
    4. journal_or_conference (string or null): Publication venue
    5. keywords (array of strings): Author-provided keywords
    6. doi (string or null): DOI if present

    Format as valid JSON without explanations.
    """

    try:
        response = model.generate_content(prompt)
        try:
            # Try to parse as JSON
            return json.loads(response.text)
        except:
            # If parsing fails, extract with regex as fallback
            title_match = re.search(r'"title":\s*"([^"]+)"', response.text)
            title = title_match.group(1) if title_match else "Unknown Title"

            authors_match = re.search(r'"authors":\s*\[(.*?)\]', response.text, re.DOTALL)
            authors = []
            if authors_match:
                authors_text = authors_match.group(1)
                author_matches = re.findall(r'"([^"]+)"', authors_text)
                authors = author_matches if author_matches else ["Unknown Authors"]

            year_match = re.search(r'"publication_year":\s*(\d{4}|null)', response.text)
            year = year_match.group(1) if year_match else None

            return {
                "title": title,
                "authors": authors,
                "publication_year": year,
                "journal_or_conference": None,
                "keywords": [],
                "doi": None
            }
    except Exception as e:
        return {
            "title": "Unknown Title",
            "authors": ["Unknown Authors"],
            "publication_year": None,
            "journal_or_conference": None,
            "keywords": [],
            "doi": None,
            "error": str(e)
        }


def compare_papers(papers_text, model_name="models/gemini-1.5-pro"):
    """Compare multiple research papers"""
    if len(papers_text) < 2:
        return "Need at least two papers to compare"

    model = genai.GenerativeModel(model_name)

    # Create summaries of each paper first to reduce token usage
    summaries = []
    for i, text in enumerate(papers_text):
        # Generate a brief summary
        brief_summary = generate_summary(text[:30000], model_name, "executive")
        summaries.append(f"Paper {i + 1}: {brief_summary}")

    combined_summaries = "\n\n".join(summaries)

    prompt = f"""
    Compare and contrast the following research papers based on these summaries:

    {combined_summaries}

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

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating comparison: {str(e)}"


# Save and load functions
def save_summary_to_file(summary, metadata, filename=None):
    """Save the summary and metadata to a markdown file"""
    if not filename:
        # Generate filename based on paper title
        title = metadata.get("title", "research_paper")
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.md"

    content = f"""# {metadata.get('title', 'Research Paper Summary')}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Paper Details
- **Authors**: {', '.join(metadata.get('authors', ['Unknown']))}
- **Year**: {metadata.get('publication_year', 'Unknown')}
- **Journal/Conference**: {metadata.get('journal_or_conference', 'Unknown')}
- **DOI**: {metadata.get('doi', 'Unknown')}

## Summary
{summary}
"""

    # Create downloads directory if it doesn't exist
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(exist_ok=True)

    file_path = downloads_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


# App UI Components
def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.header("Configuration")

    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Gemini Model",
        ["gemini-1.5-pro", "gemini-1.5-flash"],
        index=0
    )

    # Summary type selection
    summary_type = st.sidebar.radio(
        "Summary Type",
        ["comprehensive", "executive", "technical", "critique", "eli5"],
        index=0
    )

    # Analysis options
    st.sidebar.subheader("Advanced Analysis")
    analysis_options = st.sidebar.multiselect(
        "Select additional analyses",
        ["Extract Keywords", "Analyze Methodology", "Identify Future Research",
         "Extract Figures & Tables", "Generate Citation Graph", "Find Practical Applications"],
        default=["Extract Keywords"]
    )

    # API key input
    st.sidebar.subheader("API Configuration")
    api_key_input = st.sidebar.text_input("Enter your Google API Key (if not in .env)", type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)

    # History management
    st.sidebar.subheader("History")
    if st.session_state.history:
        history_titles = [f"{i + 1}. {h.get('title', 'Paper ' + str(i + 1))}" for i, h in
                          enumerate(st.session_state.history)]
        selected_history = st.sidebar.selectbox("Previously processed papers", history_titles)
        if st.sidebar.button("Load Selected Paper"):
            idx = int(selected_history.split('.')[0]) - 1
            if 0 <= idx < len(st.session_state.history):
                st.session_state.extracted_text = st.session_state.history[idx].get('text', '')
                st.session_state.current_summary = st.session_state.history[idx].get('summary', '')
                st.session_state.paper_metadata = st.session_state.history[idx].get('metadata', {})
                st.session_state.figures = st.session_state.history[idx].get('figures', [])
                st.session_state.tables = st.session_state.history[idx].get('tables', [])
                st.session_state.references = st.session_state.history[idx].get('references', [])
                st.rerun()

    if st.sidebar.button("Clear History"):
        st.session_state.history = []

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses Google's Gemini API to analyze and summarize research papers. "
        "It extracts key information, generates summaries, and provides visualizations to help "
        "understand research papers more effectively."
    )

    return model_option, summary_type, analysis_options


def render_input_panel():
    """Render the input section"""
    st.header("Paper Input")

    upload_option = st.radio("Choose input method:",
                             ["Upload PDF", "Paste Text", "Upload Multiple PDFs", "DOI Lookup (coming soon)"])

    input_text = st.session_state.extracted_text
    multiple_papers = []

    if upload_option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Extracting text and content from PDF..."):
                try:
                    text, metadata, figures, tables = extract_text_from_pdf(uploaded_file)
                    references = extract_references(text)

                    st.session_state.extracted_text = text
                    st.session_state.paper_metadata = metadata
                    st.session_state.figures = figures
                    st.session_state.tables = tables
                    st.session_state.references = references

                    input_text = text

                    st.success(
                        f"PDF processed successfully. Extracted {len(text)} characters, {len(figures)} figures, and {len(tables)} tables.")

                    # Try to extract paper details
                    try:
                        ai_metadata = extract_paper_details(text)
                        if ai_metadata and "title" in ai_metadata and ai_metadata["title"] != "Unknown Title":
                            st.session_state.paper_metadata.update(ai_metadata)
                    except:
                        pass

                    with st.expander("View extracted content"):
                        st.text_area("Text content (sample)", text[:1000] + "...", height=200)
                        if figures:
                            st.write(f"Extracted {len(figures)} figures")
                        if tables:
                            st.write(f"Extracted {len(tables)} potential tables")
                        if references:
                            st.write(f"Extracted {len(references)} references")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    elif upload_option == "Paste Text":
        input_text = st.text_area("Paste the research paper text here:", height=400,
                                  value=st.session_state.extracted_text)
        if input_text and input_text != st.session_state.extracted_text:
            st.session_state.extracted_text = input_text
            st.session_state.references = extract_references(input_text)

            # Try to extract paper details using AI
            try:
                ai_metadata = extract_paper_details(input_text)
                if ai_metadata and "title" in ai_metadata:
                    st.session_state.paper_metadata = ai_metadata
            except:
                pass

    elif upload_option == "Upload Multiple PDFs":
        uploaded_files = st.file_uploader("Upload multiple research papers (PDF)", type=["pdf"],
                                          accept_multiple_files=True)
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} files. Processing...")

            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing file {i + 1}/{len(uploaded_files)}..."):
                    try:
                        text, metadata, figures, tables = extract_text_from_pdf(file)
                        references = extract_references(text)

                        # Get paper details
                        try:
                            ai_metadata = extract_paper_details(text)
                            if ai_metadata and "title" in ai_metadata:
                                metadata.update(ai_metadata)
                        except:
                            pass

                        multiple_papers.append({
                            "text": text,
                            "metadata": metadata,
                            "filename": file.name,
                            "figures": figures,
                            "tables": tables,
                            "references": references
                        })

                        st.success(f"Processed {file.name} successfully")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

            if multiple_papers:
                st.session_state.processed_papers = multiple_papers
                st.write(f"Successfully processed {len(multiple_papers)} papers")

                paper_titles = [f"{i + 1}. {p['metadata'].get('title', p['filename'])}" for i, p in
                                enumerate(multiple_papers)]
                st.multiselect("Select papers to compare", paper_titles, key="selected_papers_for_comparison")

    return input_text, multiple_papers


def render_output_panel(model_option, summary_type, analysis_options):
    """Render the output section"""
    st.header("Analysis Output")

    if not st.session_state.extracted_text and not st.session_state.processed_papers:
        st.info("Please upload or paste a research paper to analyze")
        return

    # Tab-based interface for different outputs
    tabs = st.tabs(["Summary", "Analysis", "Visualization", "Export"])

    # Summary Tab
    with tabs[0]:
        if st.session_state.current_summary:
            st.markdown(st.session_state.current_summary)

            if st.button("Generate New Summary"):
                with st.spinner("Generating summary with Gemini AI..."):
                    try:
                        model_name = f"models/{model_option}"
                        summary = generate_summary(st.session_state.extracted_text, model_name, summary_type)
                        st.session_state.current_summary = summary

                        # Update history
                        _update_history(summary)

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
        else:
            if st.button("Generate Summary"):
                with st.spinner("Generating summary with Gemini AI..."):
                    try:
                        model_name = f"models/{model_option}"

                        # Multiple papers comparison
                        if "selected_papers_for_comparison" in st.session_state and st.session_state.selected_papers_for_comparison:
                            selected_indices = [int(s.split('.')[0]) - 1 for s in
                                                st.session_state.selected_papers_for_comparison]
                            selected_papers = [st.session_state.processed_papers[i] for i in selected_indices if
                                               i < len(st.session_state.processed_papers)]

                            if len(selected_papers) > 1:
                                papers_text = [p["text"] for p in selected_papers]
                                summary = compare_papers(papers_text, model_name)
                                st.session_state.current_summary = summary
                                st.rerun()
                            else:
                                st.warning("Please select at least two papers to compare")

                        # Single paper summary
                        else:
                            summary = generate_summary(st.session_state.extracted_text, model_name, summary_type)
                            st.session_state.current_summary = summary

                            # Update history
                            _update_history(summary)

                            st.rerun()
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

    # Analysis Tab
    with tabs[1]:
        if not st.session_state.extracted_text:
            st.info("Please process a paper first")
        else:
            # Create columns for different analyses
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Metadata")
                metadata = st.session_state.paper_metadata
                if metadata:
                    st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
                    st.write(f"**Authors:** {', '.join(metadata.get('authors', ['Unknown']))}")
                    st.write(f"**Year:** {metadata.get('publication_year', 'Unknown')}")
                    st.write(f"**Journal/Conference:** {metadata.get('journal_or_conference', 'Unknown')}")
                    if 'doi' in metadata and metadata['doi']:
                        st.write(f"**DOI:** {metadata['doi']}")
                else:
                    st.write("No metadata extracted")

                # References
                if st.session_state.references:
                    with st.expander(f"References ({len(st.session_state.references)})"):
                        for i, ref in enumerate(st.session_state.references):
                            st.write(f"{i + 1}. {ref}")

            with col2:
                # Keywords
                if "Extract Keywords" in analysis_options:
                    st.subheader("Keywords")
                    if st.button("Extract Keywords"):
                        with st.spinner("Extracting keywords..."):
                            keywords = extract_keywords(st.session_state.extracted_text)
                            if keywords:
                                # Create a dataframe for better display
                                df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
                                df["Score"] = df["Score"].round(4)
                                st.dataframe(df)
                            else:
                                st.write("No keywords extracted")

            # In-depth analyses
            st.subheader("Detailed Analysis")
            analysis_type = st.selectbox(
                "Select analysis type",
                ["methodology", "literature", "future_research", "practical_applications"]
            )

            if st.button("Generate Analysis"):
                with st.spinner("Generating detailed analysis..."):
                    model_name = f"models/{model_option}"
                    analysis = generate_detailed_analysis(
                        st.session_state.extracted_text,
                        model_name,
                        analysis_type
                    )
                    st.markdown(analysis)

            # Follow-up questions
            st.subheader("Research Questions")
            if st.button("Generate Follow-up Questions"):
                with st.spinner("Generating questions..."):
                    model_name = f"models/{model_option}"
                    questions = generate_follow_up_questions(
                        st.session_state.extracted_text,
                        model_name
                    )
                    st.markdown(questions)

    # Visualization Tab
    with tabs[2]:
        if not st.session_state.extracted_text:
            st.info("Please process a paper first")
        else:
            viz_type = st.radio(
                "Select visualization type",
                ["Keyword Cloud", "Citations by Year", "Figures & Tables"]
            )

            if viz_type == "Keyword Cloud":
                st.subheader("Keyword Cloud")
                word_cloud_data = create_word_cloud_data(st.session_state.extracted_text)

                if word_cloud_data:
                    words = [word for word, _ in word_cloud_data]
                    values = [value for _, value in word_cloud_data]

                    fig = px.bar(
                        x=values[:15],  # Top 15 keywords
                        y=words[:15],
                        orientation='h',
                        title="Top Keywords by TF-IDF Score",
                        labels={"x": "Relative Importance", "y": ""}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough text to generate keyword visualization")

            elif viz_type == "Citations by Year":
                st.subheader("Citations by Year")
                citation_data = generate_citation_graph(st.session_state.references)

                if citation_data:
                    # Create data for the chart
                    years = [year for year, _ in citation_data]
                    counts = [count for _, count in citation_data]

                    # Plot
                    fig = px.bar(
                        x=years,
                        y=counts,
                        title="Citations by Publication Year",
                        labels={"x": "Year", "y": "Number of Citations"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough references to generate citation graph")

            elif viz_type == "Figures & Tables":
                st.subheader("Extracted Figures & Tables")

                if st.session_state.figures:
                    st.write(f"Displaying {len(st.session_state.figures)} extracted figures")

                    # Show figures in a grid
                    cols = st.columns(2)
                    for i, fig in enumerate(st.session_state.figures):
                        col_idx = i % 2
                        with cols[col_idx]:
                            st.image(
                                f"data:image/png;base64,{fig['data']}",
                                caption=f"Figure from page {fig['page']}",
                                use_container_width =True
                            )
                else:
                    st.info("No figures extracted from this document")

                if st.session_state.tables:
                    st.write(f"Displaying {len(st.session_state.tables)} potential tables")
                    for i, table in enumerate(st.session_state.tables):
                        with st.expander(f"Table from page {table['page']}"):
                            st.text(table['text'])
                else:
                    st.info("No tables detected in this document")

    # Export Tab
    with tabs[3]:
        st.subheader("Export Options")

        export_format = st.radio(
            "Select export format",
            ["Markdown", "PDF (requires wkhtmltopdf installed)", "JSON"]
        )

        if st.button("Export Summary"):
            if not st.session_state.current_summary:
                st.warning("Please generate a summary first")
            else:
                try:
                    if export_format == "Markdown":
                        file_path = save_summary_to_file(
                            st.session_state.current_summary,
                            st.session_state.paper_metadata
                        )

                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read()

                        st.download_button(
                            label="Download Markdown Summary",
                            data=file_content,
                            file_name=file_path.name,
                            mime="text/markdown"
                        )
                        st.success(f"Summary saved as {file_path.name}")

                    elif export_format == "PDF":
                        st.warning("PDF export requires wkhtmltopdf to be installed on your system")
                        st.info("This feature would convert the Markdown summary to PDF")

                    elif export_format == "JSON":
                        # Create JSON with all data we've extracted and generated
                        export_data = {
                            "metadata": st.session_state.paper_metadata,
                            "summary": st.session_state.current_summary,
                            "keywords": extract_keywords(st.session_state.extracted_text),
                            "references": st.session_state.references,
                            "export_date": datetime.now().isoformat()
                        }

                        # Convert to JSON string
                        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

                        # Create download button
                        title = st.session_state.paper_metadata.get("title", "research_paper")
                        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

                        st.download_button(
                            label="Download JSON Data",
                            data=json_str,
                            file_name=f"{safe_title}_data.json",
                            mime="application/json"
                        )
                        st.success("JSON data ready for download")

                except Exception as e:
                    st.error(f"Error exporting summary: {e}")


def _update_history(summary):
    """Update the history with current paper info"""
    # Add to history if not already there
    title = st.session_state.paper_metadata.get("title", "Untitled Paper")

    # Check if we already have this paper in history
    exists = False
    for paper in st.session_state.history:
        if paper.get("title") == title:
            # Update existing entry
            paper["summary"] = summary
            paper["metadata"] = st.session_state.paper_metadata
            paper["figures"] = st.session_state.figures
            paper["tables"] = st.session_state.tables
            paper["references"] = st.session_state.references
            exists = True
            break

    # Add new entry if doesn't exist
    if not exists:
        st.session_state.history.append({
            "title": title,
            "text": st.session_state.extracted_text,
            "summary": summary,
            "metadata": st.session_state.paper_metadata,
            "figures": st.session_state.figures,
            "tables": st.session_state.tables,
            "references": st.session_state.references,
            "timestamp": datetime.now().isoformat()
        })

    # Keep only the 10 most recent items
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]


# Custom function to handle paper comparison
def handle_paper_comparison():
    """Handle comparing multiple papers"""
    if not st.session_state.processed_papers:
        return

    st.header("Paper Comparison")

    # List available papers
    paper_options = [f"{i + 1}. {p['metadata'].get('title', p['filename'])}"
                     for i, p in enumerate(st.session_state.processed_papers)]

    selected = st.multiselect("Select papers to compare", paper_options)
    if not selected or len(selected) < 2:
        st.info("Please select at least two papers to compare")
        return

    comparison_type = st.radio(
        "Select comparison type",
        ["Full Comparison", "Methodology Comparison", "Results Comparison"]
    )

    if st.button("Compare Selected Papers"):
        with st.spinner("Comparing papers..."):
            # Get indices of selected papers
            indices = [int(s.split('.')[0]) - 1 for s in selected]
            papers_text = [st.session_state.processed_papers[i]["text"] for i in indices
                           if i < len(st.session_state.processed_papers)]

            # Generate comparison based on type
            if comparison_type == "Full Comparison":
                comparison = compare_papers(papers_text)
            else:
                # For more specific comparisons, we could implement custom prompts
                comparison = compare_papers(papers_text)

            st.markdown(comparison)

            # Option to save comparison
            st.download_button(
                "Download Comparison",
                comparison,
                file_name="paper_comparison.md",
                mime="text/markdown"
            )


# Main application function
def main():
    # Display header
    st.markdown('<div class="main-header">Advanced Research Paper Summarizer</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload academic papers to generate summaries, extract key information, and visualize content using Gemini AI."
    )

    # Render sidebar and get options
    model_option, summary_type, analysis_options = render_sidebar()

    # Main content area with columns
    col1, col2 = st.columns([1, 1])

    with col1:
        input_text, multiple_papers = render_input_panel()

    with col2:
        render_output_panel(model_option, summary_type, analysis_options)

    # Handle paper comparison if multiple papers uploaded
    if multiple_papers:
        handle_paper_comparison()

    # Footer
    st.divider()

if __name__ == "__main__":
    main()