import os
import warnings

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import asyncio
import streamlit as st
from transformers import pipeline
from crawl4ai import AsyncWebCrawler
import re

st.set_page_config(page_title="BBC News QA", page_icon="📰")
st.title("📰 BBC News Headlines Q&A")
st.caption("Scrapes BBC News headlines, then lets you ask questions about them using a transformer model.")


@st.cache_resource(show_spinner="Loading QA model...")
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="distilbert/distilbert-base-cased-distilled-squad",
    )


def extract_headlines(markdown: str) -> list[str]:
    """Extract headlines from the crawled BBC News markdown."""
    headlines = []
    for line in markdown.splitlines():
        line = line.strip()
        # Markdown headings
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if text and len(text) > 10:
                headlines.append(text)
        # Markdown links that look like headlines (bold or standalone)
        match = re.match(r"\[(.+?)\]\(.+?\)", line)
        if match:
            text = match.group(1).strip()
            if len(text) > 15:
                headlines.append(text)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for h in headlines:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def scrape_bbc() -> tuple[list[str], str]:
    """Scrape BBC News and return (headlines, raw_context)."""

    async def _scrape():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url="https://www.bbc.com/news")
            return result.markdown

    markdown = asyncio.run(_scrape())
    headlines = extract_headlines(markdown)
    context = " | ".join(headlines)
    return headlines, context


# --- Sidebar ---
with st.sidebar:
    if st.button("🔄 Scrape BBC News", use_container_width=True):
        st.session_state.pop("headlines", None)
        st.session_state.pop("context", None)
        st.session_state.pop("qa_history", None)

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Click the button or wait for auto-scrape\n"
        "2. Headlines are extracted from BBC News\n"
        "3. Ask questions about the headlines\n"
        "4. A DistilBERT QA model answers from the scraped context"
    )

# --- Scrape headlines ---
if "headlines" not in st.session_state:
    with st.spinner("Scraping BBC News headlines..."):
        headlines, context = scrape_bbc()
        st.session_state.headlines = headlines
        st.session_state.context = context

headlines = st.session_state.headlines
context = st.session_state.context

# --- Display headlines ---
st.subheader(f"Headlines ({len(headlines)} found)")
with st.expander("View all headlines", expanded=False):
    for i, h in enumerate(headlines, 1):
        st.markdown(f"{i}. {h}")

# --- QA Section ---
st.subheader("Ask a question about the news")

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Show previous Q&A
for item in st.session_state.qa_history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        st.markdown(f"**{item['answer']}** (confidence: {item['score']:.1%})")

if question := st.chat_input("e.g., What is the top headline about?"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            qa = load_qa_pipeline()
            result = qa(question=question, context=context)
            answer = result["answer"]
            score = result["score"]
            st.markdown(f"**{answer}** (confidence: {score:.1%})")

    st.session_state.qa_history.append(
        {"question": question, "answer": answer, "score": score}
    )
