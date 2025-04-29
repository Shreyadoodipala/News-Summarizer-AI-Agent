from agent import run_workflow
import asyncio
import streamlit as st

def run_async_function(query, num_searches_remaining, num_articles_tldr, language, sort_by, searchIn):
    if searchIn is not None and not isinstance(searchIn, list):
        raise ValueError("searchIn param should be a list of ['title', 'content', 'description']")
    return asyncio.run(run_workflow(query, num_searches_remaining, num_articles_tldr, language, sort_by, searchIn))
    #sources: Optional[str] = None,

lang_map = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl",
    "Portugese": "pt",
    "Russian": "ru",
    "Chinese": "zh"
}

st.title("News Agent")

# User Input
query = st.text_input("Enter your query:", max_chars=500)

with st.sidebar:
    num_searches_remaining = st.number_input("Number of Searches (Search depth)", min_value=1, max_value=10, value=3)
    num_articles_tldr = st.number_input("Number of Articles to summarize", min_value=1, max_value=10, value=2)
    searchIn = st.multiselect("Search in", ["title", "content", "description"], default=["title", "content", "description"])
    lang_name = st.selectbox("Language", list(lang_map.keys()))
    language = lang_map.get(lang_name)
    sort_by = st.selectbox("Sort By", ["relevancy", "popularity", "publishedAt"])


if st.button("Search News"):
    with st.spinner("Fetching news..."):
        result = run_async_function(query, num_searches_remaining, num_articles_tldr, language, sort_by, searchIn)
        st.write(result)