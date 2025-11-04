"""
Reddit AI Insights Tool - Streamlit Version
Analyze Reddit discussions with AI-powered sentiment and topic classification
"""

import streamlit as st
import os
import json
import re
import textwrap
from typing import List, Dict, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import praw
from praw.models import MoreComments

from openai import OpenAI

import nltk
from rake_nltk import Rake
from wordcloud import WordCloud
import spacy

from tenacity import retry, wait_exponential, stop_after_attempt

# Page config
st.set_page_config(
    page_title="Reddit AI Insights Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model... This may take a moment on first run.")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Initialize API clients
@st.cache_resource
def init_clients():
    """Initialize Reddit and OpenAI clients using Streamlit secrets"""
    try:
        reddit_client_id = st.secrets["REDDIT_CLIENT_ID"]
        reddit_client_secret = st.secrets["REDDIT_CLIENT_SECRET"]
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        reddit_user_agent = st.secrets.get("REDDIT_USER_AGENT", "reddit-insights-streamlit/1.0")

        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )

        oai = OpenAI(api_key=openai_api_key)

        return reddit, oai, openai_api_key
    except Exception as e:
        st.error(f"❌ Error initializing API clients: {str(e)}")
        st.info("Please configure your API keys in Streamlit secrets (Settings → Secrets)")
        st.stop()

# Topic definitions
TOPICS = [
    {
        "name": "Link Building",
        "description": "Acquiring relevant, high-quality backlinks from external sites via outreach, digital PR, content promotion, or partnerships. Includes anchor text, link quality/equity, rel=\"nofollow\"/\"sponsored\", disavow, and avoiding link schemes; excludes internal linking."
    },
    {
        "name": "Technical SEO",
        "description": "Site infrastructure and discoverability: crawling, indexation, rendering (incl. JavaScript), site speed/Core Web Vitals, mobile-friendliness, canonicalization, redirects, robots.txt, sitemaps, structured data, hreflang, and server responses. Not about content strategy or off-site links."
    },
    {
        "name": "Local SEO",
        "description": "Improving visibility in local/map results: Google Business Profile optimization (categories, hours, photos), NAP consistency, reviews, local citations/directories, location or service-area pages, and proximity signals."
    },
    {
        "name": "Tools",
        "description": "Discussions of SEO platforms/utilities used to audit/crawl sites, track rankings, analyze keywords/backlinks, or report performance (e.g., Search Console, site crawlers, rank trackers, keyword tools). Includes setup, comparisons, configurations, and interpreting outputs."
    },
    {
        "name": "Keyword Research",
        "description": "Finding and prioritizing search queries by volume, difficulty, and intent; expanding seed keywords; competitor/content-gap analysis; mapping keywords to pages and SERP features."
    },
    {
        "name": "On-page SEO",
        "description": "Optimizing on-site content and HTML elements to align with search intent: titles/title links, meta descriptions, headings, body copy, internal links, URLs, image alt text, and schema markup/E-E-A-T signals. Excludes external link acquisition."
    },
    {
        "name": "AI and SEO",
        "description": "Use and impact of generative AI on SEO workflows (ideation, drafting, optimization, clustering) and changes in search such as AI Overviews. Must align with Google guidance—AI content should be helpful and policy-compliant, not for manipulating rankings."
    }
]

# OpenAI helper functions
REASONING_TOKENS = {
    "minimal": 300,
    "low": 1000,
    "medium": 5000,
    "high": 10000,
}

def _is_gpt5_model(name: str) -> bool:
    if not name: return False
    norm = re.sub(r"[\s_]+", "-", name.strip().lower())
    return norm.startswith("gpt-5")

def _map_reasoning_effort(level: str) -> str | None:
    if not level: return None
    level = level.strip().lower()
    if level in {"off", "none", "false", "0"}:
        return None
    if level in {"minimal", "low", "medium", "high"}:
        return level
    return "low"

def _responses_create_json(
    oai: OpenAI,
    input_text: str,
    instructions: str,
    model: str,
    reasoning_level: str,
    *,
    temperature: float | None = 0,
    max_output_tokens: int = 200,
) -> str:
    is_gpt5 = _is_gpt5_model(model)
    kwargs = dict(model=model, input=input_text, instructions=instructions)
    kwargs["response_format"] = {"type": "json_object"}

    eff_level = _map_reasoning_effort(reasoning_level) if is_gpt5 else None
    effective_max_tokens = int(max_output_tokens)

    if is_gpt5:
        if eff_level:
            boost = int(REASONING_TOKENS.get(eff_level, REASONING_TOKENS["low"]))
            effective_max_tokens = max(effective_max_tokens, boost)
            kwargs["reasoning"] = {"effort": eff_level}
    else:
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

    kwargs["max_output_tokens"] = effective_max_tokens

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = oai.responses.create(**kwargs)
            return resp.output_text
        except Exception as e:
            err = str(e)
            lower = err.lower()
            changed = False

            m = re.search(r"unexpected keyword argument '([^']+)'", err)
            if m:
                bad = m.group(1)
                if bad in kwargs:
                    kwargs.pop(bad, None); changed = True

            if not changed and ("unsupported" in lower or "not supported" in lower):
                for bad in ("response_format", "temperature", "reasoning", "max_output_tokens"):
                    if bad in lower and bad in kwargs:
                        kwargs.pop(bad, None); changed = True
                if "max_output_tokens" in lower and "max_output_tokens" in kwargs:
                    val = kwargs.pop("max_output_tokens")
                    kwargs["max_tokens"] = val
                    changed = True

            if changed and attempt < 4:
                continue
            raise

def _safe_json_parse(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{"); end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise ValueError("Could not parse JSON from model output")

# Fetch Reddit data
def fetch_reddit_data(reddit, subreddit: str, num_threads: int, num_comments_total: int,
                     listing: str = "hot", max_chars: int = 4000) -> pd.DataFrame:
    sub = reddit.subreddit(subreddit)

    if listing == "new":
        submissions = list(sub.new(limit=num_threads))
    elif listing == "top":
        submissions = list(sub.top(limit=num_threads))
    else:
        submissions = list(sub.hot(limit=num_threads))

    records = []
    for s in submissions:
        text = " ".join([t for t in [s.title or "", s.selftext or ""] if t]).strip()
        records.append({
            "type": "thread",
            "id": s.id,
            "parent_id": None,
            "link_id": s.id,
            "subreddit": str(s.subreddit),
            "author": str(getattr(s, "author", None)) if getattr(s, "author", None) else None,
            "score": int(getattr(s, "score", 0)),
            "created_utc": pd.to_datetime(getattr(s, "created_utc", None), unit="s", utc=True) if getattr(s, "created_utc", None) else None,
            "title": s.title or "",
            "body": s.selftext or "",
            "permalink": f"https://www.reddit.com{s.permalink}" if getattr(s, "permalink", None) else None,
            "url": getattr(s, "url", None),
            "text": text[:max_chars]
        })

    grabbed = 0
    for s in submissions:
        if grabbed >= num_comments_total:
            break
        s.comments.replace_more(limit=0)
        for c in s.comments.list():
            if grabbed >= num_comments_total:
                break
            if isinstance(c, MoreComments):
                continue
            body = (c.body or "").strip()
            if body.lower() in {"[deleted]", "[removed]"} or not body:
                continue
            records.append({
                "type": "comment",
                "id": c.id,
                "parent_id": c.parent_id,
                "link_id": c.link_id.replace("t3_", "") if hasattr(c, "link_id") else s.id,
                "subreddit": str(s.subreddit),
                "author": str(getattr(c, "author", None)) if getattr(c, "author", None) else None,
                "score": int(getattr(c, "score", 0)),
                "created_utc": pd.to_datetime(getattr(c, "created_utc", None), unit="s", utc=True) if getattr(c, "created_utc", None) else None,
                "title": "",
                "body": body,
                "permalink": f"https://www.reddit.com{c.permalink}" if getattr(c, "permalink", None) else None,
                "url": None,
                "text": body[:max_chars]
            })
            grabbed += 1

    return pd.DataFrame.from_records(records)

# Sentiment classification
@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def classify_sentiment(oai: OpenAI, text: str, model: str, reasoning_level: str, max_chars: int = 4000) -> Tuple[str, float]:
    if not text or not text.strip():
        return ("Neutral", 0.5)

    instructions = "You are a meticulous Reddit sentiment rater. Always respond with a SINGLE JSON object and nothing else."

    prompt = textwrap.dedent(f"""
    Task: Classify the sentiment of the Reddit text into EXACTLY one of:
    ["Positive", "Neutral", "Negative", "Mixed"].

    Output JSON ONLY with this schema:
    {{
      "sentiment": "Positive|Neutral|Negative|Mixed",
      "confidence": 0.0_to_1.0
    }}

    Text:
    \"\"\"{text[:max_chars].strip()}\"
    \"\"\"
    """)

    raw = _responses_create_json(oai, prompt, instructions, model, reasoning_level, temperature=0, max_output_tokens=100)
    data = _safe_json_parse(raw)
    sent = data.get("sentiment", "Neutral")
    conf = float(data.get("confidence", 0.5))
    return (sent, conf)

# Topic classification
@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def classify_topics(oai: OpenAI, text: str, topics: List[Dict[str, str]], model: str,
                   reasoning_level: str, max_chars: int = 4000) -> Tuple[List[str], str, Dict[str, float]]:
    if not text or not text.strip():
        return (["Other"], "Other", {"Other": 0.5})

    catalog = {t["name"]: t["description"] for t in topics}
    topic_names = list(catalog.keys())

    instructions = "You are an expert Reddit topic classifier. Return a SINGLE JSON object only, with keys 'labels', 'top_topic', and 'confidence'."

    prompt = textwrap.dedent(f"""
    Task: Assign zero or more relevant topics to the Reddit text from the allowed set below.
    Also select 'top_topic' as the single best-fitting topic.

    Allowed topics (name -> description):
    {json.dumps(catalog, indent=2)}

    If nothing fits, use "Other" as both the only label and the top_topic.

    Output JSON ONLY with this schema:
    {{
      "labels": ["TopicName", ...],
      "top_topic": "TopicName",
      "confidence": {{
         "TopicName": 0.0_to_1.0,
         "...": 0.0_to_1.0
      }}
    }}

    Text:
    \"\"\"{text[:max_chars].strip()}\"
    \"\"\"
    """)

    raw = _responses_create_json(oai, prompt, instructions, model, reasoning_level, temperature=0, max_output_tokens=200)
    data = _safe_json_parse(raw)
    labels = data.get("labels") or []
    if not isinstance(labels, list):
        labels = [str(labels)]
    top_topic = data.get("top_topic") or (labels[0] if labels else "Other")
    conf = data.get("confidence") or {}
    try:
        conf = {k: float(v) for k, v in conf.items()}
    except Exception:
        conf = {}
    if not labels:
        labels = ["Other"]
        top_topic = "Other"
        conf = {"Other": conf.get("Other", 0.5)}
    return (labels, top_topic, conf)

# Extract keywords using RAKE
def extract_rake_keywords(
    texts: List[str],
    top_n: int = 100,
    min_len: int = 1,
    max_len: int = 3,
) -> List[Tuple[str, float]]:
    r = Rake(min_length=min_len, max_length=max_len)
    r.extract_keywords_from_text("\n".join([t for t in texts if t]))
    raw = r.get_ranked_phrases_with_scores()

    cleaned = {}
    for score, phrase in raw:
        p = " ".join((phrase or "").split())
        p = re.sub(r"(https?://\S+|www\.\S+)", "", p)
        p = p.strip(" .,:;!?()[]{}\"'`•–-").strip()
        if not p:
            continue
        k = p.casefold()
        if k not in cleaned or float(score) > cleaned[k][0]:
            cleaned[k] = (float(score), p)

    top = sorted(cleaned.values(), key=lambda x: x[0], reverse=True)[:top_n]
    return [(p, float(s)) for s, p in top]

# Extract entities using spaCy
def extract_entities_spacy(nlp, texts: List[str]) -> Counter:
    ALLOWED_NER_LABELS = {"PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","FAC","NORP"}
    counts = Counter()
    for doc in nlp.pipe([t for t in texts if t], disable=["tagger","parser","lemmatizer"]):
        for ent in doc.ents:
            if ent.label_ in ALLOWED_NER_LABELS:
                token = ent.text.strip()
                if token:
                    counts[token] += 1
    return counts

# Generate word cloud
def generate_wordcloud(freq_pairs, title: str = "Word Cloud"):
    if isinstance(freq_pairs, dict):
        items = list(freq_pairs.items())
    else:
        items = list(freq_pairs)

    if not items:
        return None

    vals = np.array([s for _, s in items], dtype=float)
    vals = np.power(vals, 0.7)
    items = sorted(zip([p for p, _ in items], vals), key=lambda x: x[1], reverse=True)[:120]
    freq_map = dict(items)

    wc = WordCloud(
        width=1800,
        height=900,
        scale=2,
        background_color="white",
        max_words=120,
        min_font_size=16,
        max_font_size=240,
        prefer_horizontal=0.95,
        relative_scaling=0.3,
        collocations=False,
        normalize_plurals=True,
        repeat=True,
        margin=1,
    ).generate_from_frequencies(freq_map)

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    return fig

# Plot sentiment distribution
def plot_sentiment_distribution(frame: pd.DataFrame, title: str = "Sentiment distribution"):
    order = ["Positive", "Neutral", "Negative", "Mixed"]
    counts = frame["sentiment"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", rot=0, ax=ax, color=['#2ecc71', '#95a5a6', '#e74c3c', '#f39c12'])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom")
    plt.tight_layout()
    return fig

# Plot topic distribution
def plot_topic_distribution(frame: pd.DataFrame, title: str = "Topic distribution"):
    counts = frame["top_topic"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", rot=45, ax=ax, color='#3498db')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Top topic", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom", rotation=0)
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.title("🔍 Reddit AI Insights Tool")
    st.markdown("Analyze Reddit discussions with AI-powered sentiment and topic classification")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Reddit settings
    st.sidebar.subheader("Reddit Settings")
    subreddit = st.sidebar.text_input("Subreddit", value="SEO", help="Enter subreddit name without r/")
    num_threads = st.sidebar.slider("Number of Threads", min_value=1, max_value=50, value=5)
    num_comments = st.sidebar.slider("Number of Comments", min_value=5, max_value=200, value=15)
    listing_type = st.sidebar.selectbox("Listing Type", ["hot", "new", "top"])

    # OpenAI settings
    st.sidebar.subheader("OpenAI Settings")
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"],
        index=1,
        help="GPT-5 models may require account verification"
    )
    reasoning_level = st.sidebar.selectbox(
        "Reasoning Level",
        ["minimal", "low", "medium", "high"],
        index=1,
        help="Higher reasoning uses more tokens but may improve accuracy"
    )

    # Analysis settings
    st.sidebar.subheader("Analysis Settings")
    num_keywords = st.sidebar.slider("Keywords to Extract", min_value=20, max_value=200, value=50)

    # Initialize clients
    try:
        reddit, oai, openai_api_key = init_clients()
    except:
        st.error("❌ Failed to initialize API clients. Please check your secrets configuration.")
        st.info("""
        Add these keys to your Streamlit secrets:
        ```
        REDDIT_CLIENT_ID = "your_reddit_client_id"
        REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
        OPENAI_API_KEY = "your_openai_api_key"
        REDDIT_USER_AGENT = "reddit-insights-streamlit/1.0"
        ```
        """)
        return

    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        try:
            # Fetch Reddit data
            with st.spinner(f"📡 Fetching data from r/{subreddit}..."):
                df = fetch_reddit_data(reddit, subreddit, num_threads, num_comments, listing_type)

            st.success(f"✅ Fetched {len(df[df['type']=='thread'])} threads and {len(df[df['type']=='comment'])} comments")

            # Sentiment classification
            st.subheader("💭 Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                progress_bar = st.progress(0)
                sentiments = []
                confs = []
                total = len(df)

                for idx, text in enumerate(df["text"].tolist()):
                    s, c = classify_sentiment(oai, text, model, reasoning_level)
                    sentiments.append(s)
                    confs.append(c)
                    progress_bar.progress((idx + 1) / total)

                df["sentiment"] = sentiments
                df["sentiment_confidence"] = confs
                progress_bar.empty()

            col1, col2 = st.columns([1, 1])
            with col1:
                fig = plot_sentiment_distribution(df, f"Sentiment Distribution for r/{subreddit}")
                st.pyplot(fig)

            with col2:
                st.markdown("**Sentiment Summary**")
                sent_counts = df["sentiment"].value_counts()
                for sent in ["Positive", "Neutral", "Negative", "Mixed"]:
                    count = sent_counts.get(sent, 0)
                    pct = (count / len(df) * 100) if len(df) > 0 else 0
                    st.metric(sent, f"{count} ({pct:.1f}%)")

            # Topic classification
            st.subheader("📊 Topic Analysis")
            with st.spinner("Classifying topics..."):
                progress_bar = st.progress(0)
                labels_list, top_topics, topic_conf_maps = [], [], []

                for idx, text in enumerate(df["text"].tolist()):
                    labels, top, confmap = classify_topics(oai, text, TOPICS, model, reasoning_level)
                    labels_list.append(labels)
                    top_topics.append(top)
                    topic_conf_maps.append(confmap)
                    progress_bar.progress((idx + 1) / total)

                df["topics"] = labels_list
                df["top_topic"] = top_topics
                df["topic_confidence"] = topic_conf_maps
                progress_bar.empty()

            fig = plot_topic_distribution(df, f"Topic Distribution for r/{subreddit}")
            st.pyplot(fig)

            # Keywords
            st.subheader("🔑 Keyword Analysis")
            with st.spinner("Extracting keywords..."):
                all_texts = df["text"].fillna("").tolist()
                kw_pairs = extract_rake_keywords(all_texts, top_n=num_keywords)

            if kw_pairs:
                fig = generate_wordcloud(kw_pairs, f"Keyword Cloud for r/{subreddit}")
                if fig:
                    st.pyplot(fig)

            # Entities
            st.subheader("🏢 Entity Analysis")
            with st.spinner("Extracting entities..."):
                nlp = load_spacy_model()
                ent_counts = extract_entities_spacy(nlp, all_texts)

            if ent_counts:
                fig = generate_wordcloud(dict(ent_counts), f"Entity Cloud for r/{subreddit}")
                if fig:
                    st.pyplot(fig)

            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(
                df[["type", "author", "score", "sentiment", "top_topic", "text"]].head(10),
                use_container_width=True
            )

            # Download data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"reddit_insights_{subreddit}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to start")
        st.markdown("""
        ### Features:
        - 💭 **Sentiment Analysis**: Classify posts as Positive, Neutral, Negative, or Mixed
        - 📊 **Topic Classification**: Categorize discussions into SEO topics
        - 🔑 **Keyword Extraction**: Identify key phrases using RAKE algorithm
        - 🏢 **Entity Recognition**: Extract people, organizations, and products
        - 📈 **Visualizations**: Word clouds and distribution charts

        ### Requirements:
        This tool requires API keys configured in Streamlit Secrets:
        - Reddit API (Client ID & Secret)
        - OpenAI API Key
        """)

if __name__ == "__main__":
    main()
