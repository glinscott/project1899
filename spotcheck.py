import streamlit as st
from datasets import load_from_disk
import json
import random
from pathlib import Path

import build_data as bd

# Re‑use the compiled regex & helper for highlighting modern terms.
MODERN_RE = bd._MODERN_RE  # compiled regex from build_data.py

# ---------------------------------------------------------------------------
#  Streamlit sidebar: dataset locations & controls
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pre‑1900 Corpus Spot‑Check", layout="wide")

st.sidebar.header("Dataset locations")
kept_path = st.sidebar.text_input(
    "Path to *kept* HuggingFace dataset (save_to_disk folder)", "./pre1900_corpus.arrow"
)
removed_path = st.sidebar.text_input(
    "Path to *removed* samples file (JSONL)", "./filtered_out_samples.jsonl"
)

sample_size = st.sidebar.slider("Sample size", 5, 200, 25, 5)
reload_btn = st.sidebar.button("Reload datasets")

# ---------------------------------------------------------------------------
#  Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading kept dataset …")
def load_kept_ds(path_str):
    path = Path(path_str)
    if not path.exists():
        st.warning(f"Kept dataset path not found: {path}")
        return None
    return load_from_disk(str(path))

@st.cache_data(show_spinner="Loading removed samples …")
def load_removed_samples(path_str, limit=20000):
    path = Path(path_str)
    if not path.exists():
        st.warning(f"Removed‑samples file not found: {path}")
        return []
    data = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

if reload_btn or "kept_ds" not in st.session_state:
    st.session_state.kept_ds = load_kept_ds(kept_path)
    st.session_state.removed = load_removed_samples(removed_path)

kept_ds = st.session_state.get("kept_ds")
removed_samples = st.session_state.get("removed", [])

# ---------------------------------------------------------------------------
#  Helper for HTML highlighting
# ---------------------------------------------------------------------------

def highlight_modern(text: str) -> str:
    """Wrap modern‑term matches in a red‑background span for easy visual scan."""
    def _repl(match):
        return f"<span style='background-color:#ffcccc'>{match.group()}</span>"

    return MODERN_RE.sub(_repl, text)

# ---------------------------------------------------------------------------
#  Two‑column layout: kept vs removed
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.header("Kept samples (expected to be clean)")

    if kept_ds is None:
        st.info("Load a valid kept dataset to view samples.")
    else:
        total_rows = len(kept_ds)
        st.markdown(f"**Total rows:** {total_rows}")
        indices = random.sample(range(total_rows), min(sample_size, total_rows))
        for idx in indices:
            ex = kept_ds[int(idx)]
            text = (ex.get("text", "") or "")[:800]  # truncate long passages
            meta = ex.get("publication_date", "n/a")
            st.markdown(
                f"**Index {idx}** | *{meta}*<br>" + highlight_modern(text),
                unsafe_allow_html=True,
            )
            st.divider()

with right:
    st.header("Removed samples (flagged as modern)")

    if not removed_samples:
        st.info("No removed‑sample file loaded or file is empty.")
    else:
        st.markdown(f"**Total removed logged:** {len(removed_samples)}")
        subset = random.sample(removed_samples, min(sample_size, len(removed_samples)))
        for ex in subset:
            snippet = ex.get("snippet", "")
            highlighted = highlight_modern(snippet)
            meta = ex.get("publication_date", "n/a")
            title = ex.get("short_book_title", "")
            st.markdown(
                f"*{meta} – {title}*<br>" + highlighted,
                unsafe_allow_html=True,
            )
            st.divider()

# ---------------------------------------------------------------------------
#  Footer
# ---------------------------------------------------------------------------

st.caption(
    "Project 1899 spot‑check dashboard — re‑using filtering logic from build_data.py."
)