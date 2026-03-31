import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import modal
import time
import joblib
import tantivy
from sentence_transformers import SentenceTransformer

app = modal.App("ltr-search-engine-ui")
MOUNT = "/data"
volume = modal.Volume.from_name("ltr-ms-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas",
        "pyarrow",
        "tantivy",
        "lightgbm",
        "sentence-transformers",
        "torch",
        "streamlit",
        "symspellpy",
    )
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)


@app.cls(
    image=image,
    volumes={MOUNT: volume},
    timeout=60 * 10,
    gpu="T4",
    memory=65536,
    scaledown_window=600,
)
class ProductionSearchEngine:

    @modal.enter()
    def initialize(self):
        """
        This runs once when the container boots up. It loads models into memory.
        """
        import joblib
        import torch
        from sentence_transformers import SentenceTransformer
        import tantivy
        import time
        from pathlib import Path

        print("🔧 Spinning up Search Engine...")
        start = time.time()

        # 1. Load Neural Model (GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.semantic_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device=self.device
        )

        # 2. Load LightGBM Ranker Model
        model_path = Path("/data/models/ms_marco_v2/lgbm_ranker_jamal.joblib")
        if not model_path.exists():
            raise FileNotFoundError("Model not found. You must run training first!")
        self.ranker_model = joblib.load(model_path)

        # 3. Connect to the Tantivy Database (INSTANT 0.01s BOOT)
        print("Connecting to Tantivy Database (8.8 Million Documents)...")
        self.index = tantivy.Index.open("/data/tantivy_index")
        self.searcher = self.index.searcher()

        # 4. Initialize SymSpell Dictionary
        from symspellpy import SymSpell, Verbosity
        import pkg_resources

        print("Loading SymSpell Dictionary...")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Load Standard English
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        # 🟢 Load Custom MS MARCO Vocabulary
        try:
            self.sym_spell.load_dictionary(
                "/data/ms_marco_dictionary.txt", term_index=0, count_index=1
            )
            print("Loaded Custom MS MARCO SymSpell Dictionary!")
        except Exception as e:
            print(f"Skipping custom dictionary: {e}")

        print(f"Search Engine is WARM and READY! Boot time: {time.time()-start:.1f}s")

    @modal.method()
    # 🟢 UPDATE: Added `offset` and `limit` parameters to control which slice of results we return.
    def search(self, query: str, offset: int = 0, limit: int = 10):
        """
        The actual Search API Endpoint.
        """
        import numpy as np
        import pandas as pd
        import torch
        import time
        import tantivy

        start = time.time()
        # ---------------------------------------------
        # PRE-SEARCH: SPELL CHECK & AUTO-CORRECT
        # ---------------------------------------------
        from symspellpy import Verbosity

        corrected_terms = []
        for word in query.split():
            # Don't spell check tiny words or numbers
            if len(word) < 4 or word.isnumeric() or not word.isalpha():
                corrected_terms.append(word)
            else:
                # 🟢 PRO FIX FOR NAMES: Check if the word exactly exists in the Tantivy dataset first.
                # If a name like 'Areeb' or 'Nile' gets hits, it's valid, so we bypass SymSpell!
                word_query = self.index.parse_query(word, ["document"])
                if len(self.searcher.search(word_query, 1).hits) > 0:
                    corrected_terms.append(word)
                    continue

                # If the word doesn't exist in our data at all, let SymSpell correct it
                suggestions = self.sym_spell.lookup(
                    word, Verbosity.CLOSEST, max_edit_distance=2
                )
                if suggestions:
                    corrected_terms.append(suggestions[0].term)
                else:
                    corrected_terms.append(word)

        corrected_query = " ".join(corrected_terms)

        # ---------------------------------------------
        # STAGE 1: FAST RETRIEVAL (TANTIVY)
        # ---------------------------------------------
        query_parser = self.index.parse_query(query, ["document"])

        # 🟢 UPDATE: Increased the initial Tantivy retrieval from 100
        # we have enough candidates to rank and paginate through multiple pages.
        result = self.searcher.search(query_parser, 100)

        # If exact search returned too few results, retry with upgraded fuzzy logic (Option 2)
        if len(result.hits) < 20:
            fuzzy_query = " ".join(
                (
                    f"{word}~2"
                    if len(word) >= 4
                    else f"{word}~1" if len(word) > 3 else word
                )
                for word in query.split()
            )
            query_parser = self.index.parse_query(fuzzy_query, ["document"])
            result = self.searcher.search(query_parser, 100)

        # Extract the documents
        doc_ids = []
        documents = []
        for score, doc_address in result.hits:
            doc = self.searcher.doc(doc_address)
            doc_ids.append(doc["doc_id"][0])
            documents.append(doc["document"][0])

        if len(doc_ids) == 0:
            return {
                "time": time.time() - start,
                "results": [],
                "corrected_query": corrected_query,
            }

        candidates = pd.DataFrame({"doc_id": doc_ids, "document": documents})

        # ---------------------------------------------
        # STAGE 2: EXTRACT FEATURES FOR CANDIDATES
        # ---------------------------------------------
        q_tokens = query.lower().split()

        candidates["query_len"] = len(q_tokens)
        candidates["doc_len"] = candidates["document"].apply(
            lambda x: len(str(x).split())
        )

        q_set = set(q_tokens)

        def calc_overlap(doc_text):
            d_set = set(str(doc_text).lower().split())
            if not q_set:
                return 0
            return len(q_set.intersection(d_set)) / len(q_set)

        candidates["overlap_ratio"] = candidates["document"].apply(calc_overlap)

        def calc_exact(doc_text):
            d_set = set(str(doc_text).lower().split())
            return sum(1 for q in q_tokens if q in d_set)

        candidates["exact_match_count"] = candidates["document"].apply(calc_exact)

        def calc_proxies(doc_text):
            import numpy as np

            d_words = str(doc_text).lower().split()
            d_len = len(d_words)
            bm25 = 0
            tfidf = 0
            freqs = {}
            for w in d_words:
                freqs[w] = freqs.get(w, 0) + 1
            for q in q_tokens:
                tf = freqs.get(q, 0)
                bm25 += tf / (tf + 1.2)
                tfidf += tf / np.log(d_len + 2)
            return bm25, tfidf

        proxy_results = candidates["document"].apply(calc_proxies)
        candidates["bm25_proxy"] = proxy_results.apply(lambda x: x[0])
        candidates["tf_idf_proxy"] = proxy_results.apply(lambda x: x[1])

        # Neural Semantic Similarity (GPU)
        q_emb = self.semantic_model.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        ).unsqueeze(0)
        d_emb = self.semantic_model.encode(
            candidates["document"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if d_emb.dim() == 1:
            d_emb = d_emb.unsqueeze(0)

        cos_sim = (
            torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=1).cpu().numpy()
        )
        candidates["semantic_sim"] = cos_sim

        # ---------------------------------------------
        # STAGE 3: RE-RANKING WITH LIGHTGBM
        # ---------------------------------------------
        features = [
            "query_len",
            "doc_len",
            "overlap_ratio",
            "bm25_proxy",
            "exact_match_count",
            "tf_idf_proxy",
            "semantic_sim",
        ]

        ranking_scores = self.ranker_model.predict(candidates[features])
        candidates["ml_rank_score"] = ranking_scores

        # 🟢 UPDATE: Replaced `.head(top_k)` with `.iloc[offset : offset + limit]`.
        # This slices the sorted dataframe to only grab the specific 10 documents for the current page.
        final_sorted = candidates.sort_values(by="ml_rank_score", ascending=False).iloc[
            offset : offset + limit
        ]

        elapsed = time.time() - start

        results = final_sorted[
            [
                "doc_id",
                "ml_rank_score",
                "document",
                "semantic_sim",
                "tf_idf_proxy",
                "exact_match_count",
                "overlap_ratio",
                "bm25_proxy",
            ]
        ].to_dict(orient="records")
        return {"time": elapsed, "results": results, "corrected_query": corrected_query}


# --- STREAMLIT FRONTEND ---
import pandas as pd
import math

st.set_page_config(page_title="Semantrix Search", layout="wide")

# Custom CSS for Modern Dark/Neon UI (Dribbble Inspiration)
st.markdown(
    """
<style>
/* Reset and Container Styling */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 950px;
}
.stApp {
    background-color: transparent;
}

/* Header & Slogan */
.header-container {
    text-align: center;
    padding-bottom: 40px;
    animation: fadeInDown 0.8s ease-out;
}
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 5rem;
    font-weight: 900;
    color: #FFFFFF;
    text-shadow: 0 0 15px rgba(225, 0, 255, 0.5), 0 0 30px rgba(0, 240, 255, 0.4);
    margin-bottom: -15px;
    letter-spacing: 1px;
}
.slogan {
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    color: #00F0FF;
    font-weight: 300;
    letter-spacing: 5px;
    text-transform: uppercase;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
}

/* Search Results UI Cards */
.semantrix-card {
    background-color: rgba(26, 21, 37, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    animation: fadeIn 0.5s ease-in;
}
.semantrix-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(225, 0, 255, 0.2);
    border-color: rgba(0, 240, 255, 0.4);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.rank-badge {
    background: rgba(0,0,0,0.4);
    color: #E0E0E0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: bold;
    border: 1px solid rgba(255,255,255,0.1);
}
.rank-badge.gold { background: rgba(255, 215, 0, 0.1); color: #FFD700; border-color: #FFD700; box-shadow: 0 0 8px rgba(255, 215, 0, 0.2);}
.rank-badge.silver { background: rgba(192, 192, 192, 0.1); color: #C0C0C0; border-color: #C0C0C0; box-shadow: 0 0 8px rgba(192, 192, 192, 0.2);}
.rank-badge.bronze { background: rgba(205, 127, 50, 0.1); color: #CD7F32; border-color: #CD7F32; box-shadow: 0 0 8px rgba(205, 127, 50, 0.2);}

.doc-link {
    color: #00F0FF;
    font-size: 0.95rem;
    text-decoration: none;
    font-family: 'Courier New', monospace;
    text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
}

.doc-snippet {
    color: #B3B3B3;
    font-size: 1.05rem;
    line-height: 1.6;
    margin-bottom: 15px;
}

/* Footer styling */
.footer {
    text-align: center;
    color: #666666;
    font-size: 0.9rem;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# --- 1. SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("## Semantrix Settings")
    top_k = st.slider(
        "Max Results to Fetch", min_value=5, max_value=50, value=10, step=5
    )
    show_features = st.toggle("Show Deep ML Features", value=True)

    st.markdown("---")
    st.markdown("### About Semantrix")
    st.markdown(
        '<div style="background-color: rgba(26, 21, 37, 0.8); color: #E0E0E0; padding: 15px; border-radius: 10px; font-size: 0.95rem; line-height: 1.5; margin-bottom:15px; border: 1px solid rgba(225, 0, 255, 0.2);">A state-of-the-art Search Engine powered by <strong style="color:#00F0FF;">Tantivy</strong>, <strong style="color:#00F0FF;">Sentence Transformers</strong>, and <strong style="color:#00F0FF;">LightGBM</strong>. It runs purely on Modal serverless GPUs, instantly surfacing the most relevant content.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<small>v2.0-production</small>", unsafe_allow_html=True)


# --- 2. HEADER ---
st.markdown(
    """
<div class="header-container">
    <div class="main-title">Semantrix</div>
    <div class="slogan">Learn, Rank, Resonate</div>
</div>
""",
    unsafe_allow_html=True,
)


# Ensure session state exists
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Example Queries (Chips)
st.markdown("##### Try an example:")
col1, col2, col3 = st.columns(3)
if col1.button("What is Nile University?", use_container_width=True):
    st.session_state.search_query = "What is Nile University?"
if col2.button("How to reduce blood pressure?", use_container_width=True):
    st.session_state.search_query = "How to reduce blood pressure?"
if col3.button("Machine learning basics", use_container_width=True):
    st.session_state.search_query = "Machine learning basics"

# Main Search Input
user_query = st.text_input(
    "Search:",
    value=st.session_state.search_query,
    placeholder="Search millions of documents...",
    label_visibility="collapsed",
)


# Cache the engine connection to avoid re-initializing
@st.cache_resource(show_spinner=False)
def get_engine():
    EngineClass = modal.Cls.from_name("ltr-search-engine-ui", "ProductionSearchEngine")
    return EngineClass()


if st.button("Resonate", type="primary", use_container_width=True) or (
    user_query and user_query != ""
):
    if user_query:
        st.session_state.search_query = user_query

        with st.spinner("Connecting to Semantrix Core Matrix..."):
            try:
                engine = get_engine()
                # Execute semantic search via Modal
                response = engine.search.remote(user_query, offset=0, limit=top_k)

                results = response.get("results", [])
                elapsed = response.get("time", 0.0)
                corrected = response.get("corrected_query", user_query)

                # Spell Correction Notification
                if corrected.lower() != user_query.lower():
                    st.markdown(
                        f'<div style="background-color: rgba(26, 21, 37, 0.8); color: white; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid rgba(0, 240, 255, 0.3);">Did you mean: <strong style="color: #00F0FF;">{corrected}</strong>?</div>',
                        unsafe_allow_html=True,
                    )

                st.success(
                    f"Generated **{len(results)}** resonances in **{elapsed:.3f}** seconds."
                )

                # --- RESULTS DISPLAY ---
                if results:
                    # Download Data Prep
                    df_export = pd.DataFrame(results)
                    csv_data = df_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results Data (CSV)",
                        data=csv_data,
                        file_name="semantrix_export.csv",
                        mime="text/csv",
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    for i, res in enumerate(results):
                        rank = i + 1
                        doc_id = res["doc_id"]
                        doc_text = res["document"]
                        score = res["ml_rank_score"]

                        # Dynamic Icon/Color for Top 3 Ranks
                        if rank == 1:
                            badge_class = "gold"
                            icon = "1st"
                        elif rank == 2:
                            badge_class = "silver"
                            icon = "2nd"
                        elif rank == 3:
                            badge_class = "bronze"
                            icon = "3rd"
                        else:
                            badge_class = ""
                            icon = f"#{rank}"

                        # Format progress bar standard range (Sigmoid normalize the raw ranking score)
                        norm_score = 1.0 / (
                            1.0 + math.exp(-max(min(score, 10.0), -10.0))
                        )

                        # Truncate text logic allowing them to view more
                        words = doc_text.split()
                        truncated = (
                            " ".join(words[:60]) + "..."
                            if len(words) > 60
                            else doc_text
                        )

                        st.markdown(
                            f"""
                        <div class="semantrix-card">
                            <div class="result-header">
                                <span class="rank-badge {badge_class}">{icon}</span>
                                <span class="doc-link">DOC_{doc_id}</span>
                            </div>
                            <div class="doc-snippet">{truncated}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Native Streamlit components for interactiveness within the card flow
                        st.progress(
                            norm_score, text=f"Model Confidence Score: {score:.3f}"
                        )

                        if show_features:
                            with st.expander(f"ML Features Breakdown for Rank {rank}"):
                                feat_data = {
                                    "Semantic Sim. (Neural)": [
                                        round(res.get("semantic_sim", 0), 3)
                                    ],
                                    "Exact Match Count": [
                                        res.get("exact_match_count", 0)
                                    ],
                                    "TF-IDF Proxy": [
                                        round(res.get("tf_idf_proxy", 0), 3)
                                    ],
                                    "BM25 Proxy": [round(res.get("bm25_proxy", 0), 3)],
                                    "Overlap Ratio": [
                                        round(res.get("overlap_ratio", 0), 3)
                                    ],
                                }
                                st.dataframe(
                                    pd.DataFrame(feat_data),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                                # A mini feature importance bar using native streamlit bar_chart
                                plot_df = pd.DataFrame(
                                    [
                                        res.get("semantic_sim", 0),
                                        res.get("tf_idf_proxy", 0),
                                        res.get("bm25_proxy", 0),
                                    ],
                                    index=[
                                        "Semantic Sim",
                                        "TF-IDF Proxy",
                                        "BM25 Proxy",
                                    ],
                                    columns=["Score"],
                                )
                                st.bar_chart(plot_df, height=150)

                        st.markdown(
                            "<div style='margin-bottom: 25px;'></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No documents found for your search. Try different terms!")

            except Exception as e:
                st.error(
                    "Semantrix Core Error: Unable to communicate with Modal backend."
                )
                st.exception(e)
                if st.button("Retry Connection"):
                    st.rerun()

st.markdown(
    '<div class="footer">Semantrix Search Engine v2.0 | Built with Modal & Streamlit</div>',
    unsafe_allow_html=True,
)
