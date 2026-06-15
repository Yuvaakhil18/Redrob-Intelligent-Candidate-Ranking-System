import streamlit as st
import pandas as pd
import json
import io
import time
import os
import sys

# Add local path to ensure imports work
sys.path.append(os.path.dirname(__file__))

from loader import load_candidates
from normalizer import normalize
from embeddings import embed_batch, embed_text
from scorer import score_candidate
from honeypot import is_honeypot_or_suspicious
from reasoner import generate_reasoning

# --- CONFIG ---
st.set_page_config(page_title="Redrob Ranker", page_icon="🎯", layout="wide")

DEFAULT_JD = """We are looking for a Senior Machine Learning Engineer with 6+ years of experience.
Expertise in Python, Vector Databases (Pinecone/Milvus), and building RAG pipelines.
Experience with Retrieval-Augmented Generation, embeddings, and semantic search systems.
Location: Pune or Remote (India)."""

# --- SIDEBAR ---
st.sidebar.title("🎯 Redrob Ranker")
st.sidebar.info("Offline-to-Online Candidate Ranking Demo")
st.sidebar.subheader("About")
st.sidebar.write("This interactive demo runs the complete pipeline:")
st.sidebar.write("1. Embeds the JD and candidates using `SentenceTransformers`")
st.sidebar.write("2. Extracts standardized dimensions & signals")
st.sidebar.write("3. Scores candidates logically based on JD cosine similarity + heuristics")
st.sidebar.write("4. Detects impossible 'honeypot' claims and penalizes them")
st.sidebar.write("5. Deterministically generates reasoned explanations")

st.sidebar.subheader("Performance")
st.sidebar.write("The production orchestrator handles 100K candidates in ~19 seconds.")

# --- UI ---
st.title("🎯 Redrob Intelligent Candidate Ranking System")
st.subheader("AI-powered ranking demo")

st.header("1. Job Description")
jd_text = st.text_area("Edit or paste custom JD:", value=DEFAULT_JD, height=150)

st.header("2. Candidates")
input_mode = st.radio("Choose input method:", ["Use sample (50 candidates)", "Upload file (JSON/JSONL/CSV)"])
uploaded_file = None
if "Upload file" in input_mode:
    uploaded_file = st.file_uploader("Upload candidates file", type=["json", "jsonl", "csv"])

if st.button("Rank Candidates", type="primary"):
    # Load candidates
    candidates = []
    if "Use sample" in input_mode:
        try:
            with open("sample_candidates.json", "r", encoding="utf-8") as f:
                candidates = json.load(f)
        except Exception as e:
            st.error(f"Failed to load sample: {e}")
            st.stop()
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".json") or uploaded_file.name.endswith(".jsonl"):
                # Try parsing json list or jsonl
                try:
                    candidates = json.load(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    candidates = [json.loads(line) for line in uploaded_file]
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                candidates = df.to_dict('records')
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")
            st.stop()
            
    if not candidates:
        st.error("No candidates loaded.")
        st.stop()
        
    total_candidates = len(candidates)
    
    with st.spinner(f"Step 1/2: Precomputing Embeddings ({total_candidates} candidates)..."):
        t0 = time.time()
        jd_embedding = embed_text(jd_text)
        
        normalized_candidates = []
        career_texts = []
        for c in candidates:
            try:
                norm_c = normalize(c)
                normalized_candidates.append(norm_c)
                career_texts.append(norm_c.get('career_text', ''))
            except Exception as e:
                pass
                
        embeddings = embed_batch(career_texts, batch_size=100)
        t1 = time.time()
        st.success(f"Precomputed {len(career_texts)} embeddings in {t1-t0:.2f}s")
        
    with st.spinner("Step 2/2: Ranking & Analyzing..."):
        t_start_rank = time.time()
        scored_results = []
        
        for i, norm_c in enumerate(normalized_candidates):
            final_score, dimensions = score_candidate(norm_c, jd_embedding, embeddings[i])
            scored_results.append({
                'candidate_id': norm_c['candidate_id'],
                'candidate_index': i,
                'score': final_score,
                'dimensions': dimensions
            })
            
        honeypot_count = 0
        for res in scored_results:
            idx = res['candidate_index']
            norm_c = normalized_candidates[idx]
            is_suspicious, flags = is_honeypot_or_suspicious(norm_c)
            if is_suspicious:
                res['score'] = 0.05
                honeypot_count += 1
                
        # Round before sort
        for res in scored_results:
            res['score'] = round(res['score'], 4)
            
        scored_results.sort(key=lambda x: (-x['score'], x['candidate_id']))
        top_100 = scored_results[:100]
        
        final_rankings = []
        for rank, res in enumerate(top_100, 1):
            idx = res['candidate_index']
            norm_c = normalized_candidates[idx]
            reasoning = generate_reasoning(norm_c, res['score'], res['dimensions'])
            
            final_rankings.append({
                'candidate_id': res['candidate_id'],
                'rank': rank,
                'score': f"{res['score']:.4f}",
                'reasoning': reasoning,
                'dimensions': res['dimensions']
            })
            
        t2 = time.time()
        st.success(f"Ranking completed in {t2-t_start_rank:.2f}s. Detected {honeypot_count} honeypots.")
        
    st.header("Results")
    
    # Create DF for display
    df_display = pd.DataFrame(final_rankings)[['rank', 'candidate_id', 'score', 'reasoning']]
    # Convert score to float for chart/display formatting
    df_display['score'] = df_display['score'].astype(float)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "candidate_id": st.column_config.TextColumn("Candidate ID"),
            "score": st.column_config.ProgressColumn(
                "Score",
                format="%.4f",
                min_value=0,
                max_value=1.2,
            ),
            "reasoning": st.column_config.TextColumn("Reasoning")
        }
    )
    
    # Download
    csv_buffer = io.StringIO()
    # Format to strictly 4 decimal places explicitly for CSV
    df_csv = df_display.copy()
    df_csv['score'] = df_csv['score'].apply(lambda x: f"{x:.4f}")
    df_csv.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv"
    )
    
    st.header("Top 10 Detail View")
    for row in final_rankings[:10]:
        with st.expander(f"Rank {row['rank']}: {row['candidate_id']} (Score: {row['score']})"):
            st.write(f"**Reasoning:** {row['reasoning']}")
            st.json(row['dimensions'])
