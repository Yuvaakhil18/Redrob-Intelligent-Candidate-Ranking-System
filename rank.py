import argparse
import os
import time
import numpy as np

from loader import load_candidates
from normalizer import normalize
from scorer import score_candidate
from honeypot import is_honeypot_or_suspicious
from reasoner import generate_reasoning
from csv_writer import write_submission_csv

def main(candidates_file: str, embeddings_dir: str, output_csv: str):
    """
    Main orchestration function that runs the complete online ranking pipeline.
    Expects pre-computed embeddings and a full candidates file.
    """
    t_start_total = time.time()
    print(f"Starting orchestration pipeline...")
    
    # STEP 1: Load pre-computed embeddings from disk
    t_step = time.time()
    emb_path = os.path.join(embeddings_dir, "embeddings.npy")
    ids_path = os.path.join(embeddings_dir, "candidate_ids.txt")
    jd_path = os.path.join(embeddings_dir, "jd_embedding.npy")
    
    if not os.path.exists(emb_path) or not os.path.exists(jd_path):
        raise FileNotFoundError(f"Missing embeddings in {embeddings_dir}. Run precompute.py first.")
        
    # Memory map to save RAM (crucial for 100K 384-d vectors)
    embeddings = np.load(emb_path, mmap_mode='r')
    jd_embedding = np.load(jd_path)
    
    with open(ids_path, 'r', encoding='utf-8') as f:
        candidate_ids = [line.strip() for line in f if line.strip()]
        
    print(f"✓ Loaded embeddings in {time.time() - t_step:.2f}s")
    
    # STEP 2: Load candidates from JSONL
    t_step = time.time()
    candidates = load_candidates(candidates_file)
    total_candidates = len(candidates)
    
    if total_candidates != len(candidate_ids) or total_candidates != embeddings.shape[0]:
        raise ValueError(f"Mismatch! Candidates: {total_candidates}, Embeddings: {embeddings.shape[0]}")
        
    print(f"✓ Loaded {total_candidates:,} candidates in {time.time() - t_step:.2f}s")
    
    # STEP 3: Normalize all candidates
    t_step = time.time()
    normalized_candidates = []
    
    for i, c in enumerate(candidates):
        normalized = normalize(c)
        normalized_candidates.append(normalized)
        
        if (i + 1) % 20000 == 0:
            print(f"  Normalized {i + 1}/{total_candidates}")
            
    norm_time = time.time() - t_step
    ms_per_cand = (norm_time / max(1, total_candidates)) * 1000
    print(f"✓ Normalized in {norm_time:.2f}s ({ms_per_cand:.2f}ms per candidate)")
    
    # STEP 4: Score all candidates
    t_step = time.time()
    scored_results = []
    
    for i, norm_c in enumerate(normalized_candidates):
        career_emb = embeddings[i]
        
        # Scorer function handles the weighted dimension logic
        final_score, dimensions = score_candidate(norm_c, jd_embedding, career_emb)
        
        scored_results.append({
            'candidate_id': norm_c['candidate_id'],
            'candidate_index': i,
            'score': final_score,
            'dimensions': dimensions
        })
        
        if (i + 1) % 20000 == 0:
            print(f"  Scored {i + 1}/{total_candidates}")
            
    score_time = time.time() - t_step
    ms_per_cand = (score_time / max(1, total_candidates)) * 1000
    print(f"✓ Scored all in {score_time:.2f}s ({ms_per_cand:.2f}ms per candidate)")
    
    # STEP 5: Detect honeypots
    t_step = time.time()
    honeypot_count = 0
    
    for res in scored_results:
        idx = res['candidate_index']
        norm_c = normalized_candidates[idx]
        
        is_suspicious, flags = is_honeypot_or_suspicious(norm_c)
        if is_suspicious:
            res['score'] = 0.05
            honeypot_count += 1
            
    print(f"✓ Honeypot detection in {time.time() - t_step:.2f}s")
    print(f"  Detected {honeypot_count} honeypots")
    
    # STEP 6: Sort and select top 100
    t_step = time.time()
    # Round scores to 4 decimal places BEFORE sorting so sort order matches CSV output
    for res in scored_results:
        res['score'] = round(res['score'], 4)
    scored_results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    top_100 = scored_results[:100]
    
    top_score = top_100[0]['score'] if top_100 else 0.0
    rank_100_score = top_100[-1]['score'] if len(top_100) == 100 else (top_100[-1]['score'] if top_100 else 0.0)
    
    print(f"✓ Selected top 100 in {time.time() - t_step:.2f}s")
    print(f"  Top score: {top_score:.3f}")
    if len(top_100) >= 100:
        print(f"  Rank 100 score: {rank_100_score:.3f}")
        
    # STEP 7: Generate reasoning strings
    t_step = time.time()
    final_rankings = []
    
    for rank, res in enumerate(top_100, 1):
        idx = res['candidate_index']
        norm_c = normalized_candidates[idx]
        
        reasoning = generate_reasoning(norm_c, res['score'], res['dimensions'])
        
        final_rankings.append({
            'candidate_id': res['candidate_id'],
            'rank': rank,
            'score': round(res['score'], 4),
            'reasoning': reasoning
        })
        
    print(f"✓ Generated reasoning in {time.time() - t_step:.2f}s")
    
    # STEP 8: Write CSV
    t_step = time.time()
    write_submission_csv(output_csv, final_rankings)
    print(f"✓ Wrote {output_csv} in {time.time() - t_step:.2f}s")
    
    # FINAL REPORT
    total_time = time.time() - t_start_total
    print("\n" + "="*40)
    print("FINAL REPORT")
    print("="*40)
    print(f"Total time     : {total_time:.2f} seconds")
    print(f"Total processed: {total_candidates:,} candidates")
    print(f"Total honeypots: {honeypot_count:,}")
    
    if total_time <= 300.0:
        print("\n\u2705 Runtime within 5-minute limit".encode("utf-8").decode("utf-8", "ignore"))
    else:
        print("\n\u26a0\ufe0f Warning: Runtime exceeded 5 minutes".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Online Candidate Ranking Orchestrator")
    parser.add_argument("--candidates", required=True, help="Path to full candidates.jsonl.gz file")
    parser.add_argument("--embeddings", default="embeddings_output", help="Directory with precomputed .npy files")
    parser.add_argument("--out", default="submission.csv", help="Path to write the output CSV")
    
    args = parser.parse_args()
    
    main(
        candidates_file=args.candidates,
        embeddings_dir=args.embeddings,
        output_csv=args.out
    )
