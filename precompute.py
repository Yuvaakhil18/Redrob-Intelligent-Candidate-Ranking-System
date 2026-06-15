import argparse
import os
import json
import time
import numpy as np
from typing import List

from loader import load_candidates
from normalizer import normalize
from embeddings import embed_batch, embed_text

DEFAULT_JD = """
We are looking for a Senior Machine Learning Engineer with 6+ years of experience.
Expertise in Python, Vector Databases (Pinecone/Milvus), and building RAG pipelines.
Experience with Retrieval-Augmented Generation, embeddings, and semantic search systems.
Location: Pune or Remote (India).
"""

def precompute(
    candidates_file: str, 
    output_dir: str, 
    jd_text: str, 
    batch_size: int = 100
):
    """
    Offline precomputation of embeddings for all candidates and the job description.
    Outputs the numpy arrays, candidate IDs, and metadata to disk.
    """
    print(f"Starting precomputation...")
    print(f"Candidates file: {candidates_file}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load all candidates
    t0 = time.time()
    candidates = load_candidates(candidates_file)
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.2f}s")
    
    # 2. Extract career_text from each
    t1 = time.time()
    candidate_ids = []
    career_texts = []
    
    for idx, c in enumerate(candidates):
        try:
            norm_c = normalize(c)
            candidate_ids.append(norm_c['candidate_id'])
            career_texts.append(norm_c.get('career_text', ''))
        except Exception as e:
            # Graceful error handling for corrupt candidate shapes
            print(f"Warning: Failed to normalize candidate at index {idx}. Error: {e}")
            
    print(f"Normalized {len(career_texts)} candidates in {time.time()-t1:.2f}s")
    
    # 3. Embed all career texts
    print(f"Embedding {len(career_texts)} career texts...")
    t2 = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Delegate entirely to SentenceTransformer which handles batching optimally
    final_embeddings = embed_batch(career_texts, batch_size=batch_size)
    
    print(f"Embedding completed in {time.time()-t2:.2f}s")
    
    # 4. Embed Job Description
    print("Embedding Job Description...")
    jd_embedding = embed_text(jd_text)
    
    # 5. Save everything to disk
    t3 = time.time()
    
    emb_path = os.path.join(output_dir, "embeddings.npy")
    ids_path = os.path.join(output_dir, "candidate_ids.txt")
    jd_path = os.path.join(output_dir, "jd_embedding.npy")
    meta_path = os.path.join(output_dir, "precompute_metadata.json")
    
    np.save(emb_path, final_embeddings)
    np.save(jd_path, jd_embedding)
    
    with open(ids_path, 'w', encoding='utf-8') as f:
        for cid in candidate_ids:
            f.write(f"{cid}\n")
            
    metadata = {
        "num_candidates": len(candidate_ids),
        "embedding_shape": final_embeddings.shape,
        "batch_size": batch_size,
        "jd_length": len(jd_text),
        "timestamp": time.time(),
        "model": "all-MiniLM-L6-v2"
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved artifacts to {output_dir} in {time.time()-t3:.2f}s")
    print("\nPrecomputation successful!")

def main():
    parser = argparse.ArgumentParser(description="Offline Precomputation Engine")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--output-dir", default="embeddings_output", help="Directory to save outputs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding generation")
    parser.add_argument("--jd-text", type=str, default=DEFAULT_JD, help="Job description text (optional override)")
    
    args = parser.parse_args()
    
    precompute(
        candidates_file=args.candidates,
        output_dir=args.output_dir,
        jd_text=args.jd_text,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()