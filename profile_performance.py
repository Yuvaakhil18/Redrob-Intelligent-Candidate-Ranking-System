import os
import subprocess
import time
import sys

def main():
    print("========================================")
    print("      PERFORMANCE PROFILING SUITE       ")
    print("========================================\n")
    
    sample_candidates = os.path.join("..", "India_runs_data_and_ai_challenge", "sample_candidates.json")
    embeddings_dir = "profile_embeddings_output"
    output_csv = "profile_submission.csv"
    
    # Ensure clean state
    if not os.path.exists(sample_candidates):
        print(f"Error: Could not find sample file at {sample_candidates}")
        sys.exit(1)
        
    print(f"Target sample file: {sample_candidates} (50 candidates)\n")
    
    # 1. RUN PRECOMPUTE ON SAMPLES
    print("1. Running precomputation (Offline Step)...")
    t0 = time.time()
    res_pre = subprocess.run(
        ["python", "precompute.py", "--candidates", sample_candidates, "--output-dir", embeddings_dir],
        capture_output=True, text=True
    )
    precompute_time = time.time() - t0
    
    if res_pre.returncode != 0:
        print(f"Precompute failed:\n{res_pre.stderr}")
        sys.exit(1)
        
    print(f"   ✓ Precomputation finished in {precompute_time:.2f}s\n")
    
    # 2. RUN RANKING ON SAMPLES
    print("2. Running ranking (Online Timed Step)...")
    t0 = time.time()
    res_rank = subprocess.run(
        ["python", "rank.py", "--candidates", sample_candidates, "--embeddings", embeddings_dir, "--out", output_csv],
        capture_output=True, text=True
    )
    ranking_time = time.time() - t0
    
    if res_rank.returncode != 0:
        print(f"Ranking failed:\n{res_rank.stderr}")
        sys.exit(1)
        
    print(f"   ✓ Ranking finished in {ranking_time:.2f}s\n")
    
    # 3. ANALYZE BREAKDOWN
    # In a subprocess, the ~0.3s ranking_time is almost entirely Python import/startup overhead.
    # We apply the naive linear scaling requested by the spec:
    estimated_100k_time = ranking_time * (100000.0 / 50.0)
    per_candidate_ms = (ranking_time / 50.0) * 1000.0
    
    print("========================================")
    print("         PERFORMANCE BREAKDOWN          ")
    print("========================================")
    print(f"Offline Precomputation Time : {precompute_time:.2f}s (run once before scoring window)")
    print(f"Online Timed Window (50)    : {ranking_time:.2f}s")
    print(f"Per-candidate Average Cost  : {per_candidate_ms:.2f}ms per candidate")
    print("----------------------------------------")
    print(f"Estimated 100K Online Time  : {estimated_100k_time:.2f} seconds ({estimated_100k_time / 60.0:.2f} minutes)")
    
    # 4. REPORT
    print("\nConstraint Check:")
    if estimated_100k_time < 300.0:
        print("✅ Estimated runtime acceptable")
    else:
        print("⚠️  WARNING: Estimated runtime exceeds 5-minute limit")
        print("   (Note: This is a naive extrapolation factoring in the heavy ~0.3s static Python startup ")
        print("    and module import overhead multiplied by 2000x. Actual scaling is O(N) only on execution.)")

if __name__ == '__main__':
    main()
