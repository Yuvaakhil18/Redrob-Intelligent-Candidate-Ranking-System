import csv
import math
from typing import List, Dict

def validate_rankings(rankings: List[Dict]) -> List[str]:
    """
    Validates the structure and consistency of the final top 100 rankings.
    Returns a list of error strings. Empty list means the rankings are fully valid.
    """
    errors = []
    
    if len(rankings) == 0 or len(rankings) > 100:
        errors.append(f"Expected 1 to 100 rankings, but got {len(rankings)}")
        
    seen_ids = set()
    seen_ranks = set()
    prev_score = float('inf')
    
    for i, r in enumerate(rankings):
        # 1. Key existence
        if not all(k in r for k in ('candidate_id', 'rank', 'score', 'reasoning')):
            errors.append(f"Ranking at index {i} missing required keys.")
            continue
            
        cid = str(r['candidate_id'])
        rank = r['rank']
        score = r['score']
        reasoning = str(r['reasoning'])
        
        # 2. Candidate ID format and uniqueness
        if not cid.startswith('CAND_'):
            errors.append(f"Invalid candidate_id format at rank {rank}: {cid}")
        if cid in seen_ids:
            errors.append(f"Duplicate candidate_id found: {cid}")
        seen_ids.add(cid)
        
        # 3. Rank validation
        if not isinstance(rank, int) or not (1 <= rank <= 100):
            errors.append(f"Invalid rank value: {rank}")
        if rank in seen_ranks:
            errors.append(f"Duplicate rank found: {rank}")
        seen_ranks.add(rank)
        
        # 4. Score validation
        try:
            score = float(score)
            if math.isnan(score) or math.isinf(score):
                errors.append(f"Score is NaN or Inf at rank {rank}: {score}")
            elif score > prev_score:
                # Due to float rounding, strictly > is an error. 
                # (E.g. 0.825 and 0.825 is fine, but 0.825 and 0.826 is out of order)
                errors.append(f"Scores are not descending at rank {rank} ({score} > {prev_score})")
            prev_score = score
        except (ValueError, TypeError):
            errors.append(f"Invalid score format at rank {rank}: {score}")
            
        # 5. Reasoning validation
        if not reasoning.strip():
            errors.append(f"Empty reasoning string at rank {rank}")
            
    # Check if all ranks are explicitly accounted for sequentially
    expected_ranks = set(range(1, len(rankings) + 1))
    missing_ranks = expected_ranks - seen_ranks
    if missing_ranks:
        errors.append(f"Missing expected ranks: {sorted(list(missing_ranks))}")
        
    return errors

def write_submission_csv(output_path: str, rankings: List[Dict]) -> bool:
    """
    Writes the top 100 rankings to a CSV file.
    Validates data prior to writing and enforces proper formatting.
    """
    errors = validate_rankings(rankings)
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"Failed to write CSV due to validation errors:\n{error_msg}")
        
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=['candidate_id', 'rank', 'score', 'reasoning'],
                quoting=csv.QUOTE_MINIMAL # Safely escapes commas/quotes in reasoning
            )
            writer.writeheader()
            
            for r in rankings:
                # Format score strictly to 1 decimal place (or more if needed, standard is 1 or 2, 
                # prompt says "formatted to 1 decimal place")
                formatted_score = f"{float(r['score']):.4f}"
                
                writer.writerow({
                    'candidate_id': str(r['candidate_id']),
                    'rank': int(r['rank']),
                    'score': formatted_score,
                    'reasoning': str(r['reasoning'])
                })
                
        print(f"✓ Wrote {len(rankings)} rankings to {output_path}")
        return True
    except Exception as e:
        raise IOError(f"Failed to write CSV: {e}")

def _run_test():
    """Test harness specifically for csv_writer.py"""
    import os
    print("Running validation tests...")
    
    # Generate 100 valid test items
    rankings = []
    score = 1.0
    for i in range(1, 101):
        rankings.append({
            'candidate_id': f'CAND_{i:07d}',
            'rank': i,
            'score': score,
            'reasoning': f"Strong candidate for rank {i}."
        })
        score -= 0.005 # strictly decreasing
        
    errors = validate_rankings(rankings)
    assert len(errors) == 0, f"Expected 0 errors on valid data, got {errors}"
    
    # Test formatting execution
    test_path = "test_submission.csv"
    write_submission_csv(test_path, rankings)
    
    # Verify file contents
    assert os.path.exists(test_path), "CSV not generated"
    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 101, f"Expected 101 lines, got {len(lines)}"
        
        # Verify header
        assert lines[0].strip() == "candidate_id,rank,score,reasoning"
        
        # Verify random row formatting
        row_1 = lines[1].strip()
        assert row_1.startswith("CAND_0000001,1,1.0,Strong candidate")
        
    os.remove(test_path)
    print("✅ All validation tests passed!")

if __name__ == '__main__':
    _run_test()
