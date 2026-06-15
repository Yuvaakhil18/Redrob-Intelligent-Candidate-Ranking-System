import json
import os
from typing import Dict, Tuple, List

def is_honeypot_or_suspicious(candidate_normalized: Dict) -> Tuple[bool, List[str]]:
    """
    Detects honeypot or suspicious candidate profiles based on 6 heuristic patterns.
    Returns (is_suspicious, [list of flags]).
    """
    flags = []
    is_critical = False
    
    # Extract common fields
    skills = candidate_normalized.get('skills', [])
    career_history = candidate_normalized.get('career_history', [])
    profile = candidate_normalized.get('profile', {})
    years_of_experience = profile.get('years_of_experience', 0.0)
    completeness = candidate_normalized.get('profile_completeness_score', 100.0)
    
    # Lists for patterns
    expert_zero_dur = []
    adv_exp_zero_end = []
    zero_end_experts = 0
    
    # Iterate through skills
    for skill in skills:
        name = str(skill.get('name', 'Unknown'))
        prof = str(skill.get('proficiency', '')).lower()
        duration = float(skill.get('duration_months', 0.0))
        endorsements = int(skill.get('endorsements', 0))
        
        # PATTERN 1: Expert skills with zero duration
        if prof == 'expert' and duration == 0.0:
            expert_zero_dur.append(name)
            
        # PATTERN 2: Expert/Advanced skills with zero endorsements
        if prof in ('advanced', 'expert') and endorsements == 0:
            adv_exp_zero_end.append(name)
            
        # PATTERN 4 helper: Count expert skills with 0 endorsements
        if prof == 'expert' and endorsements == 0:
            zero_end_experts += 1
            
    # PATTERN 1 Evaluation
    if len(expert_zero_dur) >= 3:
        flags.append(f"Expert skills with 0 duration: {expert_zero_dur}")
        
    # PATTERN 2 Evaluation
    if len(adv_exp_zero_end) >= 3:
        flags.append(f"Expert skills with 0 endorsements: {adv_exp_zero_end}")
        
    # PATTERN 4 Evaluation
    if zero_end_experts >= 8:
        flags.append("Excessive expert skills with no validation")
        
    # PATTERN 3: Years experience mismatch
    total_months_history = sum(float(job.get('duration_months', 0.0)) for job in career_history)
    actual_years = total_months_history / 12.0
    
    # 10% discrepancy or 1 year minimum threshold
    threshold = max(1.0, years_of_experience * 0.1)
    
    if abs(years_of_experience - actual_years) > threshold:
        flags.append(f"Years mismatch: stated={years_of_experience:.1f}, actual={actual_years:.1f}")
        is_critical = True  # CRITICAL FLAG
        
    # PATTERN 5: Zero career history but claims experience
    if not career_history and years_of_experience > 0.0:
        flags.append("Claims years but no job history")
        is_critical = True  # CRITICAL FLAG
        
    # PATTERN 6: Very low profile completeness
    if completeness < 40.0:
        flags.append("Very incomplete profile")
        
    # DECISION LOGIC
    is_suspicious = False
    if is_critical or len(flags) >= 3:
        is_suspicious = True
        
    return is_suspicious, flags

def _run_test():
    """Test harness for honeypot.py against sample data"""
    from loader import load_candidates
    from normalizer import normalize
    
    sample_file = os.path.join("..", "India_runs_data_and_ai_challenge", "sample_candidates.json")
    if not os.path.exists(sample_file):
        print("Sample data not found, skipping validation.")
        return
        
    print("Running Honeypot Detection on sample data...\n")
    candidates = load_candidates(sample_file)
    
    suspicious_count = 0
    
    for c in candidates:
        try:
            norm_c = normalize(c)
            is_suspicious, flags = is_honeypot_or_suspicious(norm_c)
            
            if is_suspicious:
                suspicious_count += 1
                print(f"Candidate: {norm_c['candidate_id']}")
                for f in flags:
                    print(f"  - {f}")
                print()
        except Exception as e:
            print(f"Error normalizing candidate: {e}")
            
    print(f"Total Suspicious Candidates Found: {suspicious_count} out of {len(candidates)}")
    print("✅ Detection test complete!")

if __name__ == '__main__':
    _run_test()
