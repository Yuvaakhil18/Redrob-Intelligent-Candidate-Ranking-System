from typing import Dict
from normalizer import normalize
from loader import load_sample

def score_dim3_experience(candidate_normalized: Dict) -> float:
    """
    Dimension 3: Experience Level Fit (15% weight)
    Optimal range is 6-8 years. Caps at 1.0.
    """
    years = float(candidate_normalized.get('profile', {}).get('years_of_experience', 0.0))
    
    if 6 <= years <= 8:
        base_score = 1.0
    elif 5 <= years < 6 or 8 < years <= 9:
        base_score = 0.85
    elif 4 <= years < 5 or 9 < years <= 12:
        base_score = 0.7
    elif years > 12:
        base_score = 0.4
    else: # < 4 years
        base_score = 0.3
        
    company_types = candidate_normalized.get('company_types', [])
    if 'startup' in company_types or 'scaleup' in company_types:
        base_score += 0.1
        
    return min(1.0, max(0.0, base_score))

def score_dim4_location(candidate_normalized: Dict) -> float:
    """
    Dimension 4: Location & Availability (15% weight)
    Scores preferred Indian tech hubs highly, applies penalty for relocation resistance
    and scales based on notice period.
    """
    profile = candidate_normalized.get('profile', {})
    signals = candidate_normalized.get('redrob_signals', {})
    
    # Safely extract and lower case the location string
    loc_str = str(profile.get('location', '')).lower()
    
    is_india = 'india' in loc_str or 'ind' in loc_str
    
    preferred_cities = ['pune', 'noida', 'delhi', 'hyderabad', 'bangalore', 'bengaluru', 'mumbai']
    in_preferred = any(city in loc_str for city in preferred_cities)
    
    willing_to_relocate = signals.get('willing_to_relocate', False)
    
    # 1. Location scoring
    if is_india and in_preferred:
        loc_score = 1.0
    elif is_india and willing_to_relocate:
        loc_score = 0.85
    elif is_india and not willing_to_relocate:
        loc_score = 0.6
    elif not is_india and willing_to_relocate:
        loc_score = 0.4
    else:
        loc_score = 0.1
        
    # 2. Notice period scoring
    notice_days = signals.get('notice_period_days', 60) # Default to typical
    if notice_days <= 30:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.85
    elif notice_days <= 90:
        notice_score = 0.6
    else:
        notice_score = 0.3
        
    final_score = loc_score * notice_score
    return min(1.0, max(0.0, final_score))

def score_dim5_education(candidate_normalized: Dict) -> float:
    """
    Dimension 5: Education (5% weight)
    """
    education = candidate_normalized.get('education', [])
    if not education:
        return 0.6
        
    # Helpers for inference
    cs_keywords = ['computer science', 'cs', 'it', 'information technology', 'software', 'artificial intelligence', 'data']
    bachelors_keywords = ['b.tech', 'btech', 'b.e', 'b.s', 'bs', 'bachelor']
    masters_keywords = ['m.tech', 'mtech', 'm.s', 'ms', 'master']
    phd_keywords = ['phd', 'ph.d', 'doctor']
    tier1_2_keywords = ['iit', 'indian institute of technology', 'nit', 'national institute of technology', 'bits', 'birla', 'iiit']
    
    best_score = 0.0
    
    for edu in education:
        deg = edu.get('degree', '').lower()
        major = edu.get('major', '').lower()
        uni = edu.get('university', '').lower()
        
        is_cs = any(kw in major or kw in deg for kw in cs_keywords)
        is_tier12 = any(kw in uni for kw in tier1_2_keywords)
        
        score = 0.5 # Default other
        
        if any(kw in deg for kw in phd_keywords):
            score = 0.9
        elif any(kw in deg for kw in masters_keywords):
            score = 1.0 if is_cs else 0.8
        elif any(kw in deg for kw in bachelors_keywords):
            if is_cs:
                if is_tier12:
                    score = 1.0
                else:
                    # Generic Tier 3 mapping for others
                    score = 0.85 
            else:
                score = 0.5
                
        best_score = max(best_score, score)
        
    return best_score

def main():
    sample_path = r"..\India_runs_data_and_ai_challenge\sample_candidates.json"
    candidates = load_sample(sample_path, n=3)
    
    for c in candidates:
        norm_c = normalize(c)
        
        dim3 = score_dim3_experience(norm_c)
        dim4 = score_dim4_location(norm_c)
        dim5 = score_dim5_education(norm_c)
        
        assert 0.0 <= dim3 <= 1.0
        assert 0.0 <= dim4 <= 1.0
        assert 0.0 <= dim5 <= 1.0
        
        print(f"Candidate: {c['candidate_id']}")
        print(f"  Years: {norm_c.get('computed_total_experience', 0)}")
        print(f"  Location: {norm_c.get('profile', {}).get('location', '')}")
        print(f"  Notice: {norm_c.get('computed_notice_period', 0)}")
        print(f"  Dim 3 (Experience): {dim3:.3f}")
        print(f"  Dim 4 (Location):   {dim4:.3f}")
        print(f"  Dim 5 (Education):  {dim5:.3f}\n")

    print("\u2705 scorer_dims_3_4_5.py works".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    main()
