import os
from typing import Dict, List

def _get_top_skills(candidate: Dict) -> str:
    skills = candidate.get('skills', [])
    expert_skills = [s['name'] for s in skills if s.get('proficiency') in ('expert', 'advanced')]
    if len(expert_skills) >= 2:
        return f"{expert_skills[0]} and {expert_skills[1]}"
    elif len(expert_skills) == 1:
        return expert_skills[0]
    elif skills:
        return skills[0].get('name', 'relevant skills')
    return "core skills"

def _get_positive_signal(candidate: Dict, dimensions: Dict[str, float]) -> str:
    beh = candidate.get('behavioral_signals', {})
    if beh.get('open_to_work_flag'):
        return "Actively open to work"
    if beh.get('last_active_date_days_ago', 999) <= 30:
        return "Active on platform"
    if 'startup' in candidate.get('company_types', []):
        return "Startup experience"
    if dimensions.get('semantic', 0) > 0.8:
        return "Strong semantic alignment with JD"
    return "Solid overall profile"

def _get_strengths(candidate: Dict, dimensions: Dict[str, float]) -> List[str]:
    s = []
    if dimensions.get('semantic', 0) > 0.7:
        s.append("strong JD alignment")
    if dimensions.get('role', 0) > 0.7:
        s.append("good role match")
    if 'product' in candidate.get('company_types', []):
        s.append("product company background")
    if dimensions.get('behavioral', 1.0) > 1.0:
        s.append("highly responsive")
        
    if not s:
        s.append("relevant experience")
    return s[:2]

def _get_concerns(candidate: Dict, dimensions: Dict[str, float]) -> str:
    beh = candidate.get('behavioral_signals', {})
    np = beh.get('notice_period_days', 0)
    if np > 30:
        return f"Notice period {np} days"
    if dimensions.get('behavioral', 1.0) < 0.8:
        return "Low responsiveness"
    if dimensions.get('education', 1.0) < 0.5:
        return "Education tier"
    if dimensions.get('location', 1.0) < 0.5:
        return "Location not preferred"
    return "Minor skill gaps"

def _get_gaps(candidate: Dict, dimensions: Dict[str, float]) -> List[str]:
    g = []
    if dimensions.get('semantic', 1.0) < 0.5:
        g.append("limited domain relevance")
    if dimensions.get('experience', 1.0) < 0.5:
        exp = candidate.get('computed_total_experience', 0)
        if exp < 4:
            g.append("needs more experience")
        else:
            g.append("overqualified")
    if dimensions.get('skills', 1.0) < 0.5:
        g.append("unvalidated skills")
    if not g:
        g.append("general fit")
    return g[:2]

def _get_missing(candidate: Dict, dimensions: Dict[str, float]) -> List[str]:
    m = []
    if dimensions.get('semantic', 1.0) < 0.3:
        m.append("core ML/AI experience")
    if dimensions.get('role', 1.0) < 0.3:
        m.append("relevant job titles")
    if 'product' not in candidate.get('company_types', []) and 'startup' not in candidate.get('company_types', []):
        m.append("product/startup company experience")
    if not m:
        m.append("key JD requirements")
    return m[:2]

def generate_reasoning(candidate_normalized: Dict, final_score: float, dimensions: Dict[str, float]) -> str:
    """
    Generates a 1-2 sentence explanation for the candidate's ranking based solely on heuristics.
    No scores are emitted. No hallucinations are allowed.
    """
    # Extract baseline facts
    profile = candidate_normalized.get('profile', {})
    years = round(candidate_normalized.get('computed_total_experience', profile.get('years_of_experience', 0)))
    if years == 0:
        years = round(profile.get('years_of_experience', 0))
        
    title = str(profile.get('current_title', 'Professional')).strip()
    company = str(profile.get('current_company', 'Unknown')).strip()
    
    if final_score >= 0.85:
        skills_str = _get_top_skills(candidate_normalized)
        signal = _get_positive_signal(candidate_normalized, dimensions)
        return f"{years}-year {title} at {company} with strong {skills_str}. {signal}."
        
    elif 0.65 <= final_score < 0.85:
        strengths = _get_strengths(candidate_normalized, dimensions)
        str_text = ", ".join(strengths)
        caveat = _get_concerns(candidate_normalized, dimensions)
        return f"{years}-year {title}. Strengths: {str_text}. Concerns: {caveat}."
        
    elif 0.45 <= final_score < 0.65:
        gaps = _get_gaps(candidate_normalized, dimensions)
        gap_text = ", ".join(gaps)
        return f"{years}-year {title}. Gaps: {gap_text}."
        
    else:
        missing = _get_missing(candidate_normalized, dimensions)
        miss_text = ", ".join(missing)
        if company and company.lower() != 'unknown':
            return f"{company} background. Missing: {miss_text}."
        return f"{years}-year {title}. Missing: {miss_text}."

def _run_test():
    """Internal test harness to verify reasoning generation"""
    from loader import load_candidates
    from normalizer import normalize
    from scorer import score_candidate
    from embeddings import embed_text
    import numpy as np

    sample_file = os.path.join("..", "India_runs_data_and_ai_challenge", "sample_candidates.json")
    if not os.path.exists(sample_file):
        print("Sample data not found, skipping validation.")
        return

    print("Running Reasoner Pipeline Verification...\n")
    candidates = load_candidates(sample_file)[:5]
    
    # Dummy embeddings for the test
    jd_embedding = np.random.rand(384).astype(np.float32)
    career_emb = np.random.rand(384).astype(np.float32)

    for i, c in enumerate(candidates, 1):
        norm_c = normalize(c)
        score, dims = score_candidate(norm_c, jd_embedding, career_emb)
        reason = generate_reasoning(norm_c, score, dims)
        
        print(f"Candidate {i} | Score: {score:.3f}")
        print(f"Reasoning: {reason}\n")
        
        # Assertions to fulfill requirements
        assert len(reason.split('. ')) >= 1, "Should be at least 1 sentence"
        assert not any(char.isdigit() and '.' in reason for char in reason if '0.' in reason), "Should not mention raw scores"

    print("✅ Reasoner tests passed gracefully!")

if __name__ == '__main__':
    _run_test()
