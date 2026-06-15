import numpy as np
from typing import Dict
from normalizer import normalize
from loader import load_sample

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def score_dim1_role(candidate_normalized: Dict, jd_embedding: np.ndarray, career_embedding: np.ndarray) -> float:
    """
    Evaluates Dimension 1: Role & Career Substance (30% weight in final system).
    """
    # Component A: Semantic similarity
    sem_score = cosine_similarity(jd_embedding, career_embedding)
    
    # Scale semantic score from typical [-1, 1] cosine range to [0, 1] range safely
    sem_score = max(0.0, min(1.0, sem_score))
    
    # Component B: Company type validation
    is_consulting = candidate_normalized.get('is_consulting_only', False)
    comp_score = 0.4 if is_consulting else 1.0
    
    # Component C: Keyword matching
    career_text = candidate_normalized.get('career_text', '').lower()
    keywords = ['retrieval', 'embedding', 'embeddings', 'ranking', 'vector', 'rag']
    found = sum(1 for kw in keywords if kw in career_text)
    
    if found >= 3:
        kw_score = 1.0
    elif found >= 1:
        kw_score = 0.6
    else:
        kw_score = 0.0
        
    # Combine
    final_score = (sem_score * 0.5) + (comp_score * 0.2) + (kw_score * 0.3)
    
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, final_score))

def _score_skill(skill_data: Dict) -> float:
    if not skill_data:
        return 0.0
        
    prof = skill_data.get('proficiency', '').lower()
    dur = skill_data.get('duration_months', 0)
    end = skill_data.get('endorsements', 0)
    
    if prof in ('advanced', 'expert'):
        if dur > 12 and end > 5:
            return 1.0
        elif 6 <= dur <= 12:
            return 0.8
        elif dur == 0 and end == 0:
            return 0.3 # KEYWORD STUFFER FLAG
        else:
            return 0.6
    elif prof == 'intermediate':
        return 0.5
    elif prof == 'beginner':
        return 0.2
    return 0.2

def score_dim2_skills(candidate_normalized: Dict) -> float:
    """
    Evaluates Dimension 2: Skills Validation (20% weight in final system).
    """
    skills_dict = {k.lower(): v for k, v in candidate_normalized.get('skills_dict', {}).items()}
    
    req_skills = ['python', 'embeddings', 'vector database', 'retrieval']
    opt_skills = ['llm fine-tuning', 'evaluation frameworks', 'ranking']
    
    # Match required skills (allows partial string matching for things like 'vector databases')
    req_scores = []
    for req in req_skills:
        matched = False
        for sk_name, sk_data in skills_dict.items():
            if req in sk_name or sk_name in req:
                req_scores.append(_score_skill(sk_data))
                matched = True
                break
        if not matched:
            req_scores.append(0.0)
            
    req_avg = sum(req_scores) / len(req_scores) if req_scores else 0.0
    
    if req_avg < 0.4:
        return 0.0 # Fails required skills threshold
        
    opt_scores = []
    for opt in opt_skills:
        matched = False
        for sk_name, sk_data in skills_dict.items():
            if opt in sk_name or sk_name in opt:
                opt_scores.append(_score_skill(sk_data))
                matched = True
                break
        if not matched:
            opt_scores.append(0.5)
            
    opt_avg = sum(opt_scores) / len(opt_scores) if opt_scores else 0.0
    
    final_score = (req_avg * 0.7) + (opt_avg * 0.3)
    
    return max(0.0, min(1.0, final_score))

def main():
    sample_path = r"..\India_runs_data_and_ai_challenge\sample_candidates.json"
    candidates = load_sample(sample_path, n=5) # Load 5 to have enough variety
    
    # Mock JD embedding (ones)
    jd_emb = np.ones(384) / np.sqrt(384)
    
    results = []
    for c in candidates:
        norm_c = normalize(c)
        
        # Mock career embedding (random)
        np.random.seed(hash(c['candidate_id']) % (2**32))
        c_emb = np.random.rand(384)
        c_emb = c_emb / np.linalg.norm(c_emb)
        
        dim1 = score_dim1_role(norm_c, jd_emb, c_emb)
        dim2 = score_dim2_skills(norm_c)
        
        assert 0.0 <= dim1 <= 1.0
        assert 0.0 <= dim2 <= 1.0
        
        results.append({
            'id': c['candidate_id'],
            'dim1': dim1,
            'dim2': dim2,
            'data': norm_c
        })
        
        print(f"Candidate: {c['candidate_id']}")
        print(f"  Dim 1 (Role):   {dim1:.3f}")
        print(f"  Dim 2 (Skills): {dim2:.3f}\n")
        
    print("\u2705 scorer_dims_1_2.py works".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    main()
