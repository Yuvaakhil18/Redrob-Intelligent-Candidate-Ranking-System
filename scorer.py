import numpy as np
from typing import Dict, Tuple
from normalizer import normalize
from loader import load_sample

from scorer_dims_1_2 import score_dim1_role, score_dim2_skills, cosine_similarity
from scorer_dims_3_4_5 import score_dim3_experience, score_dim4_location, score_dim5_education
from scorer_dim_6_behavioral import score_dim6_behavioral

def score_candidate(
    candidate_normalized: Dict, 
    jd_embedding: np.ndarray, 
    career_embedding: np.ndarray, 
    weights: Dict = None
) -> Tuple[float, Dict[str, float]]:
    """
    Orchestrates all 6 dimensions to calculate the final ranking score.
    Returns the final score and a dictionary breaking down the contribution of each dimension.
    """
    if weights is None:
        weights = {
            'dim1': 0.30,
            'dim2': 0.20,
            'dim3': 0.15,
            'dim4': 0.15,
            'dim5': 0.05
        }
        
    # Safely handle missing/None embeddings
    if jd_embedding is None or career_embedding is None:
        jd_embedding = np.zeros(384)
        career_embedding = np.zeros(384)
        
    # Calculate each dimension
    dim1 = score_dim1_role(candidate_normalized, jd_embedding, career_embedding)
    dim2 = score_dim2_skills(candidate_normalized)
    dim3 = score_dim3_experience(candidate_normalized)
    dim4 = score_dim4_location(candidate_normalized)
    dim5 = score_dim5_education(candidate_normalized)
    mult6 = score_dim6_behavioral(candidate_normalized)
    
    # Calculate base score (max 0.85 mathematically since sum of weights = 0.85, 
    # wait: 0.3 + 0.2 + 0.15 + 0.15 + 0.05 = 0.85. 
    # Let me normalize the weights if they don't sum to 1.0, 
    # but the spec says "base = (dim1*0.30 + dim2*0.20 + dim3*0.15 + dim4*0.15 + dim5*0.05)"
    # Actually 0.3 + 0.2 + 0.15 + 0.15 + 0.05 = 0.85. 
    # Wait, 15% + 15% + 5% + 30% + 20% = 85%. 
    # If the spec literally says that, I'll follow it exactly.
    
    base_score = (
        dim1 * weights['dim1'] +
        dim2 * weights['dim2'] +
        dim3 * weights['dim3'] +
        dim4 * weights['dim4'] +
        dim5 * weights['dim5']
    )
    
    # In case weights were custom and sum > 1.0
    base_score = max(0.0, min(1.0, base_score))
    
    # Apply behavioral multiplier
    final_score = base_score * mult6
    
    # Clamp to [0.0, 1.2]
    final_score = max(0.0, min(1.2, final_score))
    
    # Bonus info
    sem_sim = cosine_similarity(jd_embedding, career_embedding)
    
    dimensions_dict = {
        'dim1_role': dim1,
        'dim2_skills': dim2,
        'dim3_experience': dim3,
        'dim4_location': dim4,
        'dim5_education': dim5,
        'dim6_behavioral_multiplier': mult6,
        'base_score': base_score,
        'final_score': final_score,
        'semantic_similarity': sem_sim
    }
    
    return float(final_score), dimensions_dict

def main():
    sample_path = r"..\India_runs_data_and_ai_challenge\sample_candidates.json"
    candidates = load_sample(sample_path, n=3)
    
    jd_emb = np.ones(384) / np.sqrt(384)
    
    for c in candidates:
        norm_c = normalize(c)
        
        np.random.seed(hash(c['candidate_id']) % (2**32))
        c_emb = np.random.rand(384)
        c_emb = c_emb / np.linalg.norm(c_emb)
        
        final_score, dims = score_candidate(norm_c, jd_emb, c_emb)
        
        assert 0.0 <= final_score <= 1.2
        assert 'dim1_role' in dims
        assert 'dim6_behavioral_multiplier' in dims
        assert 'final_score' in dims
        
        print(f"Candidate: {c['candidate_id']}")
        print(f"  Base Score: {dims['base_score']:.3f}")
        print(f"  Multiplier: {dims['dim6_behavioral_multiplier']:.3f}")
        print(f"  Final Score: {dims['final_score']:.3f}")
        print("  Breakdown:")
        for k, v in dims.items():
            print(f"    {k}: {v:.3f}")
        print()

    print("\u2705 scorer.py works".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    main()
