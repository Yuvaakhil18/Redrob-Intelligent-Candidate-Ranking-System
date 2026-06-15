import numpy as np
from typing import List, Optional, Any

# Global variable to cache the model across calls
_MODEL = None

def load_model() -> Any:
    """
    Lazy-loads the sentence-transformers model 'all-MiniLM-L6-v2' globally.
    Downloads it once on the first call, caches it in memory, and returns it.
    """
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using cpu device specifically as requested
            _MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except ImportError:
            raise ImportError("Please install sentence-transformers using: pip install sentence-transformers")
    return _MODEL

def embed_text(text: Optional[str]) -> np.ndarray:
    """
    Embeds a single text string into a 384-dimensional vector.
    Returns an array of zeros if the input is None or empty.
    """
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)
        
    model = load_model()
    # SentenceTransformer outputs a numpy array or torch tensor, we convert to numpy just in case
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.astype(np.float32)

def embed_batch(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """
    Embeds multiple texts efficiently in batches.
    Empty strings are safely handled by the model naturally or manually.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
        
    # Replace None with empty string for the transformer
    clean_texts = ["" if t is None else t for t in texts]
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model = load_model()
    embeddings = model.encode(
        clean_texts, 
        batch_size=batch_size, 
        convert_to_numpy=True, 
        show_progress_bar=True
    )
    return embeddings.astype(np.float32)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 384-dimensional vectors.
    Returns 0.0 if either vector is zero.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    # Dot product divided by norms, safely float casted
    sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    # Clip to [0.0, 1.0] to prevent floating point inaccuracies going slightly above 1
    # or mapping entirely negative vectors gracefully based on constraints
    # (Typically cosine sim is -1 to 1, but we usually scale or clamp it. 
    # For embedding similarity, the prompt implies "0.0-1.0")
    return max(0.0, min(1.0, sim))

def main():
    import time
    print("Testing embeddings.py...")
    
    # 1. Load Model
    start = time.time()
    model = load_model()
    print(f"Model loaded in {time.time() - start:.2f} seconds.")
    
    # 2. Embed Single Text
    single_text = "I build machine learning retrieval systems."
    vec1 = embed_text(single_text)
    assert vec1.shape == (384,)
    assert vec1.dtype == np.float32
    print("Embed single text \u2713")
    
    # 3. Embed Batch
    batch = ["First string", "Second string", "Third string"]
    vecs = embed_batch(batch)
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32
    print("Embed batch of 3 texts \u2713")
    
    # 4. Cosine Similarity - Identical
    vec2 = embed_text(single_text)
    sim_ident = cosine_similarity(vec1, vec2)
    assert sim_ident > 0.99
    print(f"Similarity between identical vectors: {sim_ident:.4f} \u2713")
    
    # 5. Cosine Similarity - Orthogonal/Unrelated
    unrelated = embed_text("The quick brown fox jumps over the lazy dog.")
    sim_unrelated = cosine_similarity(vec1, unrelated)
    assert sim_unrelated < 0.5 # They shouldn't be very similar
    print(f"Similarity between unrelated vectors: {sim_unrelated:.4f} \u2713")
    
    # Orthogonal mathematically
    ortho_vec = np.zeros(384)
    ortho_vec[0] = 1.0
    zero_sim = cosine_similarity(vec1, ortho_vec)
    print(f"Similarity with manual orthogonal/sparse vector: {zero_sim:.4f} \u2713")
    
    # 6. Edge cases
    empty_vec = embed_text("")
    assert empty_vec.shape == (384,)
    assert not np.any(empty_vec) # Should be all zeros based on spec
    zero_sim2 = cosine_similarity(vec1, empty_vec)
    assert zero_sim2 == 0.0
    print("Empty text handling \u2713")
    
    print("\n\u2705 Embedding service works".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    main()
