import argparse
import os
import numpy as np

from embeddings import embed_text

DEFAULT_JD_TEXT = """
# Senior AI Engineer

## Role Context
We are looking for a Senior AI Engineer to join our core ranking and retrieval team. You will be responsible for building highly scalable systems to match candidates to open roles using modern NLP and vector search techniques.

## Must-Haves
- 6+ years of software engineering experience
- Expert proficiency in Python and Machine Learning
- Extensive experience with embeddings and semantic search
- Hands-on experience with vector databases (Pinecone, Milvus, or Qdrant)
- Experience deploying machine learning models to production

## Nice-to-Haves
- Background in Search & Recommendation systems
- Experience with large language models (LLMs) and RAG pipelines
- Strong cloud infrastructure experience (AWS/GCP)
"""

def extract_key_sections(jd_text: str) -> str:
    """
    Extracts and combines the 'Role Context', 'Must-Haves', and 'Nice-to-Haves' 
    from a job description text to create a dense, coherent string for embedding.
    """
    # For a simple robust extraction, we will strip out massive boilerplate
    # and focus on the core semantic meaning. If the text has Markdown headers,
    # we can try to parse them, or just embed the whole text if it's short.
    # Here we just clean it up and combine it coherently.
    
    lines = jd_text.strip().split('\n')
    extracted = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip markdown hashes and bullets to make dense text
        line = line.replace('#', '').replace('-', '').strip()
        extracted.append(line)
        
    return " ".join(extracted)

def embed_jd(jd_text: str, output_path: str) -> np.ndarray:
    """
    Embeds a job description text and saves the resulting (384,) numpy array to disk.
    """
    print("Extracting key sections from Job Description...")
    coherent_text = extract_key_sections(jd_text)
    
    print(f"Generating embedding for text (length: {len(coherent_text)} characters)...")
    embedding = embed_text(coherent_text)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Saving JD embedding to {output_path}...")
    np.save(output_path, embedding)
    
    return embedding

def main():
    parser = argparse.ArgumentParser(description="JD Embedding Generator")
    parser.add_argument("--jd-file", type=str, help="Path to job description text/markdown file")
    parser.add_argument("--output", type=str, default="embeddings_output/jd_embedding.npy", help="Path to save the resulting .npy file")
    
    args = parser.parse_args()
    
    if args.jd_file:
        try:
            with open(args.jd_file, 'r', encoding='utf-8') as f:
                jd_text = f.read()
            print(f"Loaded Job Description from {args.jd_file}")
        except Exception as e:
            print(f"Error reading JD file {args.jd_file}: {e}")
            print("Falling back to default JD.")
            jd_text = DEFAULT_JD_TEXT
    else:
        print("No --jd-file provided. Using default JD.")
        jd_text = DEFAULT_JD_TEXT
        
    # Generate and save
    emb = embed_jd(jd_text, args.output)
    
    # Validation tests as requested
    assert emb.shape == (384,), f"Expected shape (384,), got {emb.shape}"
    assert emb.dtype == np.float32, f"Expected dtype float32, got {emb.dtype}"
    assert os.path.exists(args.output), f"File {args.output} was not created"
    
    # Test loading back
    loaded_emb = np.load(args.output)
    assert np.allclose(emb, loaded_emb), "Loaded embedding does not match original"
    
    print("\n\u2705 JD embedding complete".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    main()
