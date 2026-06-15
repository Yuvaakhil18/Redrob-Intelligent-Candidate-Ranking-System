import json
import gzip
import os
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_candidate(candidate: Dict) -> Tuple[bool, List[str]]:
    """
    Validates a candidate dictionary against the expected schema.
    """
    errors = []
    
    if not isinstance(candidate, dict):
        return False, ["Candidate is not a dictionary"]
        
    # Required top-level fields
    for field in ['candidate_id', 'profile', 'career_history']:
        if field not in candidate:
            errors.append(f"Missing required field: {field}")
            
    if errors:
        return False, errors
        
    # Profile validation
    profile = candidate.get('profile', {})
    if 'years_of_experience' in profile:
        yoe = profile['years_of_experience']
        if not isinstance(yoe, (int, float)):
            errors.append(f"years_of_experience must be a number, got {type(yoe)}")
            
    # Career history validation
    career_history = candidate.get('career_history', [])
    if not isinstance(career_history, list) or len(career_history) < 1:
        errors.append("career_history must be a list with at least 1 entry")
        
    # Skills validation
    valid_proficiencies = {'beginner', 'intermediate', 'advanced', 'expert'}
    skills = candidate.get('skills', [])
    if isinstance(skills, list):
        for skill in skills:
            prof = skill.get('proficiency')
            if prof and prof not in valid_proficiencies:
                errors.append(f"Invalid proficiency value: {prof}")
                
    if errors:
        return False, errors
        
    return True, []

def load_candidates(file_path: str, sample_size: int = None) -> List[Dict]:
    """
    Loads candidates from .json, .jsonl, or .jsonl.gz files.
    Validates them and returns a clean list of valid candidates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    valid_candidates = []
    processed_count = 0
    
    # Handle single JSON array file (like sample_candidates.json)
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for c in data:
                        is_valid, errors = validate_candidate(c)
                        if is_valid:
                            valid_candidates.append(c)
                        else:
                            logging.warning(f"Skipping invalid candidate: {errors}")
                            
                        processed_count += 1
                        if processed_count % 10000 == 0:
                            logging.info(f"Processed {processed_count} candidates...")
                            
                        if sample_size and len(valid_candidates) >= sample_size:
                            break
            except json.JSONDecodeError as e:
                logging.error(f"Malformed JSON in file: {e}")
        return valid_candidates

    # Handle JSONL and GZipped JSONL
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'
    encoding = 'utf-8'
    
    with open_func(file_path, mode, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                c = json.loads(line)
                is_valid, errors = validate_candidate(c)
                if is_valid:
                    valid_candidates.append(c)
                else:
                    logging.warning(f"Skipping invalid candidate: {errors}")
            except json.JSONDecodeError as e:
                logging.error(f"Malformed JSON line, skipping. Error: {e}")
                continue
                
            processed_count += 1
            if processed_count % 10000 == 0:
                logging.info(f"Processed {processed_count} candidates...")
                
            if sample_size and len(valid_candidates) >= sample_size:
                break
                
    return valid_candidates

def load_sample(file_path: str, n: int = 50) -> List[Dict]:
    """
    Loads the first N valid candidates. Used for testing.
    """
    return load_candidates(file_path, sample_size=n)

def main():
    # Use the sample dataset provided for testing
    sample_path = r"..\India_runs_data_and_ai_challenge\sample_candidates.json"
    
    print(f"Testing loader on: {sample_path}")
    candidates = load_sample(sample_path, n=50)
    
    print(f"\nLoaded {len(candidates)} valid candidates.")
    
    if len(candidates) > 0:
        print("\n✅ loader.py works")
        c = candidates[0]
        print(f"Sample Candidate ID: {c.get('candidate_id')}")
        print(f"Years of Experience: {c.get('profile', {}).get('years_of_experience')}")
        print(f"Career History length: {len(c.get('career_history', []))}")
    else:
        print("\n❌ Failed to load valid candidates.")

if __name__ == '__main__':
    main()
