from typing import Dict, List
from loader import load_sample

CONSULTING_FIRMS = {
    'infosys', 'tcs', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl', 
    'tech mahindra', 'mphasis', 'hexaware', 'everest', 'sapient', 
    'dxc technology', 'synechron', 'genpact', 'sonata software', 
    'mindtree', 'persistent'
}

def _infer_company_type(company_name: str, company_size: str) -> str:
    if company_name and company_name.lower().strip() in CONSULTING_FIRMS:
        return "consulting"
        
    if not company_size:
        return "unknown"
        
    size_str = company_size.replace(',', '').replace('+', '').strip()
    
    # Simple mapping based on known ranges
    if size_str in ('1-10', '11-50', '1-50'):
        return "startup"
    elif size_str in ('51-200', '201-500'):
        return "scaleup"
    elif size_str in ('501-1000', '1001-5000', '5001-10000', '10001'):
        return "enterprise"
    return "unknown"

def normalize(candidate: Dict) -> Dict:
    """
    Normalizes candidate data by adding computed fields.
    """
    profile = candidate.get('profile', {})
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    
    # 1. profile_text
    p_text_parts = []
    if profile.get('headline'): p_text_parts.append(profile['headline'])
    if profile.get('summary'): p_text_parts.append(profile['summary'])
    if profile.get('current_title'): p_text_parts.append(profile['current_title'])
    candidate['profile_text'] = " | ".join(p_text_parts)
    
    # 2. career_text, company_list, company_types
    c_text_parts = []
    company_list = []
    company_types = []
    total_months = 0
    consulting_count = 0
    
    for job in career_history:
        # Job description
        desc = job.get('description', '')
        if desc:
            c_text_parts.append(desc)
            
        # Company lists and types
        company_name = job.get('company_name', '')
        if company_name:
            company_list.append(company_name)
            
            # Type inference
            c_type = _infer_company_type(company_name, job.get('company_size', ''))
            company_types.append(c_type)
            if c_type == 'consulting':
                consulting_count += 1
                
        # Duration
        total_months += job.get('duration_months', 0)
        
    candidate['career_text'] = "\n".join(c_text_parts)
    candidate['company_list'] = company_list
    candidate['company_types'] = company_types
    
    # 3. is_consulting_only
    if len(company_list) > 0:
        candidate['is_consulting_only'] = (consulting_count / len(company_list)) >= 0.8
    else:
        candidate['is_consulting_only'] = False
        
    # 4. years_experience_consistent
    stated_years = float(profile.get('years_of_experience', 0.0))
    calculated_years = total_months / 12.0
    
    difference = abs(stated_years - calculated_years)
    allowed_discrepancy = max(1.0, stated_years * 0.1)
    
    candidate['years_experience_consistent'] = difference <= allowed_discrepancy
    
    # 5. skills_dict
    skills_dict = {}
    for skill in skills:
        name = skill.get('name')
        if name:
            skills_dict[name] = {
                'proficiency': skill.get('proficiency'),
                'endorsements': skill.get('endorsements', 0),
                'duration_months': skill.get('duration_months', 0)
            }
    candidate['skills_dict'] = skills_dict
    
    # 6. Default honeypot fields
    candidate['is_suspicious'] = False
    candidate['honeypot_flags'] = []
    
    return candidate

def main():
    sample_path = r"..\India_runs_data_and_ai_challenge\sample_candidates.json"
    print(f"Testing normalizer on: {sample_path}")
    
    try:
        candidates = load_sample(sample_path, n=5)
    except Exception as e:
        print(f"Error loading candidates: {e}")
        return
        
    if not candidates:
        print("No candidates loaded.")
        return
        
    success = True
    for c in candidates:
        try:
            norm_c = normalize(c)
            # Verify fields
            assert isinstance(norm_c['career_text'], str)
            assert isinstance(norm_c['skills_dict'], dict)
            assert isinstance(norm_c['company_types'], list)
            assert isinstance(norm_c['is_consulting_only'], bool)
            assert isinstance(norm_c['years_experience_consistent'], bool)
        except Exception as e:
            print(f"Validation failed for candidate {c.get('candidate_id')}: {e}")
            success = False
            
    if success:
        print("\n" + "\u2705 normalizer.py works".encode("utf-8").decode("utf-8", "ignore"))
        
        # Print a sample of computed fields
        sample = candidates[0]
        print(f"\nSample computed fields for {sample['candidate_id']}:")
        print(f"- Company List: {sample['company_list']}")
        print(f"- Company Types: {sample['company_types']}")
        print(f"- Consulting Only: {sample['is_consulting_only']}")
        print(f"- Years Consistent: {sample['years_experience_consistent']}")
    else:
        print("\n❌ normalizer.py failed validation tests")

if __name__ == '__main__':
    main()
