from typing import Dict

def extract_semantic_text(candidate: Dict) -> str:
    """
    Cleans and concatenates career history, skills, and summary for semantic embedding.
    """
    profile = candidate.get('profile', {})
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    
    parts = []
    
    # 1. Headline and Summary
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    if headline:
        parts.append(f"Headline: {headline}")
    if summary:
        parts.append(f"Summary: {summary}")
        
    # 2. Career History
    parts.append("Career History:")
    for role in career_history:
        title = role.get('title', '')
        company = role.get('company', '')
        duration = role.get('duration_months', 0)
        desc = role.get('description', '')
        role_str = f"Role: {title} at {company} ({duration} months). Description: {desc}"
        parts.append(role_str)
        
    # 3. Skills
    parts.append("Skills:")
    skill_strs = []
    for skill in skills:
        name = skill.get('name', '')
        prof = skill.get('proficiency', '')
        duration = skill.get('duration_months', 0)
        skill_strs.append(f"{name} ({prof}, {duration} months)")
    
    parts.append(", ".join(skill_strs))
    
    return "\n".join(parts)
