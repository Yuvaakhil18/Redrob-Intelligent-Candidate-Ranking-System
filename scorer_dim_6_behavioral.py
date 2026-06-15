from typing import Dict
from datetime import datetime, timezone
import dateutil.parser
from loader import load_sample
from normalizer import normalize

def score_dim6_behavioral(candidate_normalized: Dict) -> float:
    """
    Dimension 6: Behavioral Signals Multiplier.
    Evaluates engagement, responsiveness, and activity to yield a multiplier between 0.4 and 1.2.
    """
    multiplier = 1.0
    signals = candidate_normalized.get('redrob_signals', {})
    if not signals:
        return multiplier
        
    # 1. Open to work
    if signals.get('open_to_work_flag') is True:
        multiplier += 0.1
        
    # 2. Last active date
    last_active = signals.get('last_active_date')
    if last_active:
        try:
            dt = dateutil.parser.isoparse(last_active)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_ago = (now - dt).days
            
            if days_ago <= 30:
                multiplier += 0.15
            elif days_ago <= 90:
                multiplier += 0.05
            elif days_ago > 365:
                multiplier -= 0.2
        except Exception:
            pass # Ignore malformed dates
            
    # 3. Recruiter response rate
    resp_rate = signals.get('recruiter_response_rate')
    if resp_rate is not None:
        if resp_rate >= 0.7:
            multiplier += 0.1
        elif resp_rate < 0.3:
            multiplier -= 0.15
            
    # 4. Interview completion rate
    int_rate = signals.get('interview_completion_rate')
    if int_rate is not None:
        if int_rate >= 0.8:
            multiplier += 0.05
        elif int_rate < 0.5:
            multiplier -= 0.1
            
    # 5. Github activity score
    gh_score = signals.get('github_activity_score')
    if gh_score is not None and gh_score > 30:
        multiplier += 0.05
        
    # 6. Verified email and phone
    email = signals.get('verified_email', False)
    phone = signals.get('verified_phone', False)
    if email and phone:
        multiplier += 0.05
        
    # 7. Offer acceptance rate
    oar = signals.get('offer_acceptance_rate')
    if oar is not None:
        if oar >= 0.8:
            multiplier += 0.05
        elif 0 < oar <= 0.3:
            multiplier -= 0.05
            
    # Floor at 0.4, Cap at 1.2
    return max(0.4, min(1.2, multiplier))

def main():
    sample_path = r"..\India_runs_data_and_ai_challenge\sample_candidates.json"
    candidates = load_sample(sample_path, n=3)
    
    for c in candidates:
        norm_c = normalize(c)
        mult = score_dim6_behavioral(norm_c)
        
        assert 0.4 <= mult <= 1.2
        
        print(f"Candidate: {c['candidate_id']}")
        print(f"  Open to Work: {norm_c.get('redrob_signals', {}).get('open_to_work_flag')}")
        print(f"  Response Rate: {norm_c.get('redrob_signals', {}).get('recruiter_response_rate')}")
        print(f"  Final Multiplier: {mult:.3f}\n")

    print("\u2705 scorer_dim_6_behavioral.py works".encode("utf-8").decode("utf-8", "ignore"))

if __name__ == '__main__':
    main()
