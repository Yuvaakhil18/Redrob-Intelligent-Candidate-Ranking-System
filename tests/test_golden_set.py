import unittest
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalizer import normalize
from scorer import score_candidate

class TestGoldenSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jd_emb = np.ones(384).astype(np.float32) / np.sqrt(384)
        
        # Candidate 1: Perfect Fit
        cls.c1 = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "headline": "Senior AI Engineer",
                "summary": "Expert in ML, embeddings, and retrieval",
                "current_title": "Senior AI Engineer",
                "current_company": "Stripe",
                "years_of_experience": 7.0,
                "education_level": "Masters",
                "location": {"city": "Pune", "country": "India"}
            },
            "career_history": [
                {
                    "company_name": "Stripe",
                    "title": "Senior AI Engineer",
                    "duration_months": 84,
                    "description": "Led embeddings generation, retrieval algorithms, and vector DB optimization for LLM pipelines."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 60, "endorsements": 100},
                {"name": "Machine Learning", "proficiency": "expert", "duration_months": 60, "endorsements": 80},
                {"name": "Embeddings", "proficiency": "expert", "duration_months": 48, "endorsements": 80},
                {"name": "Vector Database", "proficiency": "advanced", "duration_months": 36, "endorsements": 50},
                {"name": "Retrieval", "proficiency": "expert", "duration_months": 40, "endorsements": 90}
            ],
            "redrob_signals": {
                "open_to_work_flag": True,
                "last_active_date": "2026-06-14T00:00:00Z",
                "recruiter_response_rate": 0.95,
                "interview_completion_rate": 1.0,
                "github_activity_score": 100,
                "verified_email": True,
                "verified_phone": True
            }
        }
        
        # Candidate 2: Keyword Stuffer
        cls.c2 = {
            "candidate_id": "CAND_0000002",
            "profile": {
                "headline": "Marketing Manager",
                "current_title": "Marketing Manager",
                "current_company": "AdCorp",
                "years_of_experience": 2.0,
                "education_level": "Bachelors",
                "location": {"city": "Delhi", "country": "India"}
            },
            "career_history": [
                {
                    "company_name": "AdCorp",
                    "title": "Marketing Manager",
                    "duration_months": 24,
                    "description": "Managed campaigns and budgets."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
                {"name": "AI", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
            ],
            "behavioral_signals": {
                "open_to_work_flag": False,
                "last_active_date_days_ago": 300,
                "notice_period_days": 90
            }
        }
        
        # Candidate 3: Consulting Only
        cls.c3 = {
            "candidate_id": "CAND_0000003",
            "profile": {
                "headline": "Data Engineer",
                "current_title": "Data Engineer",
                "current_company": "TCS",
                "years_of_experience": 8.0,
                "education_level": "Bachelors",
                "location": {"city": "Pune", "country": "India"}
            },
            "career_history": [
                {
                    "company_name": "TCS",
                    "title": "Data Engineer",
                    "duration_months": 96,
                    "description": "ETL pipelines for enterprise clients."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 10}
            ],
            "behavioral_signals": {
                "open_to_work_flag": True,
                "last_active_date_days_ago": 15,
                "notice_period_days": 60
            }
        }

        # Candidate 4: Overqualified Inactive
        cls.c4 = {
            "candidate_id": "CAND_0000004",
            "profile": {
                "headline": "VP of Engineering",
                "current_title": "Director of Data Science",
                "current_company": "BigTech",
                "years_of_experience": 15.0,
                "education_level": "Phd",
                "location": {"city": "Pune", "country": "India"}
            },
            "career_history": [
                {
                    "company_name": "BigTech",
                    "title": "Director of Data Science",
                    "duration_months": 180,
                    "description": "Led division of 200 ML engineers."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 120, "endorsements": 500}
            ],
            "behavioral_signals": {
                "open_to_work_flag": False,
                "last_active_date_days_ago": 540,
                "notice_period_days": 90
            }
        }

        # Candidate 5: Good Fit Long Notice
        cls.c5 = {
            "candidate_id": "CAND_0000005",
            "profile": {
                "headline": "AI Engineer",
                "current_title": "AI Engineer",
                "current_company": "StartupAI",
                "years_of_experience": 6.0,
                "education_level": "Bachelors",
                "location": {"city": "Pune", "country": "India"}
            },
            "career_history": [
                {
                    "company_name": "StartupAI",
                    "title": "AI Engineer",
                    "duration_months": 72,
                    "description": "Built ML models and retrieval systems."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 60, "endorsements": 30}
            ],
            "behavioral_signals": {
                "open_to_work_flag": True,
                "last_active_date_days_ago": 5,
                "notice_period_days": 120
            }
        }

    def _eval_candidate(self, raw_cand, emb_sim=1.0):
        norm = normalize(raw_cand)
        # Mock career embedding to force the desired semantic similarity
        # Since jd_emb is all positive, career_emb can just be identical or scaled
        c_emb = self.jd_emb * emb_sim
        score, dims = score_candidate(norm, self.jd_emb, c_emb)
        print(f"\nEvaluating: {raw_cand['candidate_id']}")
        print(f"Final Score: {score:.3f}")
        print(f"Dimensions: {dims}")
        return score, dims

    def test_perfect_fit(self):
        score, dims = self._eval_candidate(self.c1, emb_sim=1.0)
        self.assertGreater(score, 0.85, "Perfect fit score too low")

    def test_keyword_stuffer(self):
        score, dims = self._eval_candidate(self.c2, emb_sim=-0.5)
        self.assertLess(score, 0.40, "Keyword stuffer score too high")

    def test_consulting_only(self):
        score, dims = self._eval_candidate(self.c3, emb_sim=0.5)
        self.assertGreaterEqual(score, 0.35)
        self.assertLessEqual(score, 0.55)

    def test_overqualified(self):
        score, dims = self._eval_candidate(self.c4, emb_sim=0.8)
        self.assertLess(score, 0.60, "Overqualified inactive score too high")

    def test_good_with_friction(self):
        score, dims = self._eval_candidate(self.c5, emb_sim=0.8)
        self.assertGreaterEqual(score, 0.50)
        self.assertLessEqual(score, 0.75)

if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=2)
    if result.result.wasSuccessful():
        print("\n✅ Golden set regression testing passed successfully")
