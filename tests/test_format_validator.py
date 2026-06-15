import unittest
import os
import subprocess
import csv
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_writer import write_submission_csv

class TestFormatValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv = "test_submission.csv"
        cls.validator_script = os.path.join("..", "India_runs_data_and_ai_challenge", "validate_submission.py")
        
        # 1. GENERATE TEST CSV
        # Create 100 valid rows meeting spec
        rankings = []
        score = 1.0
        for i in range(1, 101):
            rankings.append({
                'candidate_id': f'CAND_{i:07d}',
                'rank': i,
                'score': round(score, 4),
                'reasoning': f"Strong profile fit for rank {i} with solid background."
            })
            score -= 0.005
            
        write_submission_csv(cls.test_csv, rankings)
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)

    def test_format_with_official_validator(self):
        """Run official validator script and parse output"""
        self.assertTrue(os.path.exists(self.validator_script), "Official validator not found")
        
        result = subprocess.run(
            ["python", self.validator_script, self.test_csv],
            capture_output=True, text=True
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            print(f"\nValidator Failed. Output:\n{output}")
            
        self.assertEqual(result.returncode, 0, "Validator returned non-zero exit code")
        self.assertIn("Submission is valid", output, "Validator did not output success message")

    def test_manual_line_count(self):
        """Verify 100 data rows + 1 header = 101 lines"""
        with open(self.test_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 101)

    def test_manual_data_integrity(self):
        """Verify: Ranks 1-100 unique, scores non-increasing, CAND_XXXXXXX ids, non-empty reasoning"""
        with open(self.test_csv, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            
        self.assertEqual(len(reader), 100)
        
        expected_columns = ['candidate_id', 'rank', 'score', 'reasoning']
        self.assertListEqual(list(reader[0].keys()), expected_columns)
        
        ranks = []
        prev_score = float('inf')
        
        for row in reader:
            # Check ID format
            self.assertTrue(row['candidate_id'].startswith("CAND_"))
            self.assertEqual(len(row['candidate_id']), 12)
            
            # Check Rank
            rank = int(row['rank'])
            ranks.append(rank)
            
            # Check Score
            score = float(row['score'])
            self.assertTrue(0.0 <= score <= 1.2)
            self.assertLessEqual(score, prev_score)
            prev_score = score
            
            # Check Reasoning
            self.assertTrue(len(row['reasoning'].strip()) > 5)
            
        self.assertListEqual(ranks, list(range(1, 101)))

if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=2)
    if result.result.wasSuccessful():
        print("\n✅ Format validation passed")
