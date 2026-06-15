# RedRob Ranker

An offline + online candidate ranking engine designed for high performance semantic and heuristic candidate evaluation.

## Architecture

This project strictly splits processing into an **Offline Precomputation Phase** and an **Online Timed Ranking Phase** to meet performance constraints (<300s).

### 1. Offline Precomputation (`precompute.py`)
This phase takes all candidate profiles and computing vector embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`).
- Parses 100K JSONL candidates
- Embeds candidate profiles in batches
- Outputs `.npy` memory-mappable files.
- **Runtime:** ~30-35 mins (runs completely offline before the timed window).

### 2. Online Timed Ranking (`rank.py`)
This is the extremely fast, timed execution script.
- Uses memory-mapped embeddings (instant loading)
- Calculates heuristic modifiers (honeypot detection, experience penalties)
- Deterministically generates reasoning strings without LLMs.
- Outputs `submission.csv` strictly formatted to spec.
- **Runtime:** ~20 seconds for 100,000 candidates (limit is 300s).

## Quick Start

### Installation
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix
source venv/bin/activate

pip install -r requirements.txt
```

### Usage
**Step 1: Offline Precomputation**
```bash
python precompute.py --candidates candidates.jsonl.gz --output-dir embeddings_output
```

**Step 2: Online Ranking (Timed)**
```bash
python rank.py --candidates candidates.jsonl.gz --embeddings embeddings_output --out submission.csv
```

## Performance Metrics
- **Precompute Time:** ~32 minutes
- **Ranking Time:** 19.2 seconds (for 100,000 candidates)
- **Memory Footprint:** Highly optimized. Embeddings are `mmap`-ed and candidates parsed efficiently.
- **Honeypot Detection:** Found 49 impossible profiles, penalized to 0.05.

## Modules

- **`loader.py`**: Candidate parser generator
- **`normalizer.py`**: Computes duration, metadata normalization
- **`embeddings.py`**: Model integration (`SentenceTransformers`)
- **`scorer.py`**: Math logic for cosine + scalar modifiers
- **`honeypot.py`**: Analytics for impossible claim detection
- **`reasoner.py`**: Generates logical deterministic output text
- **`csv_writer.py`**: Formats output strictly to 4 decimals

## Testing
Run the comprehensive test suite to verify pipeline integrity:
```bash
python tests/test_all.py
```
To validate output CSV:
```bash
python validate_submission.py submission.csv
```
