"""
Offline evaluation script for the SHL Assessment Recommender.
Computes Precision@K and NDCG@K against a labelled ground-truth dataset.

Usage:
    python -m eval          # runs at K=5 and K=10
    python -m eval --k 3    # custom K
"""
import math
import time
import argparse
from fastapi.testclient import TestClient
from main import app

# Delay between queries (seconds) to respect Gemini free-tier rate limits
QUERY_DELAY_SECONDS = 35

client = TestClient(app)

# ---------------------------------------------------------------------------
# Ground Truth Eval Set
# Each entry has:
#   "query"    - a natural language request sent as a single user message
#   "relevant" - list of exact assessment names that should appear in top-K
# ---------------------------------------------------------------------------
EVAL_SET = [
    {
        "query": "I need to assess a mid-level software engineer with Python and AWS skills.",
        "relevant": [
            "Amazon Web Services (AWS) Development (New)",
            "Python (New)"
        ]
    },
    {
        "query": "Looking for agile methodology and testing assessments for a developer.",
        "relevant": [
            "Agile Testing (New)",
            "Agile Software Development"
        ]
    },
    {
        "query": "I want to hire an Angular front-end developer. What tests do you have?",
        "relevant": [
            "AngularJS (New)",
            "Angular 6 (New)"
        ]
    },
    {
        "query": "Assess a data engineering candidate who knows Apache Spark and Kafka.",
        "relevant": [
            "Apache Spark (New)",
            "Apache Kafka (New)",
            "Apache Hadoop (New)"
        ]
    },
    {
        "query": "I need .NET framework tests for a mid-level Windows developer.",
        "relevant": [
            ".NET Framework 4.5",
            "ASP.NET 4.5",
            "ASP .NET with C# (New)"
        ]
    },
    {
        "query": "Android mobile development skills test for a professional contributor.",
        "relevant": [
            "Android Development (New)"
        ]
    },
    {
        "query": "Looking for cloud infrastructure tests specifically on AWS delivery and security.",
        "relevant": [
            "Amazon Web Services (AWS) Development (New)"
        ]
    },
]


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, rec in enumerate(recommended[:k]):
        if rec in relevant:
            dcg += 1.0 / math.log2(i + 2)  # log2(rank+1), rank is 1-indexed
    return dcg


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized DCG at K. Ideal is when all relevant items are at the top."""
    actual_dcg = dcg_at_k(recommended, relevant, k)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_eval(k: int = 5):
    print(f"\n{'='*65}")
    print(f"  SHL Recommender Offline Evaluation — Precision@{k} & NDCG@{k}")
    print(f"{'='*65}\n")

    all_p_at_k = []
    all_ndcg_at_k = []

    for idx, item in enumerate(EVAL_SET):
        query = item["query"]
        relevant = set(item["relevant"])

        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": query}]}
        )

        if response.status_code != 200:
            print(f"  [{idx+1}] HTTP {response.status_code} — SKIPPED: {query[:60]}")
            continue

        data = response.json()
        recommended_names = [r["name"] for r in data.get("recommendations", [])]

        p = precision_at_k(recommended_names, relevant, k)
        n = ndcg_at_k(recommended_names, relevant, k)

        all_p_at_k.append(p)
        all_ndcg_at_k.append(n)

        # Per-query result display
        hits = [r for r in recommended_names[:k] if r in relevant]
        print(f"  [{idx+1}] {query[:60]}...")
        print(f"       Expected : {sorted(relevant)}")
        print(f"       Got top-{k}: {recommended_names[:k]}")
        print(f"       Hits     : {hits}")
        print(f"       P@{k}: {p:.2f}  |  NDCG@{k}: {n:.4f}\n")

        # Rate-limit guard: wait between queries on free tier
        if idx < len(EVAL_SET) - 1:
            print(f"       ⏳ Waiting {QUERY_DELAY_SECONDS}s for Gemini rate limit...")
            time.sleep(QUERY_DELAY_SECONDS)


    # Aggregate results
    evaluated = len(all_p_at_k)
    avg_p = sum(all_p_at_k) / evaluated if evaluated else 0.0
    avg_ndcg = sum(all_ndcg_at_k) / evaluated if evaluated else 0.0

    print(f"{'='*65}")
    print(f"  AGGREGATE ({evaluated}/{len(EVAL_SET)} queries evaluated)")
    print(f"  Mean Precision@{k} : {avg_p:.4f}")
    print(f"  Mean NDCG@{k}      : {avg_ndcg:.4f}")
    print(f"{'='*65}\n")

    return {"precision_at_k": avg_p, "ndcg_at_k": avg_ndcg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHL Recommender offline evaluation.")
    parser.add_argument("--k", type=int, default=None, help="Evaluate at a specific K. Defaults to both K=5 and K=10.")
    args = parser.parse_args()

    if args.k:
        run_eval(k=args.k)
    else:
        run_eval(k=5)
        run_eval(k=10)
