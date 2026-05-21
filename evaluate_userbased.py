"""
Evaluate the backend user-based model (backend/models/userbased.py) on the test
users defined in constants.py (Constant.TEST_USERS_USERBASED).

Usage:
    python evaluate_userbased.py

The script loads pre-generated artifacts from backend/artifacts/, runs recommend()
for each test user, then computes ranking metrics against held-out ground truth ratings.

Metrics reported per user and averaged:
    Precision@k   fraction of top-k recommendations that are "relevant"
    Recall@k      fraction of relevant items that appear in top-k
    NDCG@k        normalised discounted cumulative gain (uses actual ratings as gains)
    HitRate@k     1 if at least one relevant item is in top-k, else 0
    MRR           mean reciprocal rank of the first relevant item
    Coverage      fraction of ground-truth items the model knew about
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The root directory must be first in sys.path so that pickle can find models.UserBased
# (defined in root models.py) when loading userbased_model.pkl. Adding backend/ to
# sys.path would shadow root models.py with the backend/models/ package, breaking unpickling.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Load backend/models/userbased.py directly to avoid the sys.path naming conflict
_spec = importlib.util.spec_from_file_location(
    "backend_userbased", _root / "backend" / "models" / "userbased.py"
)
ub = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(ub)  # type: ignore[union-attr]

from constants import Constant as C


# ── Metric helpers ─────────────────────────────────────────────────────────────

def precision_at_k(rec_ids: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for mid in rec_ids[:k] if mid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(rec_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for mid in rec_ids[:k] if mid in relevant)
    return hits / len(relevant)


def ndcg_at_k(rec_ids: list[int], ground_truth: dict[int, float], k: int) -> float:
    dcg = sum(
        ground_truth.get(mid, 0.0) / np.log2(i + 2)
        for i, mid in enumerate(rec_ids[:k])
    )
    ideal = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(rec_ids: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(mid in relevant for mid in rec_ids[:k]) else 0.0


def mrr(rec_ids: list[int], relevant: set[int]) -> float:
    for rank, mid in enumerate(rec_ids, start=1):
        if mid in relevant:
            return 1.0 / rank
    return 0.0


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate() -> None:
    print("Loading user-based model artifacts...")
    try:
        ub.load()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] Artifact not found: {exc}")
        print("Generate artifacts first by running your user-based training notebook.")
        sys.exit(1)

    test_users = C.TEST_USERS_USERBASED
    k_values = C.EVAL_K_VALUES
    threshold = C.EVAL_THRESHOLD
    n_recs = max(k_values) * 5  # fetch more than k_max so ranking metrics are meaningful

    rows = []

    for user in test_users:
        name = user["name"]
        input_ratings: dict[int, float] = user["input"]
        ground_truth: dict[int, float] = user["ground_truth"]

        # Sanity check: ground-truth must not overlap with input
        overlap = set(input_ratings) & set(ground_truth)
        if overlap:
            print(f"[WARN] {name}: movie(s) {overlap} appear in both input and ground_truth — skipping user.")
            continue

        recs = ub.recommend(input_ratings, n=n_recs)
        rec_ids = [mid for mid, _ in recs]

        relevant = {mid for mid, r in ground_truth.items() if r >= threshold}

        # Coverage: share of ground-truth items the model can potentially recommend
        # (i.e. items that were in the training set)
        known_gt = sum(1 for mid in ground_truth if ub._item_index.get(mid) is not None)
        coverage = known_gt / len(ground_truth) if ground_truth else 0.0

        row: dict = {
            "user": name,
            "n_input": len(input_ratings),
            "n_gt": len(ground_truth),
            "n_relevant": len(relevant),
            "n_recs_returned": len(recs),
            "coverage": round(coverage, 3),
        }

        for k in k_values:
            row[f"P@{k}"]       = round(precision_at_k(rec_ids, relevant, k), 4)
            row[f"R@{k}"]       = round(recall_at_k(rec_ids, relevant, k), 4)
            row[f"NDCG@{k}"]    = round(ndcg_at_k(rec_ids, ground_truth, k), 4)
            row[f"HR@{k}"]      = round(hit_rate_at_k(rec_ids, relevant, k), 4)

        row["MRR"] = round(mrr(rec_ids, relevant), 4)
        rows.append(row)

    if not rows:
        print("No valid test users to evaluate.")
        return

    df = pd.DataFrame(rows)

    # Append mean row over numeric columns
    numeric_cols = [c for c in df.columns if c != "user"]
    mean_vals = df[numeric_cols].mean().round(4).to_dict()
    df = pd.concat([df, pd.DataFrame([{"user": "MEAN", **mean_vals}])], ignore_index=True)

    print("\n=== User-Based Model Evaluation ===\n")
    print(df.to_string(index=False))

    C.EVALUATION_PATH.mkdir(parents=True, exist_ok=True)
    out_path = C.EVALUATION_PATH / "userbased_eval.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    evaluate()
