"""Turn raw result records into per-model metrics and one aggregate score.

Pipeline:  record -> quality 0..1  ->  per-category mean  ->  per-model mean over categories (macro average,
so every category counts equally, however many tests it has)  ->  normalize cost and latency to 0..1
->  weighted sum = aggregate score.
The dashboard (dashboard_template.html) repeats exactly this math in JavaScript so that sliders and filters
recompute live; keep the two in sync (a test compares them on sample data when a JS engine is available).
"""
import math
import statistics

import config


def normalize_weights(weights):
    """Make the weights sum to 1. All zero -> equal weights."""
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        return {k: 1 / len(weights) for k in weights}
    return {k: max(0.0, w) / total for k, w in weights.items()}


def quality_of(rec):
    """Quality of one answer on a 0..1 scale, or None if it should not count.

    * API failure (throttled, access denied...): None -> excluded, it is not the model's fault.
    * Empty answer (model returned no text): 0.0.
    * Automatic score and judge score (1-5 mapped to 0-1) are mixed with config.AUTO_WEIGHT when both exist.
    """
    if not rec["ok"]:
        return 0.0 if rec.get("error_kind") == "empty" else None
    auto = rec["auto"]["score"] if rec.get("auto") else None
    judge_mean = rec["judge"]["mean"] if rec.get("judge") else None
    judge = (judge_mean - 1) / 4 if judge_mean is not None else None
    if auto is not None and judge is not None:
        return config.AUTO_WEIGHT * auto + (1 - config.AUTO_WEIGHT) * judge
    if auto is not None:
        return auto
    return judge   # may be None when the judge failed and the test has no checks


def cost_of(rec):
    return config.cost_usd(rec["model_id"], rec.get("input_tokens", 0), rec.get("output_tokens", 0))


def percentile(values, p):
    """Nearest-rank percentile (simple and good enough for small samples)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def min_max_score(value, low, high, log_scale=False):
    """1.0 for the best (lowest) value, 0.0 for the worst, in between by position. One model -> 1.0."""
    if high == low:
        return 1.0
    if log_scale:   # cost spans orders of magnitude; a log scale keeps cheap models distinguishable
        value, low, high = (math.log(max(x, 1e-9)) for x in (value, low, high))
    return 1.0 - (value - low) / (high - low)


def summarize(records, weights=None, models=None, categories=None):
    """Compute everything the report needs. `models`/`categories` restrict what is included."""
    weights = normalize_weights(weights or config.WEIGHTS)
    rows = [r for r in records
            if (not models or r["model"] in models) and (not categories or r["category"] in categories)]
    model_names = sorted({r["model"] for r in rows})
    category_names = sorted({r["category"] for r in rows})
    group_of = {r["category"]: r["group"] for r in rows}

    per_model = {}
    cat_scores = {}   # model -> category -> quality
    for name in model_names:
        mine = [r for r in rows if r["model"] == name]
        by_cat = {}
        for r in mine:
            q = quality_of(r)
            if q is not None:
                by_cat.setdefault(r["category"], []).append(q)
        cat_scores[name] = {c: mean(v) for c, v in by_cat.items()}
        cat_means = list(cat_scores[name].values())
        quality = mean(cat_means)
        # 95% interval from the spread of category means: a rough "how much to trust the third decimal"
        ci = 1.96 * statistics.stdev(cat_means) / math.sqrt(len(cat_means)) if len(cat_means) > 1 else 0.0
        billed = [r for r in mine if r.get("error_kind") != "api"]
        good = [r for r in mine if r["ok"]]
        latencies = [r["latency_s"] for r in good]
        per_model[name] = {
            "n_calls": len(mine), "n_scored": sum(len(v) for v in by_cat.values()),
            "quality": quality, "quality_ci": ci,
            "cost_per_call": mean([cost_of(r) for r in billed]),
            "total_cost": sum(cost_of(r) for r in billed),
            "latency_median": statistics.median(latencies) if latencies else 0.0,
            "latency_p95": percentile(latencies, 95),
            "error_rate": (len(mine) - len(good)) / len(mine) if mine else 0.0,
            "truncated": sum(1 for r in mine if r.get("stop_reason") == "max_tokens"),
            "self_judged": sum(1 for r in mine if r.get("judge") and r["judge"].get("self_judged")),
            "output_tokens": sum(r.get("output_tokens", 0) for r in mine),
        }

    # normalize cost and latency across the models being compared, then combine
    if per_model:
        costs = [m["cost_per_call"] for m in per_model.values()]
        lats = [m["latency_median"] for m in per_model.values()]
        for m in per_model.values():
            m["cost_score"] = min_max_score(m["cost_per_call"], min(costs), max(costs), log_scale=True)
            m["latency_score"] = min_max_score(m["latency_median"], min(lats), max(lats))
            m["aggregate"] = (weights["quality"] * m["quality"] + weights["cost"] * m["cost_score"]
                              + weights["latency"] * m["latency_score"])

    # group scores: mean of the category means inside each group
    group_scores = {}
    for name in model_names:
        groups = {}
        for cat, q in cat_scores[name].items():
            groups.setdefault(group_of[cat], []).append(q)
        group_scores[name] = {g: mean(v) for g, v in groups.items()}

    ranking = sorted(per_model, key=lambda n: per_model[n]["aggregate"], reverse=True)
    return {"weights": weights, "models": per_model, "categories": category_names, "group_of": group_of,
            "category_scores": cat_scores, "group_scores": group_scores, "ranking": ranking}


def category_winners(summary, tie_margin=None):
    """Per category: the highest quality wins; models within the tie margin are tied and the CHEAPER one wins."""
    margin = config.CATEGORY_TIE_MARGIN if tie_margin is None else tie_margin
    winners = {}
    for cat in summary["categories"]:
        scores = {m: s[cat] for m, s in summary["category_scores"].items() if cat in s}
        if not scores:
            continue
        best = max(scores.values())
        contenders = [m for m, q in scores.items() if best - q <= margin]
        winner = min(contenders, key=lambda m: (summary["models"][m]["cost_per_call"], m))
        winners[cat] = {"winner": winner, "quality": scores[winner], "best_quality": best, "tied": sorted(contenders)}
    return winners


def judge_agreement(records):
    """How much do two judges agree? Looks at answers scored by BOTH judges (needs a second judge).

    Returns None if no answer was double-judged, else a dict with the count, the share of identical
    scores, the mean absolute difference (in points of the 1-5 scale) and each judge's mean score.
    """
    pairs = []
    for r in records:
        scores = {j: v["score"] for j, v in ((r.get("judge") or {}).get("scores", {})).items() if v.get("score") is not None}
        if len(scores) >= 2:
            pairs.append(scores)
    if not pairs:
        return None
    judges = sorted(pairs[0])
    first, second = judges[0], judges[1]
    pairs = [p for p in pairs if first in p and second in p]
    diffs = [abs(p[first] - p[second]) for p in pairs]
    return {"n": len(pairs), "judges": [first, second],
            "exact": sum(1 for d in diffs if d == 0) / len(pairs), "mean_abs_diff": mean(diffs),
            "mean_scores": {first: mean([p[first] for p in pairs]), second: mean([p[second] for p in pairs])}}
