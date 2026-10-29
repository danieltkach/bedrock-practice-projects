"""Automatic recommendations with TRANSPARENT rules: every line says which rule picked the model and why.

No LLM is involved here, so the result is reproducible and explainable. (report.py can optionally add an
LLM-written narrative on top, clearly marked as generated.)
"""
import config


def _label(name):
    return config.MODELS.get(name, {}).get("label", name)


def _money(x):
    return f"${x:.5f}" if x < 0.01 else f"${x:.4f}"


def recommend(summary):
    """Return {"recommendations": [{title, model, rule, evidence}], "warnings": [str]}."""
    models = summary["models"]
    if not models:
        return {"recommendations": [], "warnings": ["no results to analyse"]}
    w = summary["weights"]
    best_q = max(m["quality"] for m in models.values())
    recs = []

    # 1. best overall = highest weighted aggregate score
    top = summary["ranking"][0]
    m = models[top]
    recs.append({"title": "Best overall", "model": top,
                 "rule": f"highest aggregate score with weights quality {w['quality']:.2f}, cost {w['cost']:.2f}, latency {w['latency']:.2f}",
                 "evidence": f"aggregate {m['aggregate']:.3f}, quality {m['quality']:.3f}, {_money(m['cost_per_call'])} per call, median latency {m['latency_median']:.2f}s"})

    # 2. highest quality, ignoring cost and speed
    q_top = max(models, key=lambda n: (models[n]["quality"], -models[n]["cost_per_call"]))
    m = models[q_top]
    recs.append({"title": "Highest quality", "model": q_top, "rule": "highest mean quality (cost and latency ignored)",
                 "evidence": f"quality {m['quality']:.3f} (+/- {m['quality_ci']:.3f}), {_money(m['cost_per_call'])} per call"})

    # 3. best value = cheapest model that is "good enough" (>= VALUE_RATIO of the best quality)
    ok_value = [n for n in models if models[n]["quality"] >= config.VALUE_RATIO * best_q]
    value = min(ok_value, key=lambda n: models[n]["cost_per_call"])
    m = models[value]
    recs.append({"title": "Best value", "model": value,
                 "rule": f"cheapest model with quality >= {config.VALUE_RATIO:.0%} of the best quality ({config.VALUE_RATIO * best_q:.3f})",
                 "evidence": f"quality {m['quality']:.3f}, {_money(m['cost_per_call'])} per call"})

    # 4. fastest acceptable
    ok_fast = [n for n in models if models[n]["quality"] >= config.FAST_RATIO * best_q]
    fast = min(ok_fast, key=lambda n: models[n]["latency_median"])
    m = models[fast]
    recs.append({"title": "Fastest acceptable", "model": fast,
                 "rule": f"lowest median latency among models with quality >= {config.FAST_RATIO:.0%} of the best ({config.FAST_RATIO * best_q:.3f})",
                 "evidence": f"median latency {m['latency_median']:.2f}s, quality {m['quality']:.3f}"})

    # 5. best per category group (code, reasoning, ...): highest group quality, ties go to the cheaper model
    groups = sorted({g for scores in summary["group_scores"].values() for g in scores})
    for group in groups:
        scores = {n: s[group] for n, s in summary["group_scores"].items() if group in s}
        best = max(scores.values())
        contenders = [n for n, q in scores.items() if best - q <= config.CATEGORY_TIE_MARGIN]
        pick = min(contenders, key=lambda n: (models[n]["cost_per_call"], n))
        tie_note = f"; tied within {config.CATEGORY_TIE_MARGIN} with {', '.join(_label(n) for n in contenders if n != pick)}" if len(contenders) > 1 else ""
        recs.append({"title": f"Best for {config.GROUP_TITLES.get(group, group)}", "model": pick,
                     "rule": f"highest mean quality across the '{group}' categories; near-ties go to the cheaper model",
                     "evidence": f"group quality {scores[pick]:.3f}{tie_note}"})

    # warnings: things a reader must know before trusting the picture
    warnings = []
    ranked = summary["ranking"]
    if len(ranked) > 1:
        a, b = models[ranked[0]], models[ranked[1]]
        if a["aggregate"] - b["aggregate"] < 0.02:
            warnings.append(f"The top two models ({_label(ranked[0])}, {_label(ranked[1])}) differ by less than 0.02 in aggregate score: treat them as tied.")
        if abs(a["quality"] - b["quality"]) < a["quality_ci"] + b["quality_ci"]:
            warnings.append("Quality intervals of the top two models overlap: with this sample size their quality difference is not statistically meaningful.")
    for n, m in models.items():
        if m["error_rate"] > config.MAX_ERROR_RATE:
            warnings.append(f"{_label(n)} failed {m['error_rate']:.0%} of its calls (errors or empty answers).")
        if m["truncated"]:
            warnings.append(f"{_label(n)} had {m['truncated']} answer(s) cut off by the token limit.")
        if m["self_judged"]:
            warnings.append(f"{_label(n)} had {m['self_judged']} answer(s) judged by itself (only judge available): self-preference bias possible.")
    return {"recommendations": recs, "warnings": warnings}
