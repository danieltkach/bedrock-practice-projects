"""Build ONE self-contained interactive HTML file: data embedded as JSON, all JS/CSS/SVG inline.

No CDN, no external fetch, no extra dependency: the file works offline and can be e-mailed.
The page's JavaScript lives in dashboard_template.html; this module only injects the data.
"""
import json
import os

import config
import scoring
import testdata

TEMPLATE = os.path.join(config.BASE_DIR, "dashboard_template.html")
MAX_TEXT = 3500   # response/prompt characters embedded per row (keeps the file small)


def _row(rec, prompts):
    quality = scoring.quality_of(rec)
    judge = rec.get("judge")
    return {
        "model": rec["model"], "category": rec["category"], "group": rec["group"], "test": rec["test_id"],
        "quality": quality, "cost": scoring.cost_of(rec), "latency": rec["latency_s"], "ok": rec["ok"],
        "errorKind": rec.get("error_kind", ""), "error": rec.get("error", ""), "cached": rec["cached"],
        "stop": rec.get("stop_reason", ""), "tin": rec.get("input_tokens", 0), "tout": rec.get("output_tokens", 0),
        "auto": rec["auto"]["score"] if rec.get("auto") else None,
        "checks": rec["auto"]["checks"] if rec.get("auto") else [],
        "judge": judge["mean"] if judge else None,
        "judgeDetail": {j: {"score": v.get("score"), "reason": v.get("reason", "")} for j, v in judge["scores"].items()} if judge else {},
        "selfJudged": bool(judge and judge.get("self_judged")),
        "prompt": prompts.get(rec["test_id"], "")[:MAX_TEXT],
        "response": rec.get("response", "")[:MAX_TEXT],
    }


def build_dashboard(records, meta, weights=None):
    prompts = {t["id"]: t["prompt"] for t in testdata.all_tests(testdata.load_categories())}
    payload = {
        "meta": {"finishedAt": meta.get("finished_at", ""), "spentUsd": meta.get("spent_usd", 0),
                 "judges": [config.MODELS[j]["label"] for j in meta.get("judges", []) if j in config.MODELS],
                 "mock": bool(meta.get("mock"))},
        "weights": scoring.normalize_weights(weights or config.WEIGHTS),
        "rules": {"valueRatio": config.VALUE_RATIO, "fastRatio": config.FAST_RATIO,
                  "tieMargin": config.CATEGORY_TIE_MARGIN},
        "models": {n: {"label": s["label"], "id": s["id"]} for n, s in config.MODELS.items()},
        "rows": [_row(r, prompts) for r in records],
    }
    # "</" inside a <script> block would end it early; escape it. Also escape line separators.
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    return html.replace("/*__DATA__*/null", blob)
