"""Load and validate the test suite (plain JSON files in testdata/, one file per category).

File format:
  {"category": "name", "group": "code", "description": "...",
   "tests": [{"id": "name-1", "prompt": "...", "system": "(optional)",
              "checks": [{"type": "contains", "values": ["x"]}],     # automatic checks (optional)
              "rubric": "what a good answer must do",               # needed when judge is true
              "judge": true,                                        # default true
              "reference": "an example of a good answer",           # recommended
              "max_tokens": 300}]}                                  # optional output budget
"""
import glob
import json
import os

import checks
import config


def load_categories(directory=None, only=None, limit=None):
    """Return a list of category dicts sorted by name.

    only:  optional list of category names to keep.   limit: max tests per category.
    """
    directory = directory or config.TESTDATA_DIR
    categories = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if only and data["category"] not in only:
            continue
        for test in data["tests"]:
            test["category"] = data["category"]
            test["group"] = data.get("group", "other")
        if limit:
            data["tests"] = data["tests"][:limit]
        categories.append(data)
    return categories


def all_tests(categories):
    return [t for c in categories for t in c["tests"]]


def validate(categories):
    """Return a list of problem strings (empty list = data is fine)."""
    problems, seen = [], set()
    for cat in categories:
        if not cat["tests"]:
            problems.append(f"{cat['category']}: no tests")
        if cat.get("group", "other") not in config.GROUP_TITLES:
            problems.append(f"{cat['category']}: unknown group '{cat.get('group')}'")
        for t in cat["tests"]:
            tid = t.get("id", "?")
            if tid in seen:
                problems.append(f"{tid}: duplicate id")
            seen.add(tid)
            if not t.get("prompt"):
                problems.append(f"{tid}: missing prompt")
            if not t.get("checks") and not t.get("judge", True):
                problems.append(f"{tid}: no checks and judge=false, nothing would score it")
            if t.get("judge", True) and not t.get("rubric"):
                problems.append(f"{tid}: judge=true needs a rubric")
            for spec in t.get("checks", []):
                if spec.get("type") not in checks.CHECKS:
                    problems.append(f"{tid}: unknown check type '{spec.get('type')}'")
    return problems


def validate_references(categories):
    """A test's own reference answer must pass its own automatic checks. Catches typos in the data."""
    problems = []
    for t in all_tests(categories):
        if not t.get("checks"):
            continue
        if not t.get("reference"):
            problems.append(f"{t['id']}: has checks but no reference answer")
            continue
        result = checks.run_checks(t["checks"], t["reference"])
        if result["score"] is not None and result["score"] < 0.999:
            bad = [c for c in result["checks"] if c["score"] is not None and c["score"] < 1]
            problems.append(f"{t['id']}: reference fails its own checks: {bad}")
    return problems
