"""Tiny on-disk response cache: same (model, system, prompt, params) -> same stored answer.

Why: benchmark calls cost real money. With the cache a re-run (or a re-score) is free, and the
stored latency/tokens make the report reproducible. Only successful answers are cached.
"""
import hashlib
import json
import os


def make_key(model_id, system, prompt, max_tokens, temperature):
    payload = json.dumps([model_id, system or "", prompt, max_tokens, temperature], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory):
        self.directory = directory

    def _path(self, key):
        # two-letter sub-folders keep any single folder small
        return os.path.join(self.directory, key[:2], key + ".json")

    def get(self, key):
        try:
            with open(self._path(key), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None   # missing or corrupted file = cache miss

    def put(self, key, entry):
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entry, f)
        os.replace(tmp, path)   # atomic: a crash never leaves a half-written cache file
