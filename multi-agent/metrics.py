"""Tracks every model call (tokens, time, retries) and prints a cost/latency report."""
import threading
from dataclasses import dataclass

import config


@dataclass
class CallRecord:
    role: str            # coordinator, analysis, writing, ...
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_s: float     # wall time of the whole call, retries included
    attempts: int        # 1 = worked first time
    ok: bool = True


def price_of(model_id):
    """(input, output) USD per million tokens, or None when the model is not in the table."""
    return config.PRICES_PER_MILLION.get(model_id)


def cost_usd(model_id, input_tokens, output_tokens):
    prices = price_of(model_id)
    if prices is None:
        return 0.0
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


class Tracker:
    """Collects CallRecords. Several threads add to it at once, so it uses a lock."""

    def __init__(self):
        self.records = []
        self._lock = threading.Lock()

    def add(self, record):
        with self._lock:
            self.records.append(record)

    def reset(self):
        with self._lock:
            self.records = []

    # --- totals -------------------------------------------------------------------------
    def total_cost(self):
        return sum(cost_usd(r.model_id, r.input_tokens, r.output_tokens) for r in self.records)

    def baseline_cost(self):
        """What the same tokens would cost if every call used the baseline (strong) model."""
        tin = sum(r.input_tokens for r in self.records)
        tout = sum(r.output_tokens for r in self.records)
        return cost_usd(config.BASELINE_MODEL, tin, tout)

    def sum_latency(self):
        """Total time if the calls had run one after another."""
        return sum(r.latency_s for r in self.records)

    def report(self, wall_seconds):
        """Human-readable table. wall_seconds = real elapsed time of the whole query."""
        if not self.records:
            return "(no model calls)"
        lines = ["", "Cost / latency report (costs are ESTIMATES: check config.PRICES_PER_MILLION)"]
        lines.append(f"{'role':<12}{'model':<14}{'in tok':>8}{'out tok':>8}{'sec':>7}{'tries':>6}{'est $':>10}")
        for r in self.records:
            short = r.model_id.split(".")[-1].split("-v")[0][:13]
            c = cost_usd(r.model_id, r.input_tokens, r.output_tokens)
            flag = "" if r.ok else "  FAILED"
            lines.append(f"{r.role:<12}{short:<14}{r.input_tokens:>8}{r.output_tokens:>8}"
                         f"{r.latency_s:>7.2f}{r.attempts:>6}{c:>10.5f}{flag}")
        total, base = self.total_cost(), self.baseline_cost()
        saving = (1 - total / base) * 100 if base > 0 else 0.0
        lines.append(f"Total estimated cost: ${total:.5f}   "
                     f"(same tokens all on the strong model: ${base:.5f}, saving {saving:.0f}%)")
        seq = self.sum_latency()
        lines.append(f"Latency: {wall_seconds:.2f}s real vs {seq:.2f}s if calls ran one by one "
                     f"(parallelism saved {max(seq - wall_seconds, 0):.2f}s)")
        return "\n".join(lines)
