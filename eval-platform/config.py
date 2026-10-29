"""All the knobs of the project in one place. Edit here, not in the logic files."""
import os

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# --- Models under test ------------------------------------------------------------------
# short name -> model id (inference profile), label, and ESTIMATED price in USD per 1M tokens.
# Prices are hand-typed ESTIMATES. Bedrock prices change: verify them at
# https://aws.amazon.com/bedrock/pricing/ before trusting any dollar figure.
# "default": True  -> included when --models is not given.  False -> opt-in (expensive).
# NOTE: Claude 5.x models "think" first and can spend a small maxTokens budget with no text
# block, so they are deliberately not in the list.
MODELS = {
    "sonnet-4.5": {"id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "label": "Claude Sonnet 4.5",
                   "price_in": 3.00, "price_out": 15.00, "default": True},
    "haiku-4.5": {"id": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "label": "Claude Haiku 4.5",
                  "price_in": 1.00, "price_out": 5.00, "default": True},
    "nova-pro": {"id": "us.amazon.nova-pro-v1:0", "label": "Amazon Nova Pro",
                 "price_in": 0.80, "price_out": 3.20, "default": True},
    "nova-2-lite": {"id": "us.amazon.nova-2-lite-v1:0", "label": "Amazon Nova 2 Lite",
                    "price_in": 0.30, "price_out": 2.50, "default": True},
    "nova-lite": {"id": "us.amazon.nova-lite-v1:0", "label": "Amazon Nova Lite",
                  "price_in": 0.06, "price_out": 0.24, "default": True},
    "nova-micro": {"id": "us.amazon.nova-micro-v1:0", "label": "Amazon Nova Micro",
                   "price_in": 0.035, "price_out": 0.14, "default": True},
    "opus-4.6": {"id": "us.anthropic.claude-opus-4-6-v1", "label": "Claude Opus 4.6",
                 "price_in": 5.00, "price_out": 25.00, "default": False},
}

# --- Judge (LLM-as-judge) ---------------------------------------------------------------
JUDGE_MODEL = "sonnet-4.5"     # primary judge (short name from MODELS)
SECOND_JUDGE_MODEL = None      # optional: e.g. "haiku-4.5" (CLI: --judge2 haiku-4.5)
JUDGE_MAX_TOKENS = 300
JUDGE_MAX_RESPONSE_CHARS = 3000   # the candidate answer is cut to this length before judging

# --- How one answer becomes a quality number (0..1) -------------------------------------
# A test with BOTH automatic checks and a judge score mixes them with this weight.
AUTO_WEIGHT = 0.5              # judge weight is 1 - AUTO_WEIGHT

# --- Aggregate score weights (normalized to sum to 1; CLI: --w-quality --w-cost --w-latency)
WEIGHTS = {"quality": 0.6, "cost": 0.2, "latency": 0.2}

# --- Recommendation rules (see recommend.py) --------------------------------------------
VALUE_RATIO = 0.90        # "best value" = cheapest model with quality >= 90% of the best quality
FAST_RATIO = 0.85         # "fastest acceptable" = fastest model with quality >= 85% of the best
CATEGORY_TIE_MARGIN = 0.02   # per-category: models within 0.02 of the best are tied -> cheaper wins
MAX_ERROR_RATE = 0.05     # models with more failed calls than this get a warning

# Category groups used for the "best for code / reasoning / ..." recommendations.
# A category file in testdata/ declares its own "group".
GROUP_TITLES = {
    "code": "code (generation, debugging, SQL, regex)",
    "reasoning": "reasoning and math",
    "language": "writing and language (summaries, translation, tone, legal text)",
    "structured": "structured output and instruction following",
    "knowledge": "knowledge, context and common sense",
    "safety": "safety and honesty (refusals, hallucination)",
}

# --- Running the benchmark --------------------------------------------------------------
BUDGET_USD = 3.00             # hard cap on ESTIMATED spend; the run aborts before exceeding it
MAX_WORKERS = 8               # threads in total
PER_MODEL_CONCURRENCY = 2     # at most this many simultaneous calls to ONE model (avoids throttling)
DEFAULT_MAX_TOKENS = 700      # output budget of a test call unless the test sets "max_tokens"
TEMPERATURE = 0.0             # deterministic-ish answers make comparisons fairer
EXPECTED_OUTPUT_TOKENS = 250  # only used by --dry-run to estimate cost
JUDGE_EXPECTED_INPUT_TOKENS = 450   # dry-run estimate of one judge prompt, plus the answer
ENABLE_CODE_EXEC = True       # run generated Python/regex in a restricted subprocess (CLI: --no-exec)
CODE_TIMEOUT_SECONDS = 5

# --- Retry with exponential backoff (for throttling and transient errors) ---------------
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 20.0
RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
    "ModelNotReadyException",
}

# --- Paths ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(BASE_DIR, "testdata")
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "responses")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "results")
MOCK_OUT_DIR = os.path.join(BASE_DIR, "demo_output")
LOG_FILE = os.path.join(BASE_DIR, "eval-platform.log")


def resolve_models(names):
    """Turn a list of short names (or None/'all') into [(short_name, spec), ...]."""
    if not names:
        return [(k, v) for k, v in MODELS.items() if v["default"]]
    if names == ["all"]:
        return list(MODELS.items())
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    return [(n, MODELS[n]) for n in names]


def price_of(model_id):
    """(input, output) USD per million tokens, or None for an unknown model id."""
    for spec in MODELS.values():
        if spec["id"] == model_id:
            return spec["price_in"], spec["price_out"]
    return None


def cost_usd(model_id, input_tokens, output_tokens):
    prices = price_of(model_id)
    if prices is None:
        return 0.0
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def short_name(model_id):
    for name, spec in MODELS.items():
        if spec["id"] == model_id:
            return name
    return model_id
