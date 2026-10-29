"""All the knobs of the project in one place. Edit here, not in the logic files."""
import os

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# --- Models (inference-profile IDs that work in this account) -------------------------
SONNET = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # strong, more expensive
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"    # fast and cheap
NOVA_LITE = "us.amazon.nova-lite-v1:0"                   # cheapest (not used by default)
# NOTE: avoid Claude 5.x here. Those models "think" first and can spend a small maxTokens
# budget without returning any text block.

# --- Which model each role uses (the cost/quality dial) --------------------------------
# Rule of thumb: planning and simple writing are easy -> cheap model.
# Deep reasoning, code and the final merge decide quality -> strong model.
ROLE_MODEL = {
    "coordinator": HAIKU,   # classify + plan: short structured output
    "direct": HAIKU,        # trivial questions
    "analysis": SONNET,
    "writing": HAIKU,       # drafting from material that is already worked out
    "code": SONNET,
    "aggregator": SONNET,   # merges everything: coherence matters most here
}

# Output budget per role (maxTokens). Keep them generous enough to finish a thought.
ROLE_MAX_TOKENS = {
    "coordinator": 700,
    "direct": 500,
    "analysis": 1500,
    "writing": 1000,
    "code": 2500,
    "aggregator": 3500,
}

# --- Orchestration limits ---------------------------------------------------------------
MAX_SUBTASKS = 5        # the plan is cut/rejected above this (protects cost)
MAX_PARALLEL = 4        # threads used for independent subtasks
HISTORY_TURNS = 3       # past Q/A pairs shown to the coordinator in chat mode
HISTORY_CHARS = 600     # each past answer is truncated to this many characters

# --- Retry with exponential backoff (for throttling and transient errors) ---------------
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0   # waits ~1s, 2s, 4s, 8s (plus a little random jitter)
BACKOFF_MAX_SECONDS = 20.0
RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
    "ModelNotReadyException",
}

# --- ESTIMATED prices in USD per 1 million tokens: (input, output) ----------------------
# These are ESTIMATES typed in by hand. Bedrock prices change: verify them at
# https://aws.amazon.com/bedrock/pricing/ before trusting any dollar figure.
PRICES_PER_MILLION = {
    SONNET: (3.00, 15.00),
    HAIKU: (1.00, 5.00),
    NOVA_LITE: (0.06, 0.24),
}
# The report compares the real mix of models against "everything on this model".
BASELINE_MODEL = SONNET

LOG_FILE = "multi-agent.log"
