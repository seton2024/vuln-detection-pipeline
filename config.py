"""
Central configuration, no hardcoding!
"""

import os


VULN_TYPES = [
    "sql",
    "xss",
    "command_injection",
    "xsrf",
    "path_disclosure",
    "open_redirect",
    "remote_code_execution",
]

# Stage escalation thresholds
# If Stage 1 (CodeBERT) scores above this, the code is suspicious enough to pass to Stage 2.
STAGE1_ESCALATION_THRESHOLD = 0.5

# If Stage 2 (Llama) scores BELOW this, the code is safe - stop here, don't call Claude.
STAGE2_SAFE_THRESHOLD = 0.5

# If Stage 2 scores ABOVE this, the code is definitely vulnerable — stop here, don't call Claude.
# Between 0.5 and 0.9 = uncertain => escalate to Stage 3 (Claude Haiku).
STAGE2_ESCALATION_THRESHOLD = 0.9

# Stage 3 (Claude API) costs money per call. Default OFF. Turn on with: STAGE3_ENABLED=1
STAGE3_ENABLED = os.getenv("STAGE3_ENABLED", "0") == "1"

# Stage 1 model
STAGE1_MODEL = "microsoft/graphcodebert-base"

# Max chars of code sent to Llama (Stage 2) — a center slice of the consolidated window.
STAGE2_WINDOW_CHARS = 300

# Paths
VUDENC_DATA_DIR = "data/vudenc"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# VUDENC context-block extraction parameters
VUDENC_BLOCK_LENGTH = 200
VUDENC_BLOCK_STEP = 5
