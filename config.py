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

#for Llama claude prompts, so they read naturally
VULN_TYPE_NAMES = {
    "sql": "SQL injection",
    "xss": "cross-site scripting (XSS)",
    "command_injection": "command injection",
    "xsrf": "cross-site request forgery (CSRF/XSRF)",
    "path_disclosure": "path disclosure",
    "open_redirect": "open redirect",
    "remote_code_execution": "remote code execution",
}

# ESCALATIONS

# If Stage 1 (CodeBERT) scores above this, the code is suspicious enough to pass to Stage 2.
STAGE1_ESCALATION_THRESHOLD = 0.5

# If Stage 2 (Llama) scores BELOW this, the code is safe - stop here, don't call Claude.
STAGE2_SAFE_THRESHOLD = 0.5

# If Stage 2 scores ABOVE this, the code is definitely vulnerable — stop here, don't call Claude.
# Between 0.5 and 0.9 = uncertain => escalate to Stage 3 (Claude Haiku).
STAGE2_ESCALATION_THRESHOLD = 0.9


# Stage 1 model (the GraphCodeBERT checkpoint used both as a classifier and as a
# frozen feature extractor for the CNN-BiLSTM head).
STAGE1_MODEL = "microsoft/graphcodebert-base"

# Which Stage 1 backend predict() uses: "graphcodebert" or "cnn_bilstm".
STAGE1_BACKEND = "cnn_bilstm"

# CNN-BiLSTM head hyperparameters, overwritten for each vuln type
STAGE1_HIDDEN_DIM = 128     # BiLSTM hidden size (per direction)
STAGE1_CNN_FILTERS = 64     # filters per convolution width (2,3,4,5)
STAGE1_DROPOUT = 0.3        # dropout before the final linear layer

# Max chars of code sent to Llama (Stage 2) — a center slice of the consolidated window.
STAGE2_WINDOW_CHARS = 300

#Stage 2 models
OLLAMA_MODEL = "qwen2.5-coder:7b"
STAGE2_WINDOW_CHARS = 300
OLLAMA_MOCK = True # If True, Stage 2 returns a fake score instead of calling Ollama, for testing without a model.


#Stage 3 models
#STAGE3_ENABLED = os.environ.get("STAGE3_ENABLED", "0").strip() in ("1", "true", "True", "yes")
STAGE3_ENABLED = True
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 1024
CLAUDE_API_KEY_ENV = "ANTHROPIC_API_KEY"  # set this env var to your key — never hardcode keys in source

# Paths
VUDENC_DATA_DIR = "data/vudenc"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Model weights are split into per-backend subfolders of MODELS_DIR:
#   models/cnn/cnn_bilstm_{type}.pt   and   models/graphcodebert/graphcodebert_{type}.pt
MODEL_SUBDIR = {"cnn_bilstm": "cnn", "graphcodebert": "graphcodebert"}

# Results are split by the stage they belong to (plus a folder for demo images).
STAGE1_RESULTS_DIR = os.path.join(RESULTS_DIR, "stage1")   # Stage 1 model search/eval
STAGE2_RESULTS_DIR = os.path.join(RESULTS_DIR, "stage2")   # Stage 2 (Llama)
STAGE3_RESULTS_DIR = os.path.join(RESULTS_DIR, "stage3")   # Stage 3 (Claude)
DEMO_RESULTS_DIR = os.path.join(RESULTS_DIR, "m_demo")     # demo PNGs

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# VUDENC context-block extraction parameters
VUDENC_BLOCK_LENGTH = 200
VUDENC_BLOCK_STEP = 5
