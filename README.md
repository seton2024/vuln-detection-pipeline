# AI-Assisted Vulnerability Detection Pipeline

Multi-stage cascading pipeline for detecting security vulnerabilities in Python code.

## Architecture

```
New Python file

Stage 0: Bandit (static analysis — free, instant): sets bandit_flag; always runs
Stage 1: CodeBERT / CNN-BiLSTM (fine-tuned on a frozen GraphCodeBERT encoder)
Stage 1.5: Consolidation (merge the flagged windows into the precise section)
Stage 2: Local Llama via Ollama (runs locally)
Stage 3: Claude Haiku via Batch API (paid, opt-in)
```

The cascade design means expensive AI calls are only made for genuinely ambiguous cases, keeping per-scan costs low.

## Seven vulnerability types

1. `sql`
2. `xss`
3. `command_injection`
4. `xsrf`
5. `path_disclosure`
6. `open_redirect`
7. `remote_code_execution`

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) configure secrets/toggles — copy the example and edit it
#    Windows:      copy env.example .env
#    macOS/Linux:  cp env.example .env

# 3. Sanity-check the setup
python scripts/verify_setup.py

# 4. Quick demo on the bundled vulnerable Flask app (Stage 0 + Stage 1)
python scripts/demo.py --vuln_type sql --file input_data/flask_app.py --backend cnn_bilstm

# 5. Run the full cascade (Stage 0 -> 3) on a file
python scripts/run_cascade.py --all --file input_data/flask_app.py --flagged-only
```

### Reproduce training from scratch

```bash
# Download VUDENC dataset (needed for training, not for inference)
python scripts/setup_data.py

# Train the Stage 1 models (writes models/cnn/cnn_bilstm_<type>.pt)
python scripts/train_stage1.py
```

## Configuration

Behaviour is controlled by environment variables (see `env.example`):

| Variable            | Default            | Meaning                                             |
|---------------------|--------------------|-----------------------------------------------------|
| `OLLAMA_MOCK`       | `1` (on)           | Set `0` to call Ollama for real in Stage 2          |
| `OLLAMA_MODEL`      | `qwen2.5-coder:7b` | Any model you've pulled (`ollama list` to check)    |
| `STAGE3_ENABLED`    | `0` (off)          | Set `1` to enable the Claude Stage 3 adjudicator    |
| `ANTHROPIC_API_KEY` | —                  | Your Anthropic key (only if Stage 3 enabled)        |

## Dataset

**VUDENC** — Wartschinski et al., 2022, Humboldt-Universität zu Berlin
https://github.com/LauraWartschinski/VulnerabilityDetection
