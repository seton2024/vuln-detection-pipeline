# AI-Assisted Vulnerability Detection Pipeline

Multi-stage cascading pipeline for detecting security vulnerabilities in Python code.

## Architecture

```
New Python file

Stage 0: Bandit (static analysis — free, instant): Sets bandit_flag; always runs
Stage 1: CodeBERT (fine-tuned transformer model)
Stage 1.5: Consolidation (get the precise voulnerable section from different windows)
Stage 2: Local Llama via Ollama (runs localy)
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

# 2. Download VUDENC dataset (needed for training, not for inference)
python scripts/setup_data.py

# 3. train the models
python scripts/train_stage1.py

#4. Run the pypline
python.exe 

```


## Dataset

**VUDENC** — Wartschinski et al., 2022, Humboldt-Universität zu Berlin
https://github.com/LauraWartschinski/VulnerabilityDetection

<<<<<<< HEAD
=======

what to change today:

Think how to make the labled test data as the train data. how to wor without the bad parts
>>>>>>> origin/master
