# HarmEvoSkill: Skill-Memory Self-Evolution for Zero-shot Multimodal Harmful Meme Detection

This repository contains the code for the paper *HarmEvoSkill: Skill-Memory Self-Evolution for Zero-shot Multimodal Harmful Meme Detection*. 

<p align="center">
  <img src="assets/framework_overview.png" alt="Overview of the Skill-Memory framework" width="95%">
</p>

## Overview

The framework decomposes multimodal harmful content detection into three stages:

1. **Label-free Skill-Memory Construction**: construct reusable memories and task skills from an unlabeled or training/reference split.
2. **Memory-guided Skill Planning and Execution**: retrieve related memory, select task-specific skills, and generate an evidence packet for each target meme.
3. **Graph-assisted Judgement and Self-evolving Memory Update**: organize evidence into a claim graph, predict whether the meme is harmful, and optionally generate memory update proposals.

## Keywords

multimodal NLP, harmful meme detection, memory-augmented agents, skill planning

## Repository Structure

```text
.
├── framework/                  # Core implementation of the Skill-Memory framework
│   ├── run_framework.py         # Main experiment CLI
│   ├── pipeline.py              # End-to-end pipeline
│   ├── direct_baseline.py       # Direct multimodal LLM baseline
│   ├── memory.py                # Case, insight, and risk memory modules
│   ├── signature.py             # Query and sample signature extraction
│   ├── planner.py               # Memory-guided skill planning
│   ├── skills.py                # Skill cards and evidence packet generation
│   ├── evidence_graph.py        # Claim-evidence graph construction
│   ├── judge.py                 # Graph-assisted judgement and update proposals
│   ├── boundary_rules.py        # Shared decision boundary guidance
│   └── config.py                # Model and API configuration
├── scripts/                     # Data preparation and utility scripts
│   └── prepare_datasets.py      # Dataset normalization script
├── data/                        # Dataset directory; restricted data are not redistributed
├── results/                     # Output directory for predictions and summaries
├── utils/                       # Prompt templates and supporting utilities
├── requirements.txt             # Python dependencies
└── README.md                    # Repository documentation
```

## Setup

Create a Python environment with Python 3.9 or newer.

Using `pip`:

```bash
python -m venv skill_memory_env
source skill_memory_env/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv skill_memory_env
.\skill_memory_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

If a Conda environment file is provided, the environment can also be created with:

```bash
conda env create -f environment.yml
conda activate skill-memory
```

## API Configuration

Set the API credentials before running LLM-based experiments.

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="<your-api-key>"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

Model aliases and default endpoints are configured in `framework/config.py`. If a custom endpoint uses a different model name, update the corresponding model alias before running experiments.

Do not commit API keys, endpoint secrets, local cache files, or raw outputs to the repository.

## Dataset Preparation

The experiments use the following benchmark datasets:

- `FHM`: Facebook Hateful Memes.
- `HarM`: harmful meme dataset; `HarM` may be treated as an alias of `Harm-C` in the code.
- `Harm-P`: politics-related harmful meme dataset.
- `MultiOFF`: multimodal offensive meme dataset.
- `PrideMM`: pride/LGBTQ-related multimodal meme dataset.

Due to dataset license restrictions, raw images and restricted annotations are not redistributed. Place the downloaded files under `data/raw/<DATASET>/` and then run the preparation script.

Expected layout:

```text
data/
├── raw/
│   ├── FHM/
│   │   ├── images/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   ├── Harm-C/
│   │   ├── images/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   ├── Harm-P/
│   │   ├── images/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   ├── MultiOFF/
│   │   ├── images/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   └── PrideMM/
│       ├── images/
│       ├── train.jsonl
│       └── test.jsonl
└── ...
```

## Quick Start

A deterministic smoke test can be used to check the environment and data paths without calling an external API:

```bash
python framework/run_framework.py \
  --mode main \
  --dataset FHM \
  --max_samples 5 \
  --deterministic \
  --disable_memory_update \
  --no_resume_output \
  --output_path results/smoke_fhm.jsonl
```

## Build Skill Memory

Build the label-free skill memory from the training or reference split:

```bash
python framework/run_framework.py --mode build_memory --dataset FHM --build_with_llm
python framework/run_framework.py --mode freeze_base_memory --dataset FHM --build_with_llm
```

Run memory construction for all datasets:

```bash
for d in FHM HarM Harm-P MultiOFF PrideMM; do
  python framework/run_framework.py --mode build_memory --dataset "$d" --build_with_llm
  python framework/run_framework.py --mode freeze_base_memory --dataset "$d" --build_with_llm
done
```

`freeze_base_memory` creates a clean read-only memory snapshot under:

```text
data/<DATASET>/skill_memory_base/
```

During evaluation, online update proposals are written to an isolated run directory by default. The base memory should remain unchanged for reproducible evaluation.

## Main Evaluation

Recommended single-dataset evaluation:

```bash
python framework/run_framework.py \
  --mode main \
  --dataset FHM \
  --model gpt-4o \
  --surface_retrieval_backend ssr \
  --memory_update_policy proposal_only \
  --disable_parallel_skills \
  --max_workers 1 \
  --output_path results/FHM/main_gpt4o.jsonl \
  --no_resume_output
```

If precomputed surface retrieval files are unavailable, disable surface retrieval:

```bash
python framework/run_framework.py \
  --mode main \
  --dataset FHM \
  --model gpt-4o \
  --disable_surface_retrieval \
  --memory_update_policy proposal_only \
  --disable_parallel_skills \
  --max_workers 1 \
  --output_path results/FHM/main_no_surface_gpt4o.jsonl \
  --no_resume_output
```

To resume an interrupted run, remove `--no_resume_output` and reuse the same `--output_path`.

## Output Files

Each JSONL output contains per-sample predictions and intermediate reasoning artifacts. Depending on the run mode, the output may include:

- predicted label and confidence;
- retrieved memory entries;
- selected skill cards;
- generated evidence packets;
- claim graph diagnostics;
- final judgement summary;
- optional memory update proposals.

Common output files:

```text
results/<DATASET>/main_<MODEL>.jsonl       # main framework predictions
results/<DATASET>/direct_<MODEL>.jsonl     # direct baseline predictions
results/<DATASET>/run_memory/              # isolated memory copy for one run
```
