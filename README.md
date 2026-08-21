# HarmEvoSkill: Skill-Memory Self-Evolution for Zero-shot Multimodal Harmful Meme Detection

[![EMNLP 2026 Findings](https://img.shields.io/badge/EMNLP%202026-Findings-blue)]()

This is the official repository for EMNLP 2026 Findings paper **HarmEvoSkill: Skill-Memory Self-Evolution for Zero-shot Multimodal Harmful Meme Detection**.

<p align="center">
  <img src="assets/framework_overview.png" alt="Overview of the Skill-Memory framework" width="95%">
</p>


## News

- `2026/08` 🎉 HarmEvoSkill is accepted to **EMNLP 2026 Findings**!
- `2026/05` 🚀 We release the code of HarmEvoSkill!

## Overview

HarmEvoSkill introduces a Skill-Memory self-evolution framework that learns reusable reasoning experience from unlabeled multimodal memes.

The framework contains three main components:

1. **Label-Free Skill-Memory Construction**
   
   Build reusable Case Memory, Insight Memory, and Risk Memory from unlabeled reference memes.

2. **Memory-Guided Skill Planning**

   Retrieve relevant experience, select executable reasoning skills, and generate structured evidence for each target meme.

3. **Claim Graph Reasoning and Memory Evolution**

   Organize evidence into claim-evidence graphs for final judgement and update memory through validated proposals.

## Repository Structure

```
.
├── framework/
│   ├── run_framework.py        # Main experiment entry
│   ├── pipeline.py             # End-to-end pipeline
│   ├── memory.py               # Case/Insight/Risk Memory
│   ├── signature.py             # Sample signature extraction
│   ├── planner.py               # Skill planning
│   ├── skills.py                # Skill card generation
│   ├── evidence_graph.py        # Claim-evidence graph reasoning
│   ├── judge.py                 # Final judgement and validation
│   └── config.py                # Model configuration
│
├── scripts/
│   └── prepare_datasets.py      # Dataset preparation
│
├── data/                        # Dataset directory
├── results/                     # Prediction outputs
├── utils/                       # Prompts and utilities
├── requirements.txt
└── README.md
```

## Installation

1. Create a Python environment:

```bash
python -m venv skill_memory_env
source skill_memory_env/bin/activate
pip install -r requirements.txt
```

2. Set your API key before running experiments:

```bash
export OPENAI_API_KEY="<your-api-key>"
```

3. Model settings can be modified in:

```
framework/config.py
```

## Dataset Preparation

HarmEvoSkill is evaluated on the following harmful meme benchmarks:

- **FHM**: Facebook Hateful Memes
- **Harm-C**: Harmful meme dataset
- **Harm-P**: Political harmful meme dataset
- **MultiOFF**: Multimodal offensive meme dataset
- **PrideMM**: Multimodal harmful meme dataset

Due to dataset licenses, raw images are not redistributed.

Please download the datasets from their official sources and organize them as:

```
data/
└── raw/
    ├── FHM/
    ├── Harm-C/
    ├── Harm-P/
    ├── MultiOFF/
    └── PrideMM/
```

## Quick Start

1. Build Skill Memory

Construct label-free Skill Memory from the reference split:
```
python framework/run_framework.py \
  --mode build_memory \
  --dataset FHM \
  --build_with_llm
```
Freeze the memory for reproducible evaluation:
```
python framework/run_framework.py \
  --mode freeze_base_memory \
  --dataset FHM \
  --build_with_llm
```
2. Run Evaluation

Run HarmEvoSkill on a benchmark:
```
python framework/run_framework.py \
  --mode main \
  --dataset FHM \
  --model gpt-4o \
  --memory_update_policy proposal_only \
  --output_path results/FHM/main.jsonl
```
