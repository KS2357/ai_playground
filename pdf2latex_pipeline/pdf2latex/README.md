# PDF → LaTeX Transformer Pipeline

A full training + RLHF-style fine-tuning pipeline for a vision-language transformer that converts PDF page images into LaTeX source code.

## Architecture Overview

```
pdf2latex/
├── configs/                  # YAML configs for training & finetuning
│   ├── train_config.yaml
│   └── finetune_config.yaml
├── data/
│   ├── dataset.py            # Dataset classes (PDF → image, paired with .tex)
│   └── preprocessing.py      # PDF rendering, tokenization helpers
├── models/
│   ├── encoder.py            # Vision encoder (ViT-style patch embeddings)
│   ├── decoder.py            # Autoregressive LaTeX decoder
│   ├── model.py              # Full PDF2LaTeX model
│   └── reward_model.py       # Reward model for RLHF fine-tuning
├── training/
│   ├── trainer.py            # Supervised pre-training loop
│   └── metrics.py            # BLEU, edit-distance, compilation success
├── finetuning/
│   ├── rollout.py            # Token-level rollout / sampling
│   ├── ppo_trainer.py        # PPO fine-tuning loop
│   ├── grpo_trainer.py       # GRPO (Group Relative Policy Optimization)
│   └── reward_functions.py   # Compilation, similarity, and format rewards
└── utils/
    ├── logging.py
    └── checkpoint.py
```

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision transformers datasets pdfplumber pymupdf \
            Pillow tqdm pyyaml wandb nltk editdistance accelerate
# For LaTeX compilation reward:
apt-get install texlive-full  # or tectonic
```

### 2. Pre-train on supervised data
```bash
python training/trainer.py --config configs/train_config.yaml
```

### 3. Fine-tune with PPO / GRPO rollouts
```bash
# PPO
python finetuning/ppo_trainer.py --config configs/finetune_config.yaml

# GRPO (recommended - no separate critic needed)
python finetuning/grpo_trainer.py --config configs/finetune_config.yaml
```

## Data Format

Pair every PDF page with its LaTeX source file:
```
data/
  train/
    sample_001.pdf   ←→   sample_001.tex
    sample_002.pdf   ←→   sample_002.tex
  val/
    ...
```

If you have multi-page PDFs, the dataset class automatically splits by page
and aligns to corresponding `\begin{document}...\end{document}` sections.

## Fine-tuning Strategy

Two approaches are implemented:

| Method | Key idea | Best for |
|--------|----------|----------|
| **PPO** | Actor-Critic with clipped surrogate objective | Large models, stable training |
| **GRPO** | Group sampling, no critic needed | Smaller models, faster iteration |

Both use a **hybrid reward**:
- `compilation_reward`: Does `pdflatex` compile without errors? (+1 / 0)
- `similarity_reward`: Normalised edit-distance vs. reference LaTeX
- `format_reward`: Structural checks (matching braces, environments, etc.)
