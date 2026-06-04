# Learning Dynamics & Catastrophic Forgetting in Small Language Models

> A controlled study of catastrophic forgetting in small decoder-only Transformer LMs under sequential domain adaptation across three model scales and three training regimes.

---

## Overview

This project investigates whether small Transformer language models catastrophically forget previously learned domains when trained sequentially on a new one. We train three model scales (M1, M2, M3) across three experiments (E1: Stories→News, E2: Mixed, E3: News→Stories) and measure forgetting through perplexity, backward transfer, retention, and semantic metrics.

**Research Questions:**
- **RQ1** — Does catastrophic forgetting occur under sequential domain training, and how severe is it?
- **RQ2** — Does training order affect the degree and asymmetry of forgetting?
- **RQ3** — Does model scale modulate resistance to forgetting?

---

## Repository Structure

```
lmcf/
├── train.py               # Main training script — all 9 runs
├── analyze.py             # CPU-only analysis — BWT, Retention, FWT, Gap, Weight Change
├── inference_metrics.py   # GPU inference — BERTScore, Concept Coverage (1000 samples)
├── plot.ipynb             # All result plots (10 plots + 2 radar charts)
├── requirements.txt
└── README.md
```

---

## Datasets

| Domain | Source | Avg tokens/text | Token budget |
|---|---|---|---|
| Domain A — TinyStories | `roneneldan/TinyStories` | ~222 | 2M / 4M / 6M |
| Domain B — AG News | `fancyzhx/ag_news` | ~52 | 2M / 4M / 6M |

A shared BPE tokenizer (vocab_size=8,000) is trained once on the combined corpus and fixed across all model scales.

---

## Model Configurations

| Scale | Params | Layers | d_model | Heads | FFN | LR | Dropout |
|---|---|---|---|---|---|---|---|
| M1 (Small) | ~1.5M | 2 | 128 | 4 | 512 | 3e-4 | 0.1 |
| M2 (Medium) | ~7M | 6 | 256 | 8 | 1024 | 2e-4 | 0.1 |
| M3 (Large) | ~24M | 12 | 384 | 12 | 2048 | 1e-4 | 0.1 |

All models: decoder-only causal Transformer, FlashAttention, pre-norm LayerNorm, GELU FFN, weight-tied embeddings.

---

## Experiments

| Exp | Order | Stage 1 | Stage 2 |
|---|---|---|---|
| E1 | Stories → News | Train on Domain A | Train on Domain B |
| E2 | Mixed (baseline) | Train on A+B jointly | — |
| E3 | News → Stories | Train on Domain B | Train on Domain A |

9 total runs = 3 models × 3 experiments.

---

## Setup

### Requirements

```bash
# Python 3.10 recommended
conda create -n lmcf python=3.10 -y
conda activate lmcf

# PyTorch (CUDA 12.4 for RTX 6000)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install datasets tokenizers transformers pandas matplotlib scikit-learn
pip install bert-score          # for inference_metrics.py
pip install jupyter             # for plot.ipynb
```

### Project Root

Set the project root directory — all data, checkpoints, and results are stored here:

```bash
export LMCF_ROOT=/path/to/your/project
# e.g.
export LMCF_ROOT=/path/to/your/project
```

Or hardcode it at the top of `train.py`:

```python
PROJECT_ROOT = '/path/to/your/project'
```

---

## Running

### Step 1 — Training

Edit `MODELS_TO_RUN` and `EXPERIMENTS_TO_RUN` in `train.py` to select which runs to execute:

```python
MODELS_TO_RUN = ['M1', 'M2', 'M3']
EXPERIMENTS_TO_RUN = ['E1', 'E2', 'E3']
```

Run:

```bash
# Foreground
python train.py

# Background with logging (recommended for long runs)
mkdir -p $LMCF_ROOT/logs
nohup python train.py > $LMCF_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f $LMCF_ROOT/logs/run_*.log
```

Training is crash-safe with two-level recovery:
- Final stage checkpoint exists → entire run skipped, weights reloaded
- Intermediate checkpoint exists → that stage skipped, continues from next

### Step 2 — Analysis (CPU)

Run after all 9 training runs complete:

```bash
python analyze.py
```

Outputs: `metrics_summary.csv`, `learning_speed.csv`, `weight_change_summary.csv`, `combined_{model}.csv`, `metrics_summary.txt`

### Step 3 — Inference Metrics (GPU)

Generates 1,000 samples per stage (500 A-style + 500 B-style prompts), then computes BERTScore and Concept Coverage:

```bash
pip install bert-score scikit-learn
python inference_metrics.py
```

Outputs: `inference_generations.json`, `bertscore.csv`, `concept_coverage.csv`, `inference_metrics.txt`

Crash-safe — saves after each stage. Re-running skips already-generated stages.

### Step 4 — Plots

Open `plot.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook plot.ipynb
```

---

## Output Files

```
$LMCF_ROOT/
├── dataset_cache/
│   ├── A_{budget}.json             # cached domain A texts
│   ├── B_{budget}.json             # cached domain B texts
│   ├── bpe_vocab-vocab.json        # shared BPE tokenizer
│   └── bpe_vocab-merges.txt
│
├── checkpoints/
│   ├── {M}_{E}_best_{stage}.pt     # best model per stage (weights + optimizer)
│   └── {M}_{E}_metrics.pt          # val/test PPL and forgetting scores
│
└── Results/
    ├── {M}_{E}_runs.json           # full training history per stage
    ├── {M}_{E}_eval.csv            # per-stage PPL + generation metrics
    ├── {M}_{E}_curves.png          # training curves (loss, PPL, forgetting signal)
    ├── combined_{model}.csv        # all experiments merged per model
    ├── metrics_summary.csv         # BWT, Retention, FWT, Mixed Gap
    ├── learning_speed.csv          # steps to PPL threshold per stage
    ├── weight_change.csv           # per-layer weight change S1→S2
    ├── weight_change_summary.csv   # aggregated by layer type
    ├── inference_generations.json  # 1000 generated samples per stage
    ├── bertscore.csv               # BERTScore F1 per model/experiment
    ├── concept_coverage.csv        # domain concept coverage per stage
    └── metrics_summary.txt         # human-readable summary of all metrics
```

---

## Key Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| BWT_raw | PPL_A(S2) − PPL_A(S1) | Positive = forgetting occurred |
| BWT_norm | (PPL_A(S2) − PPL_A(S1)) / PPL_A(S1) | Scale-independent; use for cross-model comparison |
| Retention | PPL_A(S1) / PPL_A(S2) | 1.0 = perfect; 0 = total forgetting |
| FWT | PPL_B(E1_S2) − PPL_B(E3_S1) | Negative = pretraining helped |
| Mixed Gap | PPL_A(E1_S2) − PPL_A(E2) | Cost of sequential vs joint training |
| BERTScore F1 | Semantic similarity S1 vs S2 gens | Drop = semantic forgetting |
| Concept Coverage | Fraction of domain concepts in gens | Domain vocabulary drift |

All metrics computed on **validation set**. Test set reserved for final reported values only.

---

## Results Summary

| Model | BWT_stories (Norm) | Ret_stories | BWT_news (Norm) | Ret_news |
|---|---|---|---|---|
| M1 (Small) | +393.85 (+20×) | 0.04 | +5,783 (+50×) | 0.02 |
| M2 (Medium) | +246.74 (+19×) | 0.05 | +6,941 (+95×) | 0.01 |
| M3 (Large) | +182.96 (+15×) | 0.06 | +2,808 (+37×) | 0.03 |

Key findings:
- Catastrophic forgetting is severe at all scales — retention near zero across all conditions
- News is forgotten 37–95× vs Stories 15–20× — strong asymmetry driven by domain complexity difference
- Larger models show modest resistance but do not prevent forgetting
- BERTScore stable ~0.80 across all conditions — Coverage Score exposes the real distribution shift
- E3 (complex→simple): lm_head dominates weight change. E1 (simple→complex): uniform layer-wide shift

---

## Training Configuration

| Parameter | Value |
|---|---|
| Sequence length | 128 tokens |
| Batch size | 256 sequences |
| Tokenizer | BPE, vocab=8,000 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| LR schedule | Linear warmup (10%) + cosine decay to lr/100 |
| Gradient clipping | max norm = 1.0 |
| Mixed precision | AMP float16 |
| Early stopping | patience = 5 eval intervals |
| Val eval interval | Every 200 steps (10-batch subset) |
| Data split | 70 / 15 / 15 (train / val / test) |

---

## Hardware

Trained on: NVIDIA RTX 6000 Ada, CUDA 13.1 driver, PyTorch cu124
Environment: conda `lmcf`, Python 3.10

---

## Links

- **Models (checkpoints):** [Google Drive](https://drive.google.com/drive/folders/1_qNfhsBrWCEd6_IAMgFkcjkZGTKsweO5?usp=sharing)
- **Report:** see `reports.pdf` in repo
