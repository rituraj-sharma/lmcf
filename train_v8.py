"""
train.py — Catastrophic Forgetting in Small Transformer LMs
Run on GPU server: python train.py
Override project root: LMCF_ROOT=/path/to/dir python train.py

tmux new -s lm
conda activate lmcf
CUDA_VISIBLE_DEVICES=0 python train_v2.py

Detach: Ctrl+B then D — job keeps running
Re-attach later: tmux attach -t lm
conda deactivate

MAX_STEPS_PER_STAGE = {
    'M1': 5_000,    # ~116 passes — T4 ~15 min/stage
    'M2': 8_000,    # ~186 passes — T4 ~50 min/stage
    'M3': 10_000,   # ~233 passes — RTX 6000 ~90 min/stage
}

EVAL_EVERY = {
    'M1': 200,    # 25 eval points total
    'M2': 320,    # 25 eval points total
    'M3': 400,    # 25 eval points total
}

LOG_EVERY = {
    'M1': 50,
    'M2': 80,
    'M3': 100,
}

EARLY_STOP_PATIENCE = 5   # now meaningful — 5 × 200 = 1000 steps without improvement


"""

# ── Imports ───────────────────────────────────────────────────────────────────
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use('Agg')          # no display needed on server
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIGURATION  (edit here before running)
# ═══════════════════════════════════════════════════════════════════════════════
# Add root_dir
LMCF_ROOT = '.'
os.environ["LMCF_ROOT"] = LMCF_ROOT

# ── Storage ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get('LMCF_ROOT', os.path.expanduser('~/lmcf_project'))
DRIVE_CACHE  = os.path.join(PROJECT_ROOT, 'dataset_cache')
CKPT_DIR     = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'Results')
for d in (DRIVE_CACHE, CKPT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
print(f'Project  : {PROJECT_ROOT}')
print(f'Results  : {RESULTS_DIR}')

# ── Run selection ─────────────────────────────────────────────────────────────
MODELS_TO_RUN = [
    # 'M1',
    'M2',
    # 'M3',
]
EXPERIMENTS_TO_RUN = [
    'E1',   # A → B
    # 'E2', # Mixed
    # 'E3', # B → A
]

# ── Tokenizer / data ──────────────────────────────────────────────────────────
VOCAB_SIZE             = 8000   # fixed across all scales
MAX_SEQ_LEN            = 128
BATCH_SIZE             = 256
TOKENS_BUDGET_PER_DOMAIN = 2_000_000
NUM_WORKERS            = min(4, os.cpu_count() or 1)
RANDOM_SEED            = 42
random.seed(RANDOM_SEED)

# ── Training ──────────────────────────────────────────────────────────────────
MAX_STEPS_PER_STAGE = {'M1': 5_000,   'M2': 5_000, 'M3': 5_000}
EVAL_EVERY          = 200           # val PPL every N steps
LOG_EVERY           = 50            # train loss print every N steps (cheap)
EARLY_STOP_PATIENCE = 5             # eval intervals without improvement
TRACK_FORGETTING_PPL = True         # set False for ~10% speed-up
VAL_BATCHES_TRAIN    = 10           # max batches during in-training eval (None = full val set)
                                    # 10 × 256 × 128 = 327K tokens — fast approximation
                                    # full val set used at stage boundary only

# ── Model configs (architecture + per-model LR and dropout) ───────────────────
MODEL_CONFIGS = {
    'M1': dict(n_layers=2,  d_model=128, n_heads=4,  lr=3e-4, dropout=0.1),
    'M2': dict(n_layers=6,  d_model=256, n_heads=8,  lr=2e-4, dropout=0.1),
    'M3': dict(n_layers=12, d_model=384, n_heads=12, lr=1e-4, dropout=0.1),
}

STAGE_LABELS = {
    'E1': ['after_A',  'after_A_then_B'],
    'E2': ['mixed'],
    'E3': ['after_B',  'after_B_then_A'],
}

# Fixed prompts reused for per-stage generation logging
PROMPTS = [
    ('once upon a time there was a little', 'A-style'),
    ('the president announced that the new', 'B-style'),
    ('the dog ran to the',                   'A-style'),
    ('scientists have discovered that',       'B-style'),
]

print(f'Models      : {MODELS_TO_RUN}')
print(f'Experiments : {EXPERIMENTS_TO_RUN}')
print(f'Steps/stage : {MAX_STEPS_PER_STAGE}')
print(f'Total runs  : {len(MODELS_TO_RUN) * len(EXPERIMENTS_TO_RUN)}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_temp_tok = AutoTokenizer.from_pretrained('gpt2')

def count_tokens(text: str) -> int:
    return len(_temp_tok.encode(text))

def stream_domain_A(seed: int = 42):
    ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
    return ds.shuffle(buffer_size=50_000, seed=seed).skip(5_000)

def load_domain_B(seed: int = 42) -> list:
    ds    = load_dataset('fancyzhx/ag_news', split='train')
    texts = [row['text'] for row in ds]
    random.seed(seed)
    random.shuffle(texts)
    return texts

def _sample_stream(dataset, budget: int) -> tuple:
    collected, total = [], 0
    for row in dataset:
        text = row['text']
        n    = count_tokens(text)
        if n == 0:
            continue
        collected.append(text)
        total += n
        if total >= budget:
            break
    return collected, total

def _sample_list(texts: list, budget: int) -> tuple:
    collected, total = [], 0
    for text in texts:
        n = count_tokens(text)
        if n == 0:
            continue
        collected.append(text)
        total += n
        if total >= budget:
            break
    return collected, total

def load_or_cache(domain: str, budget: int) -> tuple:
    path = os.path.join(DRIVE_CACHE, f'{domain}_{budget}.json')
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        print(f'[Domain {domain}] cache hit — {len(data["texts"]):,} texts, {data["n_tokens"]:,} tokens')
        return data['texts'], data['n_tokens']

    print(f'[Domain {domain}] downloading...')
    if domain == 'A':
        texts, n = _sample_stream(stream_domain_A(RANDOM_SEED), budget)
    elif domain == 'B':
        texts, n = _sample_list(load_domain_B(RANDOM_SEED), budget)
    else:
        raise ValueError(domain)

    with open(path, 'w') as fh:
        json.dump({'texts': texts, 'n_tokens': n, 'domain': domain, 'budget': budget}, fh)
    print(f'[Domain {domain}] saved {len(texts):,} texts, {n:,} tokens')
    return texts, n


print('\n--- Loading data ---')
domain_a_texts, tokens_a = load_or_cache('A', TOKENS_BUDGET_PER_DOMAIN)
domain_b_texts, tokens_b = load_or_cache('B', TOKENS_BUDGET_PER_DOMAIN)
print(f'Balance ratio: {min(tokens_a, tokens_b) / max(tokens_a, tokens_b):.3f}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BPE TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

TOKENIZER_STEM = os.path.join(DRIVE_CACHE, 'bpe_vocab')
VOCAB_FILE     = f'{TOKENIZER_STEM}-vocab.json'
MERGES_FILE    = f'{TOKENIZER_STEM}-merges.txt'

print('\n--- Tokenizer ---')
if os.path.exists(VOCAB_FILE) and os.path.exists(MERGES_FILE):
    tokenizer = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)
    print(f'Loaded BPE tokenizer (vocab={VOCAB_SIZE:,})')
else:
    print(f'Training BPE tokenizer (vocab_size={VOCAB_SIZE:,})...')
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iter(domain_a_texts + domain_b_texts),
        vocab_size     = VOCAB_SIZE,
        min_frequency  = 2,
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>'],
    )
    tokenizer.save_model(DRIVE_CACHE, 'bpe_vocab')
    print('BPE tokenizer trained and saved')

PAD_IDX           = tokenizer.token_to_id('<pad>')
BOS_IDX           = tokenizer.token_to_id('<bos>')
EOS_IDX           = tokenizer.token_to_id('<eos>')
ACTUAL_VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {ACTUAL_VOCAB_SIZE:,}  PAD={PAD_IDX}  BOS={BOS_IDX}  EOS={EOS_IDX}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def texts_to_token_ids(texts: list) -> list:
    ids = [BOS_IDX]
    for text in texts:
        if text.strip():
            ids.extend(tokenizer.encode(text).ids)
            ids.append(EOS_IDX)
    return ids

class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list):
        self.ids = token_ids
    def __len__(self):
        return max(0, len(self.ids) - MAX_SEQ_LEN)
    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx          : idx + MAX_SEQ_LEN    ], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1      : idx + MAX_SEQ_LEN + 1], dtype=torch.long)
        return x, y

def make_loader(token_ids: list, shuffle: bool) -> DataLoader:
    return DataLoader(
        NextTokenDataset(token_ids),
        batch_size         = BATCH_SIZE,
        shuffle            = shuffle,
        drop_last          = True,
        num_workers        = NUM_WORKERS,
        pin_memory         = True,
        persistent_workers = NUM_WORKERS > 0,
    )

print('\n--- Encoding and splitting ---')
t0    = time.time()
ids_a = texts_to_token_ids(domain_a_texts)
ids_b = texts_to_token_ids(domain_b_texts)
print(f'Encoded in {time.time()-t0:.1f}s  |  A:{len(ids_a):,}  B:{len(ids_b):,} tokens')

TRAIN_F = 0.70
VAL_F   = 0.15

sa_tr = int(len(ids_a) * TRAIN_F);  sa_v = int(len(ids_a) * (TRAIN_F + VAL_F))
sb_tr = int(len(ids_b) * TRAIN_F);  sb_v = int(len(ids_b) * (TRAIN_F + VAL_F))

ids_a_train, ids_a_val, ids_a_test = ids_a[:sa_tr], ids_a[sa_tr:sa_v], ids_a[sa_v:]
ids_b_train, ids_b_val, ids_b_test = ids_b[:sb_tr], ids_b[sb_tr:sb_v], ids_b[sb_v:]

loaders = {
    'A_train':     make_loader(ids_a_train,                  shuffle=True),
    'A_val':       make_loader(ids_a_val,                    shuffle=False),
    'A_test':      make_loader(ids_a_test,                   shuffle=False),
    'B_train':     make_loader(ids_b_train,                  shuffle=True),
    'B_val':       make_loader(ids_b_val,                    shuffle=False),
    'B_test':      make_loader(ids_b_test,                   shuffle=False),
    'mixed_train': make_loader(ids_a_train + ids_b_train,    shuffle=True),
    'mixed_val':   make_loader(ids_a_val   + ids_b_val,      shuffle=False),
    'mixed_test':  make_loader(ids_a_test  + ids_b_test,     shuffle=False),
}
print('Loaders ready')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = True,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class DecoderOnlyTransformerLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=256, n_heads=4,
                 n_layers=2, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim         = ff_dim or 4 * d_model
        self.ff_dim    = ff_dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.norm      = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_c  = idx[:, -self.max_seq_len:]
            logits = self(idx_c)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if nxt.item() == EOS_IDX:
                break
            idx = torch.cat([idx, nxt], dim=1)
        return idx

def build_model(model_name: str) -> DecoderOnlyTransformerLM:
    cfg = {k: v for k, v in MODEL_CONFIGS[model_name].items() if k != 'lr'}
    return DecoderOnlyTransformerLM(
        vocab_size=ACTUAL_VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, **cfg
    ).to(device)

# Quick sanity print
for name, cfg in MODEL_CONFIGS.items():
    m = build_model(name)
    print(f'{name}: layers={cfg["n_layers"]:2d} d_model={cfg["d_model"]} '
          f'heads={cfg["n_heads"]:2d} ff_dim={m.ff_dim} params={m.count_params():,}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_loss_and_ppl(model, loader, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y  = x.to(device), y.to(device)
        loss  = F.cross_entropy(model(x).view(-1, ACTUAL_VOCAB_SIZE), y.view(-1))
        total += loss.item()
        n     += 1
    avg  = total / max(1, n)
    return avg, math.exp(min(avg, 20))

def compute_forgetting_scores(before, after):
    raw  = after - before
    norm = raw / before if before else None
    return {'raw': raw, 'norm': norm}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def _cycle(loader):
    while True:
        for batch in loader:
            yield batch

def train_stage(model, loader, val_loader, max_steps=1000, stage_name='',
                lr=3e-4, lr_min=1e-5, warmup_frac=0.1, patience=5,
                eval_every=200, log_every=50, best_ckpt=None,
                extra_val_loaders=None, ckpt_config=None,
                val_batches=None):   # None = full val; int = fast subset during training

    warmup_steps = max(1, int(max_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return (lr_min + 0.5 * (1 + math.cos(math.pi * p)) * (lr - lr_min)) / lr

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    history = {
        'steps': [], 'train_loss': [], 'train_ppl': [],
        'val_loss': [], 'val_ppl': [], 'lr': [],
        'extra_ppl': {k: [] for k in (extra_val_loaders or {})},
    }

    best_val_ppl   = float('inf')
    patience_count = 0
    stopped_early  = False
    running_loss   = 0.0
    running_n      = 0
    data_iter      = _cycle(loader)

    model.train()
    for step in range(1, max_steps + 1):
        x, y = next(data_iter)
        x    = x.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == 'cuda')):
            loss = F.cross_entropy(model(x).view(-1, ACTUAL_VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.detach()   # no GPU→CPU sync
        running_n    += 1

        # Lightweight log — no val pass
        if step % log_every == 0:
            avg = (running_loss / running_n).item()
            print(f'  [{stage_name}] {step:>6}/{max_steps}  loss={avg:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

        # Full eval
        if step % eval_every == 0 or step == max_steps:
            avg_loss = (running_loss / running_n).item()
            train_ppl = math.exp(min(avg_loss, 20))
            val_loss, val_ppl = compute_loss_and_ppl(model, val_loader, max_batches=val_batches)
            model.train()
            running_loss = 0.0
            running_n    = 0

            extra_str = ''
            for vk, vl in (extra_val_loaders or {}).items():
                _, eppl = compute_loss_and_ppl(model, vl, max_batches=val_batches)
                model.train()
                history['extra_ppl'][vk].append(eppl)
                extra_str += f'  {vk}={eppl:.1f}'

            history['steps'].append(step)
            history['train_loss'].append(avg_loss)
            history['train_ppl'].append(train_ppl)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            history['lr'].append(scheduler.get_last_lr()[0])

            print(f'  [{stage_name}] {step:>6}/{max_steps}  [EVAL]  '
                  f'loss={avg_loss:.4f}/{val_loss:.4f}  '
                  f'ppl={train_ppl:.1f}/{val_ppl:.1f}'
                  f'{extra_str}')

            if val_ppl < best_val_ppl:
                best_val_ppl   = val_ppl
                patience_count = 0
                if best_ckpt:
                    torch.save({
                        'step': step, 'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'scaler_state': scaler.state_dict(),
                        'val_ppl': val_ppl, 'config': ckpt_config or {},
                    }, best_ckpt)
            else:
                patience_count += 1

            if patience > 0 and patience_count >= patience:
                print(f'  [{stage_name}] early stop at step {step}')
                stopped_early = True
                break

    history.update({'best_val_ppl': best_val_ppl,
                    'stopped_early': stopped_early,
                    'steps_trained': step})
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_ids(ids):
    return tokenizer.decode([i for i in ids if i not in (BOS_IDX, PAD_IDX, EOS_IDX)])

def generate_text(prompt, model_obj, max_new_tokens=40, temperature=0.8, top_k=50):
    model_obj.eval()
    ids = torch.tensor([[BOS_IDX] + tokenizer.encode(prompt).ids],
                       dtype=torch.long).to(device)
    with torch.no_grad():
        out = model_obj.generate(ids, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=top_k)
    return decode_ids(out[0].tolist())

def repetition_rate(text):
    t = text.split()
    return sum(t[i] == t[i-1] for i in range(1, len(t))) / max(1, len(t)-1) if len(t) > 1 else 0.0

def distinct_n(texts, n):
    ng = []
    for text in texts:
        t = text.lower().split()
        ng.extend(tuple(t[i:i+n]) for i in range(len(t)-n+1))
    return len(set(ng)) / max(1, len(ng))

def _collect_generation_samples(model_obj):
    return [{'prompt': p, 'style': s,
             'output': generate_text(p, model_obj),
             'rep_rate': round(repetition_rate(generate_text(p, model_obj)), 4)}
            for p, s in PROMPTS]

def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float(x) if hasattr(x, 'item') else x for x in obj]
    return obj.item() if hasattr(obj, 'item') else obj


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RESULT SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def _now():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

def save_run_json(run_id, entry):
    path = os.path.join(RESULTS_DIR, f'{run_id}_runs.json')
    data = json.load(open(path)) if os.path.exists(path) else {}
    ts   = _now()
    while ts in data:
        ts += '_'
    data[ts] = entry
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2, default=str)

def append_eval_csv(run_id, row):
    path   = os.path.join(RESULTS_DIR, f'{run_id}_eval.csv')
    is_new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=row.keys())
        if is_new:
            w.writeheader()
        w.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RUN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_name='M1', exp_key='E1', max_steps=None,
                   patience=EARLY_STOP_PATIENCE,
                   eval_every=EVAL_EVERY, log_every=LOG_EVERY):

    run_id     = f'{model_name}_{exp_key}'
    stage_lbls = STAGE_LABELS[exp_key]
    final_best = os.path.join(CKPT_DIR, f'{run_id}_best_{stage_lbls[-1]}.pt')

    stage_loaders = {
        'E1': [loaders['A_train'], loaders['B_train']],
        'E2': [loaders['mixed_train']],
        'E3': [loaders['B_train'],  loaders['A_train']],
    }[exp_key]
    stage_vals = {
        'E1': [loaders['A_val'],  loaders['B_val']],
        'E2': [loaders['mixed_val']],
        'E3': [loaders['B_val'],  loaders['A_val']],
    }[exp_key]
    stage_extras = {
        'E1': [{'B': loaders['B_val']}, {'A': loaders['A_val']}],
        'E2': [{'A': loaders['A_val'], 'B': loaders['B_val']}],
        'E3': [{'A': loaders['A_val']}, {'B': loaders['B_val']}],
    }[exp_key]

    ckpt_cfg = {'model_name': model_name, 'exp_key': exp_key,
                'vocab_size': ACTUAL_VOCAB_SIZE, 'max_seq_len': MAX_SEQ_LEN,
                **MODEL_CONFIGS[model_name]}

    # Level-1 crash recovery
    if os.path.exists(final_best):
        print(f'\nSKIPPING {run_id} — checkpoint found')
        m   = build_model(model_name)
        ckpt = torch.load(final_best, map_location=device)
        m.load_state_dict(ckpt['model_state'])
        metrics_path = os.path.join(CKPT_DIR, f'{run_id}_metrics.pt')
        saved = torch.load(metrics_path, map_location='cpu') if os.path.exists(metrics_path) else {}
        return {'model': m, 'stage_histories': [], **saved}

    print(f'\n{"━"*55}\nRUN: {run_id}\n{"━"*55}')
    exp_model     = build_model(model_name)
    stage_hists   = []
    stage_json    = {}
    ppl_a_stage1  = ppl_b_stage1 = None

    for s_idx, (s_ldr, s_val, s_ext) in enumerate(
            zip(stage_loaders, stage_vals, stage_extras), start=1):

        lbl       = stage_lbls[s_idx - 1]
        best_ckpt = os.path.join(CKPT_DIR, f'{run_id}_best_{lbl}.pt')

        # Level-2 crash recovery
        if os.path.exists(best_ckpt):
            print(f'\nStage {s_idx} — {lbl} (loading saved best)')
            ck = torch.load(best_ckpt, map_location=device)
            exp_model.load_state_dict(ck['model_state'])
            stage_hists.append({'skipped': True, 'val_ppl': [ck['val_ppl']]})
            if s_idx == 1 and len(stage_loaders) > 1:
                _, ppl_a_stage1 = compute_loss_and_ppl(exp_model, loaders['A_val'])
                _, ppl_b_stage1 = compute_loss_and_ppl(exp_model, loaders['B_val'])
            continue

        print(f'\nStage {s_idx}/{len(stage_loaders)} — {lbl}')
        n_steps = max_steps or MAX_STEPS_PER_STAGE[model_name]
        hist = train_stage(
            exp_model, s_ldr, s_val,
            max_steps         = n_steps,
            stage_name        = f'{run_id}-{lbl}',
            lr                = MODEL_CONFIGS[model_name]['lr'],
            patience          = patience,
            eval_every        = eval_every,
            log_every         = log_every,
            best_ckpt         = best_ckpt,
            extra_val_loaders = s_ext if TRACK_FORGETTING_PPL else None,
            ckpt_config       = ckpt_cfg,
            val_batches       = VAL_BATCHES_TRAIN,
        )
        stage_hists.append(hist)

        # Reload best checkpoint — PPL and generation must be from best model
        if os.path.exists(best_ckpt):
            ck = torch.load(best_ckpt, map_location=device)
            exp_model.load_state_dict(ck['model_state'])
            print(f'  Reloaded best {lbl} (val_ppl={ck["val_ppl"]:.2f})')

        # Val + test PPL from best model — computed once, reused everywhere
        _, s_ppl_a     = compute_loss_and_ppl(exp_model, loaders['A_val'])
        _, s_ppl_b     = compute_loss_and_ppl(exp_model, loaders['B_val'])
        _, s_ppl_a_tst = compute_loss_and_ppl(exp_model, loaders['A_test'])
        _, s_ppl_b_tst = compute_loss_and_ppl(exp_model, loaders['B_test'])
        exp_model.train()

        # Generate samples from best model weights
        stage_gens = _collect_generation_samples(exp_model)
        stage_json[lbl] = {
            'label':             lbl,
            'training_history':  _to_serialisable(hist),
            'end_result': {
                'steps_trained': hist['steps_trained'],
                'stopped_early': hist['stopped_early'],
                'ppl_a_val':     s_ppl_a,
                'ppl_b_val':     s_ppl_b,
                'ppl_a_test':    s_ppl_a_tst,
                'ppl_b_test':    s_ppl_b_tst,
            },
            'sample_generation': stage_gens,
        }

        # Reuse as forgetting baseline (Stage 1 only) — no extra passes
        if s_idx == 1 and len(stage_loaders) > 1:
            ppl_a_stage1     = s_ppl_a
            ppl_b_stage1     = s_ppl_b
            ppl_a_stage1_tst = s_ppl_a_tst
            ppl_b_stage1_tst = s_ppl_b_tst
            print(f'  Forgetting baseline — '
                  f'val: PPL(A)={ppl_a_stage1:.2f} PPL(B)={ppl_b_stage1:.2f}  '
                  f'test: PPL(A)={ppl_a_stage1_tst:.2f} PPL(B)={ppl_b_stage1_tst:.2f}')

    # Reuse final stage val+test PPL — no extra passes
    # If all stages loaded from checkpoint, compute now
    try:
        ppl_a_val = s_ppl_a;     ppl_b_val = s_ppl_b
        ppl_a_tst = s_ppl_a_tst; ppl_b_tst = s_ppl_b_tst
    except NameError:
        _, ppl_a_val = compute_loss_and_ppl(exp_model, loaders['A_val'])
        _, ppl_b_val = compute_loss_and_ppl(exp_model, loaders['B_val'])
        _, ppl_a_tst = compute_loss_and_ppl(exp_model, loaders['A_test'])
        _, ppl_b_tst = compute_loss_and_ppl(exp_model, loaders['B_test'])
    fs_a = compute_forgetting_scores(ppl_a_stage1, ppl_a_val) if ppl_a_stage1 else {}
    fs_b = compute_forgetting_scores(ppl_b_stage1, ppl_b_val) if ppl_b_stage1 else {}

    # Forgetting on both val and test
    ppl_a_stage1_tst = locals().get('ppl_a_stage1_tst')
    ppl_b_stage1_tst = locals().get('ppl_b_stage1_tst')
    fs_a_tst = compute_forgetting_scores(ppl_a_stage1_tst, ppl_a_tst) if ppl_a_stage1_tst else {}
    fs_b_tst = compute_forgetting_scores(ppl_b_stage1_tst, ppl_b_tst) if ppl_b_stage1_tst else {}

    metrics = {
        # Val metrics
        'ppl_a_val':       ppl_a_val,
        'ppl_b_val':       ppl_b_val,
        'ppl_a_stage1':    ppl_a_stage1,
        'ppl_b_stage1':    ppl_b_stage1,
        'f_a_raw':         fs_a.get('raw'),
        'f_a_norm':        fs_a.get('norm'),
        'f_b_raw':         fs_b.get('raw'),
        'f_b_norm':        fs_b.get('norm'),
        # Test metrics
        'ppl_a_test':      ppl_a_tst,
        'ppl_b_test':      ppl_b_tst,
        'ppl_a_stage1_tst': ppl_a_stage1_tst,
        'ppl_b_stage1_tst': ppl_b_stage1_tst,
        'f_a_raw_tst':     fs_a_tst.get('raw'),
        'f_a_norm_tst':    fs_a_tst.get('norm'),
        'f_b_raw_tst':     fs_b_tst.get('raw'),
        'f_b_norm_tst':    fs_b_tst.get('norm'),
    }
    torch.save(metrics, os.path.join(CKPT_DIR, f'{run_id}_metrics.pt'))

    print(f'\n{run_id} results — PPL(A)={ppl_a_val:.2f}  PPL(B)={ppl_b_val:.2f}')
    if fs_a: print(f'  F(A) raw={fs_a["raw"]:+.2f}  norm={fs_a["norm"]:+.3f}')
    if fs_b: print(f'  F(B) raw={fs_b["raw"]:+.2f}  norm={fs_b["norm"]:+.3f}')

    # Per-stage ppl_a_val/ppl_b_val already set correctly above — no backfill needed

    save_run_json(run_id, {
        'run_config': {
            'model_name': model_name, 'exp_key': exp_key,
            'max_steps': MAX_STEPS_PER_STAGE[model_name],
            'eval_every': EVAL_EVERY, 'patience': EARLY_STOP_PATIENCE,
            'vocab_size': ACTUAL_VOCAB_SIZE, 'max_seq_len': MAX_SEQ_LEN,
            **MODEL_CONFIGS[model_name],
        },
        **stage_json,
    })

    return {'model': exp_model, 'stage_histories': stage_hists, **metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

results = {}
for model_name in MODELS_TO_RUN:
    for exp_key in EXPERIMENTS_TO_RUN:
        run_id = f'{model_name}_{exp_key}'
        results[run_id] = run_experiment(
            model_name=model_name, exp_key=exp_key,
            patience=EARLY_STOP_PATIENCE,
            eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        )

print(f'\nCompleted: {list(results.keys())}')




# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — (test PPL now computed per stage inside run_experiment)
# ═══════════════════════════════════════════════════════════════════════════════
# ppl_a_test and ppl_b_test are computed for each stage's best model
# inside run_experiment and stored in the JSON end_result and metrics dict.
# No separate test eval pass needed here.

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — GENERATION EVAL + CSV  (two rows per run — one per stage)
# ═══════════════════════════════════════════════════════════════════════════════
# CSV column structure — same columns in both rows, Stage 1 leaves last block blank:
#
#   Identity   : timestamp, run_id, model, experiment, stage
#   Config     : n_layers, d_model, n_heads, max_steps, lr, dropout
#   Training   : steps_trained, early_stop, best_val_ppl
#   Stage PPL  : ppl_a_end, ppl_b_end      (PPL on both domains at end of THIS stage)
#   Generation : d1, d2, rep_rate, adh_a, adh_b
#   Forgetting : ppl_a_baseline, ppl_b_baseline,  ← Stage 1 row: blank
#                f_a_raw, f_a_norm,               ← Stage 2 row: filled
#                f_b_raw, f_b_norm
#   Test       : ppl_a_test, ppl_b_test           ← Stage 1 row: blank
# ═══════════════════════════════════════════════════════════════════════════════

print('\n--- Generation Evaluation ---')

# Build domain classifier once
domain_scorer = None
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    labels = [0] * 500 + [1] * 500
    vec    = TfidfVectorizer(max_features=5000)
    X      = vec.fit_transform(domain_a_texts[:500] + domain_b_texts[:500])
    clf    = LogisticRegression(max_iter=1000, C=1.0).fit(X, labels)
    domain_scorer = lambda gens: float((clf.predict(vec.transform(gens)) == 0).mean())
    print('Domain classifier ready')
except ImportError:
    print('sklearn not available — domain adherence skipped')


def compute_generation_metrics(samples: list) -> dict:
    """D1, D2, rep_rate, adh_a, adh_b from stored JSON samples. No model inference."""
    if not samples:
        return {'d1': None, 'd2': None, 'rep_rate': None,
                'adh_a': None, 'adh_b': None}
    all_texts = [s['output'] for s in samples]
    gens_a    = [s['output'] for s in samples if s.get('style') == 'A-style']
    gens_b    = [s['output'] for s in samples if s.get('style') == 'B-style']
    adh_a = domain_scorer(gens_a) if domain_scorer and gens_a else None
    adh_b = domain_scorer(gens_b) if domain_scorer and gens_b else None
    return {
        'd1':       round(distinct_n(all_texts, 1), 4),
        'd2':       round(distinct_n(all_texts, 2), 4),
        'rep_rate': round(sum(s['rep_rate'] for s in samples) /
                          max(1, len(samples)), 4),
        'adh_a':    round(adh_a, 4) if adh_a is not None else None,
        'adh_b':    round(adh_b, 4) if adh_b is not None else None,
    }


ts = _now()

for run_id, res in results.items():
    model_name, exp_key = run_id.split('_', 1)
    stage_lbls          = STAGE_LABELS[exp_key]
    final_lbl           = stage_lbls[-1]

    # Load latest JSON entry
    json_path = os.path.join(RESULTS_DIR, f'{run_id}_runs.json')
    if not os.path.exists(json_path):
        print(f'{run_id}: no JSON log — skipping')
        continue
    with open(json_path) as fh:
        run_log = json.load(fh)
    latest = run_log[sorted(run_log.keys())[-1]]

    print(f'\n── {run_id} ──')

    for lbl in stage_lbls:
        sd      = latest.get(lbl, {})
        end     = sd.get('end_result', {})
        samples = sd.get('sample_generation', [])
        gm      = compute_generation_metrics(samples)
        is_final = (lbl == final_lbl)

        # Print stage samples + metrics
        print(f'  [{lbl}]')
        for s in samples:
            print(f'    [{s["style"]}] {s["output"][:80]}  rep={s["rep_rate"]:.3f}')
        print(f'    D1={gm["d1"]}  D2={gm["d2"]}  rep={gm["rep_rate"]}' +
              (f'  adh_a={gm["adh_a"]}  adh_b={gm["adh_b"]}' if gm['adh_a'] is not None else ''))

        csv_row = {
            # ── Identity ──────────────────────────────────────────────────────
            'timestamp':      ts,
            'run_id':         run_id,
            'model':          model_name,
            'experiment':     exp_key,
            'stage':          lbl,
            # ── Config ────────────────────────────────────────────────────────
            'n_layers':       MODEL_CONFIGS[model_name]['n_layers'],
            'd_model':        MODEL_CONFIGS[model_name]['d_model'],
            'n_heads':        MODEL_CONFIGS[model_name]['n_heads'],
            'max_steps':      MAX_STEPS_PER_STAGE[model_name],
            'lr':             MODEL_CONFIGS[model_name]['lr'],
            'dropout':        MODEL_CONFIGS[model_name]['dropout'],
            # ── Training summary ───────────────────────────────────────────────
            'steps_trained':  end.get('steps_trained'),
            'early_stop':     end.get('stopped_early'),
            # ── PPL on both domains from best model of this stage ─────────────
            'ppl_a':          end.get('ppl_a_val'),
            'ppl_b':          end.get('ppl_b_val'),
            # ── Generation quality ─────────────────────────────────────────────
            'd1':             gm['d1'],
            'd2':             gm['d2'],
            'rep_rate':       gm['rep_rate'],
            'adh_a':          gm['adh_a'],
            'adh_b':          gm['adh_b'],
            # ── Test PPL — per stage from best model ──────────────────────────
            'ppl_a_test':          end.get('ppl_a_test'),
            'ppl_b_test':          end.get('ppl_b_test'),
            # ── Forgetting on val (Stage 2 only — Stage 1 row left blank) ─────
            'ppl_a_baseline_val':  res.get('ppl_a_stage1')     if is_final else None,
            'ppl_b_baseline_val':  res.get('ppl_b_stage1')     if is_final else None,
            'f_a_raw_val':         res.get('f_a_raw')          if is_final else None,
            'f_a_norm_val':        res.get('f_a_norm')         if is_final else None,
            'f_b_raw_val':         res.get('f_b_raw')          if is_final else None,
            'f_b_norm_val':        res.get('f_b_norm')         if is_final else None,
            # ── Forgetting on test (Stage 2 only — Stage 1 row left blank) ────
            'ppl_a_baseline_test': res.get('ppl_a_stage1_tst') if is_final else None,
            'ppl_b_baseline_test': res.get('ppl_b_stage1_tst') if is_final else None,
            'f_a_raw_test':        res.get('f_a_raw_tst')      if is_final else None,
            'f_a_norm_test':       res.get('f_a_norm_tst')     if is_final else None,
            'f_b_raw_test':        res.get('f_b_raw_tst')      if is_final else None,
            'f_b_norm_test':       res.get('f_b_norm_tst')     if is_final else None,
        }
        append_eval_csv(run_id, csv_row)

print('\nGeneration eval complete.')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — TRAINING CURVES (saved to file, no display)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n--- Saving training curves ---')

for run_id, res in results.items():
    hists = res.get('stage_histories', [])
    if not hists:
        continue

    exp_key       = run_id.split('_', 1)[1] if '_' in run_id else ''
    forget_domain = {'E1': 'A', 'E3': 'B'}.get(exp_key)
    learn_domain  = {'E1': 'B', 'E3': 'A'}.get(exp_key)

    has_extra = any(h.get('extra_ppl') and any(h['extra_ppl'].values()) for h in hists if h)
    n_cols    = 3 if has_extra else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5), squeeze=False)
    ax_loss, ax_ppl = axes[0][0], axes[0][1]
    ax_fgt  = axes[0][2] if n_cols == 3 else None

    # LR on secondary y-axis of loss plot
    ax_lr = ax_loss.twinx()
    ax_lr.set_ylabel('Learning Rate', color='gray', fontsize=7)
    ax_lr.tick_params(axis='y', labelcolor='gray', labelsize=6)

    colors = ['tab:blue', 'tab:orange']
    offset = 0

    for s_idx, hist in enumerate(hists):
        if not hist or not hist.get('steps'):
            continue
        xs = [offset + s for s in hist['steps']]
        c  = colors[s_idx % len(colors)]

        # Loss (left axis)
        ax_loss.plot(xs, hist['train_loss'], color=c, ls='-',  ms=3, marker='o',
                     label=f'train S{s_idx+1}')
        ax_loss.plot(xs, hist['val_loss'],   color=c, ls='--', ms=3, marker='s',
                     label=f'val S{s_idx+1}')

        # LR (right axis) — gray dotted, one line per stage
        if hist.get('lr'):
            ax_lr.plot(xs, hist['lr'], color='gray', ls=':', lw=1.5,
                       alpha=0.8, label=f'LR S{s_idx+1}')

        # PPL
        ax_ppl.plot(xs, hist['train_ppl'], color=c, ls='-',  ms=3, marker='o',
                    label=f'train S{s_idx+1}')
        ax_ppl.plot(xs, hist['val_ppl'],   color=c, ls='--', ms=3, marker='s',
                    label=f'val S{s_idx+1}')

        # Forgetting signal — both stages for E1/E3
        # Forgetting signal — Stage 2 only
        # Stage 1 excluded: untrained domain PPL (~20000) dominates y-axis
        # and makes Stage 2 forgetting signal (~20→350) invisible
        if ax_fgt and forget_domain and learn_domain and s_idx == 1:
            ep = hist.get('extra_ppl', {})

            # Learning domain PPL going DOWN
            if hist.get('val_ppl'):
                ax_fgt.plot(xs, hist['val_ppl'], 'tab:blue', ls='-', lw=2,
                            label=f'{learn_domain}-val S2 (learning ↓)')
            # Forgetting domain PPL going UP
            if ep.get(forget_domain):
                ax_fgt.plot(xs, ep[forget_domain], 'tab:red', ls='--', lw=2,
                            label=f'{forget_domain}-val S2 (forgetting ↑)')

            if s_idx == len(hists) - 1:
                ax_fgt.set(title=f'{run_id} — Forgetting Signal',
                           xlabel='Step', ylabel='PPL')
                ax_fgt.legend(fontsize=7)
                ax_fgt.grid(alpha=0.3)

        # Stage boundary vertical line — drawn before offset update
        if s_idx < len(hists) - 1:
            bx = offset + hist.get('steps_trained', xs[-1] if xs else 0)
            for ax in ([ax_loss, ax_ppl] + ([ax_fgt] if ax_fgt else [])):
                ax.axvline(x=bx, color='black', ls='--', lw=1.5, alpha=0.5)
                ylim = ax.get_ylim()
                ax.text(bx + max(1, bx * 0.01),
                        ylim[1] - (ylim[1] - ylim[0]) * 0.08,
                        'S1→S2', fontsize=8, color='gray',
                        ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        offset += hist.get('steps_trained', xs[-1] if xs else 0)

    ax_loss.set(title=f'{run_id} — Loss + LR', xlabel='Step',
                ylabel='Cross-entropy loss')
    ax_loss.legend(fontsize=7, loc='upper right')
    ax_lr.legend(fontsize=6, loc='center right')
    ax_loss.grid(alpha=0.3)

    ax_ppl.set(title=f'{run_id} — PPL', xlabel='Step', ylabel='Perplexity')
    ax_ppl.legend(fontsize=7)
    ax_ppl.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'{run_id}_curves.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {fig_path}')

print('\nDone.')
