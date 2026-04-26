"""
inference_metrics.py — GPU-required semantic metrics for LMCF project.
Loads saved best-model checkpoints, generates 1000 samples per stage,
then computes BERTScore and Concept Coverage.

Run on GPU server after all training is complete:
    python inference_metrics.py
    LMCF_ROOT=/home/aizan/rituraj_lm python inference_metrics.py

Dependencies:
    pip install bert-score scikit-learn

Outputs (all in Results/):
    inference_generations.json  — all 1000 generated samples per model/stage
    bertscore.csv               — BERTScore P/R/F1 per model/experiment
    concept_coverage.csv        — domain concept coverage per model/stage
    inference_metrics.txt       — human-readable summary of both
"""

import os
import json
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tokenizers import ByteLevelBPETokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
# Add root_dir
LMCF_ROOT = '.'
os.environ["LMCF_ROOT"] = LMCF_ROOT

PROJECT_ROOT = os.environ.get('LMCF_ROOT', os.path.expanduser('~/lmcf_project'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'Results')
CKPT_DIR     = os.path.join(PROJECT_ROOT, 'checkpoints')
CACHE_DIR    = os.path.join(PROJECT_ROOT, 'dataset_cache')

MODELS      = ['M1', 'M2', 'M3']
STAGE_LABELS = {
    'E1': ['after_A',  'after_A_then_B'],
    'E3': ['after_B',  'after_B_then_A'],
}
# E2 excluded — single stage, no S1 vs S2 comparison meaningful for BERTScore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device   : {device}')
print(f'Project  : {PROJECT_ROOT}')
print(f'Results  : {RESULTS_DIR}')

if device.type != 'cuda':
    print('\nWARNING: No GPU detected. Generation will be very slow.')
    print('This script is designed to run on GPU.')


# =============================================================================
# SECTION 0 - CONFIG
# =============================================================================

N_SAMPLES_PER_DOMAIN = 500  # prompts per domain (500 A + 500 B = 1000 total)
PROMPT_LEN           = 10   # tokens to use as prompt from val sequences
GEN_LEN              = 40   # new tokens to generate per prompt
TEMPERATURE          = 0.8
TOP_K                = 50
N_CONCEPTS           = 50   # top N TF-IDF words per domain

MODEL_CONFIGS = {
    'M1': dict(n_layers=2,  d_model=128, n_heads=4,  dropout=0.1),
    'M2': dict(n_layers=6,  d_model=256, n_heads=8,  dropout=0.1),
    'M3': dict(n_layers=12, d_model=384, n_heads=12,  dropout=0.1),
}

MAX_SEQ_LEN   = 128
BATCH_SIZE    = 64   # generation batch size — reduce if OOM


# =============================================================================
# SECTION 1 - LOAD TOKENIZER
# =============================================================================

print('\n--- Loading tokenizer ---')

VOCAB_FILE   = os.path.join(CACHE_DIR, 'bpe_vocab-vocab.json')
MERGES_FILE  = os.path.join(CACHE_DIR, 'bpe_vocab-merges.txt')

if not os.path.exists(VOCAB_FILE):
    raise FileNotFoundError(
        f'Tokenizer not found at {VOCAB_FILE}\n'
        f'Run train.py first to generate the tokenizer.'
    )

tokenizer       = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)
PAD_IDX         = tokenizer.token_to_id('<pad>')
BOS_IDX         = tokenizer.token_to_id('<bos>')
EOS_IDX         = tokenizer.token_to_id('<eos>')
ACTUAL_VOCAB    = tokenizer.get_vocab_size()
print(f'Vocab size: {ACTUAL_VOCAB}')


# =============================================================================
# SECTION 2 - MODEL DEFINITION (must match train.py exactly)
# =============================================================================

import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x):
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
    def __init__(self, d_model, n_heads, ff_dim, dropout):
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
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=40, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_c  = idx[:, -self.max_seq_len:]
            logits = self(idx_c)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def build_model(model_name):
    cfg = {k: v for k, v in MODEL_CONFIGS[model_name].items()}
    return DecoderOnlyTransformerLM(
        vocab_size=ACTUAL_VOCAB, max_seq_len=MAX_SEQ_LEN, **cfg
    ).to(device)


# =============================================================================
# SECTION 3 - PREPARE PROMPTS FROM VAL SET
# 1000 prompts drawn from A_val token stream.
# Each prompt = first PROMPT_LEN tokens of a val sequence.
# Same prompt set used for ALL models and stages — ensures fair comparison.
# =============================================================================

print('\n--- Preparing prompts ---')

# Find val tokens from cached data
def load_domain_texts(domain):
    for budget in [2_000_000, 4_000_000, 6_000_000, 1_000_000]:
        path = os.path.join(CACHE_DIR, f'{domain}_{budget}.json')
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)['texts'], budget
    return None, None

domain_a_texts, budget_a = load_domain_texts('A')
domain_b_texts, budget_b = load_domain_texts('B')

if not domain_a_texts:
    raise FileNotFoundError('Domain A text cache not found. Run train.py first.')
if not domain_b_texts:
    raise FileNotFoundError('Domain B text cache not found. Run train.py first.')

print(f'Domain A: {len(domain_a_texts):,} texts  (budget={budget_a:,})')
print(f'Domain B: {len(domain_b_texts):,} texts  (budget={budget_b:,})')

TRAIN_F = 0.70
VAL_F   = 0.15

def encode_texts(texts):
    ids = [BOS_IDX]
    for text in texts:
        if text.strip():
            ids.extend(tokenizer.encode(text).ids)
            ids.append(EOS_IDX)
    return ids

def extract_prompts(ids_val, n, label):
    """Extract n non-overlapping PROMPT_LEN-token prompts from val token stream."""
    stride  = max(1, (len(ids_val) - PROMPT_LEN) // n)
    prompts = []
    for i in range(0, len(ids_val) - PROMPT_LEN, stride):
        prompts.append(ids_val[i : i + PROMPT_LEN])
        if len(prompts) >= n:
            break
    print(f'  {label}: extracted {len(prompts)} prompts')
    return prompts

# Domain A val prompts
print('Encoding Domain A val...')
ids_a       = encode_texts(domain_a_texts)
a_tr        = int(len(ids_a) * TRAIN_F)
a_val_end   = int(len(ids_a) * (TRAIN_F + VAL_F))
prompts_a   = extract_prompts(ids_a[a_tr:a_val_end], N_SAMPLES_PER_DOMAIN, 'Domain A')

# Domain B val prompts
print('Encoding Domain B val...')
ids_b       = encode_texts(domain_b_texts)
b_tr        = int(len(ids_b) * TRAIN_F)
b_val_end   = int(len(ids_b) * (TRAIN_F + VAL_F))
prompts_b   = extract_prompts(ids_b[b_tr:b_val_end], N_SAMPLES_PER_DOMAIN, 'Domain B')

# Decode for JSON metadata — saved ONCE not per generation key
# Kept separate by domain — no mixing, no position arithmetic needed
def decode_prompts(prompt_list):
    return [tokenizer.decode([i for i in p if i not in (BOS_IDX, PAD_IDX, EOS_IDX)])
            for p in prompt_list]

prompt_texts_a = decode_prompts(prompts_a)
prompt_texts_b = decode_prompts(prompts_b)
print(f'Total prompts: {len(prompts_a) + len(prompts_b)} ({len(prompts_a)} A + {len(prompts_b)} B)')


# =============================================================================
# SECTION 4 - GENERATION LOOP
# For each (model, experiment, stage): load best checkpoint, generate 1000 texts.
# Uses batched generation for speed.
# Results saved to inference_generations.json as they are produced.
# =============================================================================

print('\n--- Generating samples ---')

gen_output_path = os.path.join(RESULTS_DIR, 'inference_generations.json')

# Load existing if interrupted — allows resuming
if os.path.exists(gen_output_path):
    with open(gen_output_path) as fh:
        all_generations = json.load(fh)
    print(f'Loaded existing generations: {[k for k in all_generations if k != "_prompts"]}')
else:
    all_generations = {}

# Save prompts ONCE in metadata key — separate A and B, self-documenting
if '_prompts' not in all_generations:
    all_generations['_prompts'] = {
        'A':         prompt_texts_a,   # decoded A-domain prompt texts
        'B':         prompt_texts_b,   # decoded B-domain prompt texts
        'n_a':       len(prompts_a),
        'n_b':       len(prompts_b),
        'prompt_len': PROMPT_LEN,
        'gen_len':    GEN_LEN,
    }


def generate_batch(model, prompt_ids_list, max_new=GEN_LEN,
                   temperature=TEMPERATURE, top_k=TOP_K):
    """
    Generate continuations for a list of token-id prompts.
    Pads prompts to same length within batch.
    Returns list of decoded strings.
    """
    model.eval()
    max_len  = max(len(p) for p in prompt_ids_list)
    # Pad left with PAD_IDX so all sequences end at the same position
    padded   = [[PAD_IDX] * (max_len - len(p)) + list(p) for p in prompt_ids_list]
    idx      = torch.tensor(padded, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new,
                             temperature=temperature, top_k=top_k)

    results = []
    for seq in out:
        # Strip padding and special tokens, decode continuation only
        tokens = seq[max_len:].tolist()   # only new tokens
        tokens = [t for t in tokens if t not in (PAD_IDX, BOS_IDX, EOS_IDX)]
        results.append(tokenizer.decode(tokens))
    return results


for model_name in MODELS:
    for exp_key, stage_lbls in STAGE_LABELS.items():
        for lbl in stage_lbls:
            gen_key = f'{model_name}_{exp_key}_{lbl}'

            if gen_key in all_generations:
                print(f'  {gen_key}: already generated — skipping')
                continue

            ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_{exp_key}_best_{lbl}.pt')
            if not os.path.exists(ckpt_path):
                print(f'  {gen_key}: checkpoint not found — skipping')
                continue

            print(f'\n  {gen_key}: loading checkpoint...')
            model = build_model(model_name)
            ckpt  = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            n_total = len(prompts_a) + len(prompts_b)
            print(f'  {gen_key}: generating {n_total} samples '
                  f'({len(prompts_a)} A-prompted + {len(prompts_b)} B-prompted)...')

            def run_generation(prompt_list, label):
                gens = []
                for batch_start in range(0, len(prompt_list), BATCH_SIZE):
                    batch = prompt_list[batch_start : batch_start + BATCH_SIZE]
                    gens.extend(generate_batch(model, batch))
                    if (batch_start // BATCH_SIZE) % 5 == 0:
                        done = min(batch_start + BATCH_SIZE, len(prompt_list))
                        print(f'    [{label}] {done}/{len(prompt_list)} generated...')
                return gens

            outputs_a = run_generation(prompts_a, 'A-prompts')
            outputs_b = run_generation(prompts_b, 'B-prompts')

            all_generations[gen_key] = {
                'model':      model_name,
                'experiment': exp_key,
                'stage':      lbl,
                'n_a':        len(outputs_a),
                'n_b':        len(outputs_b),
                'outputs_A':  outputs_a,   # continuations of A-domain prompts
                'outputs_B':  outputs_b,   # continuations of B-domain prompts
            }

            # Save after each stage — survives interruption
            with open(gen_output_path, 'w') as fh:
                json.dump(all_generations, fh, indent=2)
            print(f'  {gen_key}: saved {len(outputs_a)+len(outputs_b)} samples -> {gen_output_path}')

            # Free GPU memory before next model
            del model
            torch.cuda.empty_cache()


print(f'\nAll generations complete. Keys: {list(all_generations.keys())}')


# =============================================================================
# SECTION 5 - CONCEPT COVERAGE
# TF-IDF differential to extract domain-characteristic words.
# Measures what fraction of domain-specific concepts appear in generated text.
# Data-driven — no manual word lists needed.
# =============================================================================

print('\n' + '=' * 70)
print('CONCEPT COVERAGE')
print('=' * 70)

txt_path = os.path.join(RESULTS_DIR, 'inference_metrics.txt')
with open(txt_path, 'w') as f:
    f.write('=' * 70 + '\n')
    f.write('LMCF - Inference Metrics Summary\n')
    f.write(f'N_SAMPLES={N_SAMPLES_PER_DOMAIN*2} ({N_SAMPLES_PER_DOMAIN} per domain)  PROMPT_LEN={PROMPT_LEN}  GEN_LEN={GEN_LEN}\n')
    f.write('=' * 70 + '\n')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Extract domain concepts from training texts
    print('\nExtracting domain concepts via TF-IDF...')
    vectorizer = TfidfVectorizer(
        max_features = 10_000,
        stop_words   = 'english',
        ngram_range  = (1, 1),
    )
    sample_a = domain_a_texts[:2000]
    sample_b = domain_b_texts[:2000]
    vectorizer.fit(sample_a + sample_b)

    a_scores = vectorizer.transform(sample_a).mean(axis=0).A1
    b_scores = vectorizer.transform(sample_b).mean(axis=0).A1
    vocab    = vectorizer.get_feature_names_out()
    diff     = a_scores - b_scores

    domain_a_concepts = set(vocab[diff.argsort()[-N_CONCEPTS:]])
    domain_b_concepts = set(vocab[(-diff).argsort()[-N_CONCEPTS:]])

    print(f'Domain A top concepts: {sorted(domain_a_concepts)[:10]}')
    print(f'Domain B top concepts: {sorted(domain_b_concepts)[:10]}')

    def concept_coverage(texts, concepts):
        """Fraction of domain concepts present across all generated texts."""
        all_words = set(' '.join(texts).lower().split())
        return round(len(concepts & all_words) / max(1, len(concepts)), 4)

    cc_rows = []
    for gen_key, gen_data in all_generations.items():
        if gen_key == '_prompts':
            continue
        outs_a = gen_data.get('outputs_A', [])   # continuations of A-domain prompts
        outs_b = gen_data.get('outputs_B', [])   # continuations of B-domain prompts

        # Coverage on A-prompted generations (measures A retention)
        cov_a_on_a = concept_coverage(outs_a, domain_a_concepts)
        cov_b_on_a = concept_coverage(outs_a, domain_b_concepts)
        # Coverage on B-prompted generations (measures B learning)
        cov_a_on_b = concept_coverage(outs_b, domain_a_concepts)
        cov_b_on_b = concept_coverage(outs_b, domain_b_concepts)

        cc_rows.append({
            'model':             gen_data['model'],
            'experiment':        gen_data['experiment'],
            'stage':             gen_data['stage'],
            'n_a':               gen_data['n_a'],
            'n_b':               gen_data['n_b'],
            'cov_a_on_Aprompts': cov_a_on_a,  # high in S1, drops in S2 = A forgetting
            'cov_b_on_Aprompts': cov_b_on_a,  # rises in S2 = domain drift to B
            'cov_a_on_Bprompts': cov_a_on_b,  # should stay low
            'cov_b_on_Bprompts': cov_b_on_b,  # rises in S2 = B learning confirmed
        })
        print(f'  {gen_key}:')
        print(f'    A-prompts -> cov_A={cov_a_on_a:.3f}  cov_B={cov_b_on_a:.3f}')
        print(f'    B-prompts -> cov_A={cov_a_on_b:.3f}  cov_B={cov_b_on_b:.3f}')

    if cc_rows:
        cc_df   = pd.DataFrame(cc_rows)
        cc_path = os.path.join(RESULTS_DIR, 'concept_coverage.csv')
        cc_df.to_csv(cc_path, index=False)
        print(f'\nSaved -> {cc_path}')

        # Domain A concepts list for paper reference
        concepts_path = os.path.join(RESULTS_DIR, 'domain_concepts.json')
        with open(concepts_path, 'w') as fh:
            json.dump({
                'domain_A': sorted(domain_a_concepts),
                'domain_B': sorted(domain_b_concepts),
            }, fh, indent=2)
        print(f'Saved -> {concepts_path}')

        with open(txt_path, 'a') as f:
            f.write('\n\nConcept Coverage\n')
            f.write(f'Top {N_CONCEPTS} domain-characteristic words extracted via TF-IDF differential\n')
            f.write(f'Domain A concepts (top 15): {sorted(domain_a_concepts)[:15]}\n')
            f.write(f'Domain B concepts (top 15): {sorted(domain_b_concepts)[:15]}\n')
            f.write('cov_a  = fraction of Domain A concepts present in generated texts\n')
            f.write('cov_b  = fraction of Domain B concepts present\n')
            f.write('ratio  = cov_a / cov_b  (>1 = A-style,  <1 = B-style)\n')
            f.write('-' * 60 + '\n')
            f.write(cc_df.to_string(index=False))
            f.write('\n')

except ImportError:
    print('scikit-learn not installed - run: pip install scikit-learn')


# =============================================================================
# SECTION 6 - BERTScore
# Semantic similarity between Stage 1 and Stage 2 generations.
# For each model x experiment (E1, E3):
#   candidate = Stage 2 generations (same prompts as Stage 1)
#   reference  = Stage 1 generations
#   Low F1 = Stage 2 drifted semantically from Stage 1 = semantic forgetting
#
# F1 interpretation:
#   > 0.85  low semantic drift
#   0.70-0.85  moderate drift
#   < 0.70  high drift = strong semantic forgetting
# =============================================================================

print('\n' + '=' * 70)
print('BERTScore - Semantic similarity Stage 1 vs Stage 2 generations')
print(f'({N_SAMPLES_PER_DOMAIN} samples per domain per stage)')
print('=' * 70)

try:
    from bert_score import score as bert_score_fn

    bs_rows = []

    for model_name in MODELS:
        for exp_key, stage_lbls in STAGE_LABELS.items():
            lbl1, lbl2 = stage_lbls
            key1 = f'{model_name}_{exp_key}_{lbl1}'
            key2 = f'{model_name}_{exp_key}_{lbl2}'

            if key1 not in all_generations or key2 not in all_generations:
                print(f'  {model_name}_{exp_key}: generations missing — skipping')
                continue

            s1_data = all_generations[key1]
            s2_data = all_generations[key2]

            for prompt_type, refs, cands in [
                ('A-prompts',   # A-prompted: measures A retention / forgetting
                 s1_data.get('outputs_A', []),
                 s2_data.get('outputs_A', [])),
                ('B-prompts',   # B-prompted: measures B learning
                 s1_data.get('outputs_B', []),
                 s2_data.get('outputs_B', [])),
            ]:
                r_slice = refs
                c_slice = cands
                if not r_slice or not c_slice:
                    continue

                print(f'\n  {model_name}_{exp_key} [{prompt_type}]: '
                      f'BERTScore on {len(r_slice)} pairs...')
                P, R, F1 = bert_score_fn(c_slice, r_slice, lang='en',
                                         verbose=False, batch_size=64)

                mean_f1 = float(F1.mean())
                mean_p  = float(P.mean())
                mean_r  = float(R.mean())
                std_f1  = float(F1.std())

                if   mean_f1 > 0.85: interp = 'low semantic drift'
                elif mean_f1 > 0.70: interp = 'moderate semantic drift'
                else:                interp = 'HIGH semantic drift - strong forgetting'

                print(f'    F1={mean_f1:.4f} +/- {std_f1:.4f}  '
                      f'P={mean_p:.4f}  R={mean_r:.4f}  [{interp}]')

                bs_rows.append({
                    'model':          model_name,
                    'experiment':     exp_key,
                    'stage1':         lbl1,
                    'stage2':         lbl2,
                    'prompt_type':    prompt_type,  # A-prompts or B-prompts
                    'n_samples':      len(r_slice),
                    'bertscore_P':    round(mean_p,  4),
                    'bertscore_R':    round(mean_r,  4),
                    'bertscore_F1':   round(mean_f1, 4),
                    'bertscore_std':  round(std_f1,  4),
                    'interpretation': interp,
                })

    if bs_rows:
        bs_df   = pd.DataFrame(bs_rows)
        bs_path = os.path.join(RESULTS_DIR, 'bertscore.csv')
        bs_df.to_csv(bs_path, index=False)
        print(f'\nSaved -> {bs_path}')

        print('\n' + '=' * 70)
        print('BERTScore Summary')
        print('=' * 70)
        print(bs_df[['model', 'experiment', 'bertscore_F1',
                     'bertscore_std', 'interpretation']].to_string(index=False))

        with open(txt_path, 'a') as f:
            f.write('\n\nBERTScore - Semantic Similarity Stage1 vs Stage2\n')
            f.write(f'N={N_SAMPLES_PER_DOMAIN} samples per domain, Stage1=reference, Stage2=candidate\n')
            f.write('F1 > 0.85: low drift | 0.70-0.85: moderate | <0.70: high drift\n')
            f.write('-' * 60 + '\n')
            f.write(bs_df.to_string(index=False))
            f.write('\n')

except ImportError:
    print('bert-score not installed - run: pip install bert-score')


print(f'\nSaved -> {txt_path}')
print('\nDone.')
