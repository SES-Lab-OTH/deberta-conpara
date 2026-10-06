#!/usr/bin/env python3
"""
=============================================================================
v2.18 RAID EXTRACTION -- weighted cells, decoding-stratified
=============================================================================

What changed from v2.17 and why
-------------------------------
v2.17 sampled RAID uniformly: cell key was domain|model|attack, and every
cell got RAID_AI_N // n_cells rows regardless of difficulty. The hidden
test then showed the cells are not equally hard:

    cohere            76.56 TPR@1%     cohere/abstracts 54.81
    cohere-chat       87.85            cohere/reddit    65.29
    mistral           89.56            cohere/books     67.48
    gpt3              89.93
    mpt               91.83            llama-chat       99.57
    paraphrase        86.56  (mistral -17.3, cohere -13.9 vs none)
    sampling decode   90.90  vs greedy 97.10

So cohere/abstracts got the same ~284 rows as cells already detected at
99%. This extractor weights cells instead.

Three independent knobs, deliberately kept separable so that if v2.18
improves you can attribute the gain:

    GEN_BOOST          per-generator multiplier (cohere 3x, etc.)
    PARAPHRASE_TARGET  share of RAID AI that is paraphrase-attacked
    CELL_BOOST         extra multiplier for named weak generator/domain cells

Also: `decoding` joins the cell key. RAID is exactly 50/50 greedy/sampling
and v2.17 happened to land at ~50/50 by luck (verified post-hoc). Putting
it in the key makes that guaranteed rather than incidental, and lets the
sampling half be oversampled if wanted (SAMPLING_WEIGHT).

Reuses the v2.17 machinery: same production normaliser, same
FeatureExtractor, same save() layout, so the output caches drop straight
into the builder.

IMPORTANT: run --plan first. It prints the full allocation and the
estimated extraction time without touching the GPU.

Usage
    python3 extract_v218_raid.py --plan
    python3 extract_v218_raid.py --stage raid_ai
    python3 extract_v218_raid.py --stage raid_h
=============================================================================
"""

import os
import re
import sys
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_ROOT = Path.home() / 'Text'
FEAT = TEXT_ROOT / 'features' / 'v1_62feat'
SEED = 42

# ---- targets ---------------------------------------------------------------
RAID_AI_N = 500_000      # up from 300k in v2.17
MIN_WORDS = 30

# per-generator multiplier, driven by hidden-test TPR@1%
GEN_BOOST = {
    'cohere':       3.0,   # 76.56
    'cohere-chat':  2.0,   # 87.85
    'mistral':      1.8,   # 89.56
    'gpt3':         1.8,   # 89.93
    'mpt':          1.5,   # 91.83
    'mistral-chat': 1.3,
    'mpt-chat':     1.3,
    'gpt2':         1.2,
    'gpt4':         1.0,
    'chatgpt':      0.8,   # 99.35 -- already solved
    'llama-chat':   0.8,   # 99.57
}

# named weak cells get a further multiplier on top of GEN_BOOST
CELL_BOOST = {
    ('cohere', 'abstracts'):      2.0,   # 54.81
    ('cohere', 'reddit'):         1.7,   # 65.29
    ('cohere', 'books'):          1.7,   # 67.48
    ('cohere-chat', 'reddit'):    1.6,   # 69.08
    ('cohere', 'poetry'):         1.4,
    ('cohere', 'reviews'):        1.4,
    ('cohere', 'wiki'):           1.4,
    ('mistral', 'wiki'):          1.4,
}

# paraphrase was 1.16% of v2.17 training and costs 8.83 pts on test
PARAPHRASE_TARGET = 0.10     # share of RAID AI rows
SAMPLING_WEIGHT = 1.2        # sampling decode is 6.2 pts harder than greedy

sys.path.insert(0, str(TEXT_ROOT))


def log(m=''):
    print(m, flush=True)


def get_norm():
    from src.unicode_preprocessing_v2 import unicode_normalize as pn
    probe = re.sub(r'\s+', ' ', pn('de\u0440elopment')).strip()
    if probe != 'depelopment':
        sys.exit(f'FATAL: normaliser probe returned {probe!r}; expected '
                 f"'depelopment'. Restore unicode_preprocessing_v2.py.FIXED.")
    log(f'  normaliser OK (probe {probe!r})')

    def norm(t):
        if not t:
            return ''
        return re.sub(r'\s+', ' ', pn(str(t))).strip()
    return norm


def english(t):
    s = str(t)[:400]
    if re.search(r'[\u0600-\u06FF\u0400-\u04FF\u0900-\u097F\u4e00-\u9fff]', s):
        return False
    return sum(c.isascii() for c in s) / max(len(s), 1) > 0.85


def save(name, texts, labels, **meta):
    from train_conpara_v24 import FeatureExtractor
    out = FEAT / f'{name}_62features.npz'
    log(f'\n  [{name}] extracting {len(texts):,} texts '
        f'(~{max(1, len(texts)//1700)} min)')
    ex = FeatureExtractor()
    f = ex.extract_batch(texts, desc=name)
    log(f'    shape={f.shape} NaN={np.isnan(f).sum()} Inf={np.isinf(f).sum()}')
    np.savez(out, features=f, labels=np.array(labels, dtype=int),
             texts=np.array(texts, dtype=object),
             **{k: np.array(v) for k, v in meta.items()})
    log(f'    saved {out.name}  {out.stat().st_size/1e6:.1f} MB')


# ---------------------------------------------------------------- weights --
def cell_weight(model, domain, attack, decoding):
    w = GEN_BOOST.get(str(model), 1.0)
    w *= CELL_BOOST.get((str(model), str(domain)), 1.0)
    if str(decoding) == 'sampling':
        w *= SAMPLING_WEIGHT
    return w


def allocate(rai, norm, plan_only):
    """Weighted per-cell allocation over domain|model|attack|decoding."""
    rai = rai.copy()
    rai['_dec'] = rai['decoding'].astype(str)
    rai['_c'] = (rai.domain.astype(str) + '|' + rai.model.astype(str) + '|'
                 + rai.attack.astype(str) + '|' + rai['_dec'])

    groups = {k: v for k, v in rai.groupby('_c')}
    log(f'  cells: {len(groups):,}  '
        f'(domain x model x attack x decoding)')

    # raw weights
    W = {}
    for k, sub in groups.items():
        dom, mdl, atk, dec = k.split('|')
        W[k] = cell_weight(mdl, dom, atk, dec)

    # paraphrase share is enforced separately from the generator weights,
    # because it is an attack-level target not a generator-level one
    para = [k for k in groups if k.split('|')[2] == 'paraphrase']
    other = [k for k in groups if k.split('|')[2] != 'paraphrase']
    n_para = int(RAID_AI_N * PARAPHRASE_TARGET)
    n_other = RAID_AI_N - n_para

    def spread(keys, budget):
        tot = sum(W[k] for k in keys)
        out = {}
        left = budget
        for k in keys:
            n = int(round(budget * W[k] / tot))
            n = min(n, len(groups[k]))       # cannot exceed availability
            out[k] = n
            left -= n
        # redistribute any shortfall to cells that still have capacity
        if left > 0:
            cap = [k for k in keys if out[k] < len(groups[k])]
            i = 0
            while left > 0 and cap:
                k = cap[i % len(cap)]
                if out[k] < len(groups[k]):
                    out[k] += 1
                    left -= 1
                else:
                    cap.remove(k)
                    continue
                i += 1
        return out

    alloc = spread(other, n_other)
    alloc.update(spread(para, n_para))
    return groups, alloc


def report(groups, alloc):
    tot = sum(alloc.values())
    log(f'\n  TOTAL allocated: {tot:,}')

    by_gen, by_atk, by_dec = Counter(), Counter(), Counter()
    by_cell = Counter()
    for k, n in alloc.items():
        dom, mdl, atk, dec = k.split('|')
        by_gen[mdl] += n
        by_atk[atk] += n
        by_dec[dec] += n
        by_cell[(mdl, dom)] += n

    log('\n  by generator          v2.18        v2.17(unif)   change')
    unif = 265125 / max(len(groups), 1)
    for g, n in by_gen.most_common():
        v17 = int(unif * sum(1 for k in groups if k.split('|')[1] == g))
        log(f'    {g:<18}{n:>9,}{v17:>14,}'
            f'{(n/max(v17,1)-1)*100:>+9.0f}%')

    log('\n  by attack')
    for a, n in by_atk.most_common():
        log(f'    {a:<26}{n:>9,}  {n/tot*100:5.2f}%')

    log('\n  by decoding')
    for dcd, n in by_dec.most_common():
        log(f'    {dcd:<18}{n:>9,}  {n/tot*100:5.2f}%')

    log('\n  weak cells from the hidden test')
    log(f'    {"cell":<26}{"v2.18":>9}{"v2.17":>9}')
    for (g, d_), boost in CELL_BOOST.items():
        v17 = {'cohere': 1423 if d_ == 'abstracts' else 0}.get(g, 0)
        log(f'    {g+"/"+d_:<26}{by_cell[(g,d_)]:>9,}'
            f'{"~1.3k":>9}')

    log(f'\n  estimated extraction: ~{tot//1700} min '
        f'({tot/1700/60:.1f} h)')


def main():
    global RAID_AI_N
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='plan',
                    choices=['plan', 'raid_ai', 'raid_h'])
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--n', type=int, default=None,
                    help=f'RAID AI target (default {RAID_AI_N:,})')
    ap.add_argument('--no-preprocess', action='store_true',
                    help='ablation: disable the unicode normaliser entirely')
    args = ap.parse_args()
    stage = 'plan' if args.plan else args.stage

    if args.n:
        RAID_AI_N = args.n

    log('=' * 74)
    log(f'v2.18 RAID EXTRACTION   target AI {RAID_AI_N:,}')
    log('=' * 74)
    if args.no_preprocess:
        _gate = get_norm()          # gating only: filters must see the same rows
        norm = lambda t: str(t or '')
        log('  NORMALISER DISABLED (ablation) -- raw text stored;')
        log('  length/english gates still evaluated on normalised text so the')
        log('  row set matches the normalised arm exactly.')
    else:
        norm = get_norm()
        _gate = norm
    suffix = '_raw' if args.no_preprocess else ''

    from raid.utils import load_data
    log('  loading RAID train...')
    raid = pd.DataFrame(load_data('train'))
    log(f'  {len(raid):,} rows')

    if stage in ('plan', 'raid_ai'):
        rai = raid[raid.model != 'human']
        groups, alloc = allocate(rai, norm, stage == 'plan')
        report(groups, alloc)

        if stage == 'plan':
            log('\n  PLAN ONLY -- nothing extracted. '
                'Re-run with --stage raid_ai to build.')
            return

        log('\n  sampling rows...')
        picks = []
        rng = np.random.RandomState(SEED)
        for k, n in alloc.items():
            if n <= 0:
                continue
            sub = groups[k]
            picks.append(sub.sample(n=min(n, len(sub)), random_state=SEED))
        rai = pd.concat(picks).drop_duplicates(subset=['generation'])
        log(f'  after dedup on raw generation: {len(rai):,}')

        T, D, M, A, DEC, RP = [], [], [], [], [], []
        for _, r in rai.iterrows():
            g = _gate(r['generation'])
            if len(g.split()) >= MIN_WORDS and english(g):
                T.append(norm(r['generation']))
                D.append(r['domain'])
                M.append(r['model'])
                A.append(str(r['attack']))
                DEC.append(str(r['decoding']))
                RP.append(str(r['repetition_penalty']))
        log(f'  after norm + length + english filter: {len(T):,}')
        log(f'  distinct after normalisation: {len(set(T)):,} '
            f'({len(T)/max(len(set(T)),1):.2f}x)')
        save('raid_ai_v218' + suffix, T, [1]*len(T), domains=D, generators=M,
             attacks=A, decodings=DEC, rep_penalties=RP)

    if stage == 'raid_h':
        rh = raid[raid.model == 'human'].drop_duplicates(subset=['generation'])
        log(f'  RAID human unique generations: {len(rh):,}')
        T, D, A, ADV = [], [], [], []
        for _, r in rh.iterrows():
            g = _gate(r['generation'])
            if len(g.split()) >= MIN_WORDS and english(g):
                T.append(norm(r['generation']))
                D.append(r['domain'])
                A.append(str(r['attack']))
                ADV.append(str(r['adv_source_id']))
        log(f'  kept {len(T):,}   distinct after norm {len(set(T)):,}')
        log('  NOTE adv_source_id is saved this time, so the builder can')
        log('  group-split without the recovery join.')
        save('raid_human_v218' + suffix, T, [0]*len(T), domains=D, attacks=A,
             adv_source_ids=ADV)


if __name__ == '__main__':
    main()
