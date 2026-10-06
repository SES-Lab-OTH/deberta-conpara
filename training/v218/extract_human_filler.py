#!/usr/bin/env python3
"""
=============================================================================
BALANCED HUMAN FILLER -- 1/5 from each of the five m4_human_full CSVs
=============================================================================

Why this exists
---------------
The v2.18 build needs ~476k human rows to close the label balance. Drawing
them from the existing caches put 15.6% of train and 18.9% of val into a
single register: four of the five m4_human_* caches (supp_v21,
supplement, v23_topup, v22_topup) are the SAME arXiv corpus -- they open
with identical text. Only m4_human_full_v217 is mixed.

Filler that is 90% physics abstracts is not "human writing"; it is one
genre. Since tau* is chosen on val, that skews the operating point toward
a register RAID barely contains.

So: draw equally from all five raw CSVs instead.

    arxiv_human_FULL.csv       24,890,996 rows    1.8 GB
    eli5_human_FULL.csv           309,629 rows
    peerread_human_FULL.csv        96,932 rows
    wikihow_human_FULL.csv        202,865 rows
    wikipedia_human_FULL.csv  239,583,010 rows   19.7 GB

Memory
------
wikipedia is 19.7 GB and arxiv 1.8 GB, so read_csv() on either would blow
up. This streams each file once and RESERVOIR-samples: a uniform draw over
the whole file holding only the sample in memory. Seeded, so it is
reproducible -- "shuffle before drawing" without needing the file in RAM.

Quotas
------
Equal share per file (TARGET // 5). peerread has only 96,932 rows, so if
its quota exceeds availability the shortfall is redistributed across the
files that still have capacity, and the script reports exactly what came
from where.

Gates match extract_v218_sources.py: production unicode normalisation,
>= MIN_WORDS words, English-script check.

Usage
    python3 extract_human_filler.py --plan          # counts only, no GPU
    python3 extract_human_filler.py --extract       # ~4.7 h
=============================================================================
"""

import os
import re
import sys
import csv
import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.home() / 'Text'
CSVDIR = ROOT / 'data' / 'raw' / 'm4_human_full'
FEAT = ROOT / 'features' / 'v1_62feat'
SEED = 42
TARGET = 480_000          # 443,384 train + 32,438 val, with headroom
MIN_WORDS = 30

FILES = ['arxiv_human_FULL.csv', 'eli5_human_FULL.csv',
         'peerread_human_FULL.csv', 'wikihow_human_FULL.csv',
         'wikipedia_human_FULL.csv']

sys.path.insert(0, str(ROOT))
csv.field_size_limit(10 ** 9)


def log(m=''):
    print(m, flush=True)


def get_norm():
    from src.unicode_preprocessing_v2 import unicode_normalize as pn
    probe = re.sub(r'\s+', ' ', pn('de\u0440elopment')).strip()
    if probe != 'depelopment':
        sys.exit(f'FATAL: normaliser probe {probe!r}, expected depelopment')
    log(f'  normaliser OK ({probe!r})')

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


def reservoir(path, k, norm, seed, gate=None):
    """One streaming pass, uniform sample of k accepted rows.

    Only the reservoir is held in memory, so a 19.7 GB file costs the same
    as a small one. Seeded => reproducible.
    """
    if gate is None:
        gate = norm
    rng = np.random.RandomState(seed)
    res, seen = [], 0
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        r = csv.DictReader(f)
        if 'text' not in (r.fieldnames or []):
            log(f'    no `text` column in {path.name}; got {r.fieldnames}')
            return []
        for row in r:
            t = row.get('text')
            if not t:
                continue
            g = gate(t)
            if len(g.split()) < MIN_WORDS or not english(g):
                continue
            t = norm(t)
            seen += 1
            if len(res) < k:
                res.append(t)
            else:
                j = rng.randint(0, seen)
                if j < k:
                    res[j] = t
            if seen % 500_000 == 0:
                log(f'      {path.name}: {seen:,} accepted, '
                    f'reservoir {len(res):,}')
    log(f'    {path.name}: accepted {seen:,}, kept {len(res):,}')
    return res


def count_only(path, norm, cap=200_000):
    """Fast availability probe: how many rows pass the gates in the first
    `cap` accepted, and total lines. Used by --plan."""
    n = 0
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            t = row.get('text')
            if t and len(str(t).split()) >= MIN_WORDS:
                n += 1
            if n >= cap:
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--target', type=int, default=TARGET)
    ap.add_argument('--no-preprocess', action='store_true',
                    help='ablation: disable the unicode normaliser entirely')
    args = ap.parse_args()
    if not (args.plan or args.extract):
        args.plan = True

    log('=' * 74)
    log(f'HUMAN FILLER  target {args.target:,}  (1/5 per file)')
    log('=' * 74)
    if args.no_preprocess:
        _gate = get_norm()      # gating on normalised text keeps both arms aligned
        norm = lambda t: str(t or '')
        log('  NORMALISER DISABLED (ablation) -- raw text stored, gates normalised')
    else:
        norm = get_norm()
        _gate = norm

    # known row counts, so the quota maths is explicit
    known = {'arxiv_human_FULL.csv': 24_890_996,
             'eli5_human_FULL.csv': 309_629,
             'peerread_human_FULL.csv': 96_932,
             'wikihow_human_FULL.csv': 202_865,
             'wikipedia_human_FULL.csv': 239_583_010}

    per = args.target // len(FILES)
    log(f'\n  base quota per file: {per:,}')
    quota, short = {}, 0
    for fn in FILES:
        avail = known.get(fn, 10 ** 9)
        q = min(per, avail)
        quota[fn] = q
        if q < per:
            short += per - q
            log(f'    {fn:<28}{q:>9,}  (only {avail:,} rows -- '
                f'{per-q:,} short)')
        else:
            log(f'    {fn:<28}{q:>9,}')
    if short:
        cap = [fn for fn in FILES if known.get(fn, 0) > quota[fn]]
        log(f'\n  redistributing {short:,} across {len(cap)} files with '
            f'capacity:')
        add = short // max(len(cap), 1)
        for fn in cap:
            room = known.get(fn, 10**9) - quota[fn]
            take = min(add, room)
            quota[fn] += take
            log(f'    {fn:<28}+{take:,}  -> {quota[fn]:,}')

    total = sum(quota.values())
    log(f'\n  total planned: {total:,}')
    log(f'  estimated extraction: ~{total//1700} min '
        f'({total/1700/60:.1f} h)')
    log('\n  NOTE quotas are pre-filter. Rows failing the length/English')
    log('  gates reduce the yield, so the actual total will be lower.')

    if not args.extract:
        log('\n  PLAN ONLY -- nothing read or extracted.')
        return

    texts, srcs = [], []
    for i, fn in enumerate(FILES):
        p = CSVDIR / fn
        if not p.exists():
            log(f'\n  MISSING {fn}')
            continue
        log(f'\n  streaming {fn} (quota {quota[fn]:,})...')
        got = reservoir(p, quota[fn], norm, SEED + i, gate=_gate)
        texts.extend(got)
        srcs.extend([fn.replace('_human_FULL.csv', '')] * len(got))

    log(f'\n  collected {len(texts):,} rows')
    log('  by file:')
    for k, v in Counter(srcs).most_common():
        log(f'    {k:<16}{v:>9,}  {v/len(texts)*100:5.2f}%')

    # dedupe: identical normalised text is a duplicate whatever its origin
    seen, keep = set(), []
    for i, t in enumerate(texts):
        if t not in seen:
            seen.add(t)
            keep.append(i)
    log(f'  distinct after normalisation: {len(keep):,} '
        f'({len(texts)/max(len(keep),1):.3f}x)')
    texts = [texts[i] for i in keep]
    srcs = [srcs[i] for i in keep]

    # deterministic shuffle so no source is contiguous in the file
    order = np.random.RandomState(SEED).permutation(len(texts))
    texts = [texts[i] for i in order]
    srcs = [srcs[i] for i in order]

    w = np.array([len(t.split()) for t in texts[:20000]])
    log(f'  word count: median {np.median(w):.0f}  '
        f'p10 {np.percentile(w,10):.0f}  p90 {np.percentile(w,90):.0f}')

    from train_conpara_v24 import FeatureExtractor
    log(f'\n  extracting {len(texts):,} texts '
        f'(~{max(1,len(texts)//1700)} min)...')
    ex = FeatureExtractor()
    f = ex.extract_batch(texts, desc='human_filler_v218')
    log(f'    shape={f.shape} NaN={np.isnan(f).sum()} Inf={np.isinf(f).sum()}')

    _sfx = '_raw' if args.no_preprocess else ''
    out = FEAT / f'm4_human_filler_v218{_sfx}_62features.npz'
    np.savez(out, features=f, labels=np.zeros(len(texts), dtype=int),
             texts=np.array(texts, dtype=object),
             source_files=np.array(srcs, dtype=object))
    log(f'    saved {out.name}  {out.stat().st_size/1e6:.1f} MB')
    log('\n  Next: point FILLER in build_v218_splits.py at this single cache.')


if __name__ == '__main__':
    main()
