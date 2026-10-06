#!/usr/bin/env python3
"""
=============================================================================
ATTACH adv_source_id TO raid_ai_v218
=============================================================================

Why
---
raid_ai_v218_62features.npz carries domains / generators / attacks /
decodings / rep_penalties, but NOT adv_source_id. The human cache
(raid_human_v218) does carry it.

A deterministic, leakage-free train/val split needs the id on BOTH sides:
in RAID every generation shares its adv_source_id with the human article
it was derived from, so splitting on that id is what keeps an article and
all of its generated + attacked variants on the same side of the split.
Without it, variants of one source article would straddle train and val
and the val score would be optimistic.

Method
------
The same join used twice already, at 100% coverage on RAID rows:
cached texts are ALREADY production-normalised, so only the RAID side
needs normalising, and the lookup is exact string equality -- no
randomness, no heuristics, no fuzzy matching.

    raid['_n'] = raid.generation.map(norm)
    lut[_n] -> {adv_source_id, ...}
    ids = [lut.get(text) for text in cache['texts']]

Ambiguity is reported rather than resolved silently: several attacks
(homoglyph, whitespace, zero_width_space) collapse to byte-identical text
after normalisation, so one normalised key can map to several RAID rows.
Those rows share the same adv_source_id when they come from one article,
which is the normal case; where they genuinely disagree the row is left
unmatched and counted.

Usage
    CUDA_VISIBLE_DEVICES="" python3 attach_adv_ids_v218.py

Writes
    features/v1_62feat/raid_ai_v218_srcids.npz
        adv_source_ids, known
=============================================================================
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.home() / 'Text'
FEAT = ROOT / 'features' / 'v1_62feat'
sys.path.insert(0, str(ROOT))


def log(m=''):
    print(m, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', action='store_true',
                    help='ablation: identity normaliser, _raw caches')
    args = ap.parse_args()
    SFX = '_raw' if args.raw else ''

    if args.raw:
        norm = lambda t: str(t or '')
        log('  RAW MODE: identity normaliser, no probe')
    else:
        from src.unicode_preprocessing_v2 import unicode_normalize as pn

        def norm(t):
            if not t:
                return ''
            return re.sub(r'\s+', ' ', pn(str(t))).strip()

        probe = norm('de\u0440elopment')
        log(f'  normaliser probe {probe!r} '
            f'({"FIXED" if probe == "depelopment" else "NOT FIXED"})')
        if probe != 'depelopment':
            sys.exit('FATAL: live normaliser is not the fixed one.')

    fp = FEAT / f'raid_ai_v218{SFX}_62features.npz'
    d = np.load(fp, allow_pickle=True)
    texts = d['texts']
    log(f'  {fp.name}: {len(texts):,} rows')

    import pandas as pd
    from raid.utils import load_data
    log('  loading RAID train (few minutes)...')
    raid = pd.DataFrame(load_data('train'))
    raid = raid[raid.model != 'human']
    log(f'  RAID AI rows {len(raid):,}')

    log('  normalising RAID generations...')
    keys = raid.generation.map(norm)

    lut = defaultdict(set)
    for k, a in zip(keys, raid.source_id.astype(str)):
        lut[k].add(a)
    log(f'  distinct normalised keys: {len(lut):,}')
    amb_keys = sum(1 for v in lut.values() if len(v) > 1)
    log(f'  keys mapping to >1 adv_source_id: {amb_keys:,} '
        f'({amb_keys/len(lut)*100:.3f}%)')
    del raid

    ids = np.full(len(texts), '', dtype=object)
    ambiguous = 0
    for i, t in enumerate(texts):
        v = lut.get(str(t))
        if not v:
            continue
        if len(v) > 1:
            ambiguous += 1
            continue                       # never guess
        ids[i] = next(iter(v))

    known = ids != ''
    log(f'\n  matched   {int(known.sum()):>8,} / {len(texts):,} '
        f'({known.mean()*100:.2f}%)')
    log(f'  ambiguous {ambiguous:>8,}')
    ug = len(set(ids[known].tolist()))
    log(f'  distinct source_id: {ug:,}')

    # sanity: do the ids overlap with the human cache?
    h = np.load(FEAT / f'raid_human_v218{SFX}_62features.npz', allow_pickle=True)
    hid = set(np.asarray(h['adv_source_ids']).astype(str).tolist())
    ov = len(set(ids[known].tolist()) & hid)
    log(f'  human cache distinct ids: {len(hid):,}')
    log(f'  overlap AI <-> human    : {ov:,} '
        f'({ov/max(ug,1)*100:.1f}% of AI ids)')
    log('  High overlap is expected and is what makes a grouped split')
    log('  meaningful: the same article appears on both sides as human')
    log('  text and as generations derived from it.')

    out = FEAT / f'raid_ai_v218{SFX}_srcids.npz'
    np.savez_compressed(out, source_ids=ids.astype(str), known=known)
    log(f'\n  wrote {out}')

    if known.mean() < 0.99:
        log('\n  WARNING coverage below 99%. Unmatched rows cannot be')
        log('  group-split and would have to be excluded from the build.')


if __name__ == '__main__':
    main()
