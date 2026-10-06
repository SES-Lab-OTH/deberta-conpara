#!/usr/bin/env python3
"""
=============================================================================
BUILD v2.18 TRAIN / VAL SPLITS   -- fully deterministic, no RNG
=============================================================================

Design decisions, and where they came from
------------------------------------------
GROUPING KEY IS source_id, NOT adv_source_id.
    RAID has both. source_id is the ARTICLE: 13,371 ids, present on both
    human and AI rows, 100% overlap. adv_source_id groups a row with its
    OWN attack variants -- 13,371 ids on the human side but 454,614 on the
    AI side, and the two sets share NOTHING.
    v2.17 grouped RAID by adv_source_id. For human rows that is equivalent
    to source_id, so the human side was fine. For AI rows it is not: it
    means generations derived from a val article could sit in train.
    v2.18 groups by source_id so an article, every generation prompted from
    it, and all their attack variants stay on one side.

NO RANDOMNESS ANYWHERE.
    Val is chosen by sorting article ids WITHIN each domain and taking
    every Nth. Deterministic, reproducible, and it gives val the same
    domain mix as train. Non-RAID sources without an official val split are
    sliced the same way, ordered by a stable hash of the text so the choice
    does not depend on file ordering.

OFFICIAL VAL SPLITS ARE USED WHERE THEY EXIST.
    HC3 (combined_eval_HC3_val_QA / _SI) and MAGE (mage_val) ship their own
    val sets and are already extracted. Using them keeps the numbers
    comparable with published work instead of inventing new splits.

FULL SOURCES ARE USED IN FULL.
    HC3-Plus, MAGE, WikiHow and PeerRead go in entirely, as agreed. The
    human/AI balance is then closed with m4_human_full and the M4 human
    supplements, which exist precisely as balance filler.

Usage
    python3 build_v218_splits.py --plan     # compose and report, write nothing
    python3 build_v218_splits.py --build    # write the npz files
=============================================================================
"""

import os
import sys
import json
import hashlib
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.home() / 'Text'
FEAT = ROOT / 'features' / 'v1_62feat'
OUT = ROOT / 'features' / 'v218'
VAL_EVERY = 10          # every 10th article id -> val

# sources used IN FULL (no subsampling), as agreed
FULL_SOURCES = [
    ('HC3_Plus',  'hc3plus_train_62features.npz'),
    ('MAGE',      'mage_train_62features.npz'),
    ('PeerRead',  'm4gt_peerread_train_62features.npz'),
    ('WikiHow',   'wikihow_new_generators_62features.npz'),
    ('M4',        'm4_v217_62features.npz'),
]

# human-only filler, drawn in order until the label balance closes
FILLER = [
    ('m4_human_filler', 'm4_human_filler_v218_62features.npz'),
]

# official val splits, used as-is
OFFICIAL_VAL = [
    ('HC3_Plus', 'combined_eval_HC3_val_QA_62features.npz'),
    ('HC3_Plus', 'combined_eval_HC3_val_SI_62features.npz'),
    ('MAGE',     'mage_val_62features.npz'),
]

RAID_AI = 'raid_ai_v218_62features.npz'
RAID_AI_IDS = 'raid_ai_v218_srcids.npz'
RAID_H = 'raid_human_v218_62features.npz'


def log(m=''):
    print(m, flush=True)


def load(name):
    p = FEAT / name
    if not p.exists():
        log(f'  MISSING {name}')
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def stable_order(texts):
    """Deterministic ordering independent of file layout."""
    return np.argsort([hashlib.blake2b(str(t).encode('utf-8', 'ignore'),
                                       digest_size=8).hexdigest()
                       for t in texts])


def col(d, key, n, fill=''):
    v = d.get(key)
    if v is None:
        return np.full(n, fill, dtype=object)
    return np.asarray(v, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--raw', action='store_true',
                    help='ablation: use the _raw (no-preprocess) caches')
    args = ap.parse_args()

    global OUT, FILLER, RAID_AI, RAID_AI_IDS, RAID_H, FULL_SOURCES
    if args.raw:
        OUT = ROOT / 'features' / 'v218_raw'
        FILLER = [('m4_human_filler',
                   'm4_human_filler_v218_raw_62features.npz')]
        RAID_AI = 'raid_ai_v218_raw_62features.npz'
        RAID_AI_IDS = 'raid_ai_v218_raw_srcids.npz'
        RAID_H = 'raid_human_v218_raw_62features.npz'
        FULL_SOURCES = [
            ('HC3_Plus',  'hc3plus_train_62features.npz'),
            ('MAGE',      'mage_train_62features.npz'),
            ('PeerRead',  'm4gt_peerread_train_raw_62features.npz'),
            ('WikiHow',   'wikihow_new_generators_raw_62features.npz'),
            ('M4',        'm4_v217_raw_62features.npz'),
        ]
        log('  RAW MODE: reading _raw caches, writing features/v218_raw/')
    if not (args.plan or args.build):
        args.plan = True

    log('=' * 78)
    log('v2.18 SPLIT BUILD   (grouped on source_id, no randomness)')
    log('=' * 78)

    train, val = defaultdict(list), defaultdict(list)

    def add(bucket, src, d, idx):
        n = len(idx)
        if n == 0:
            return
        bucket['features'].append(d['features'][idx])
        bucket['labels'].append(np.asarray(d['labels'])[idx])
        bucket['texts'].append(np.asarray(d['texts'], dtype=object)[idx])
        bucket['sources'].append(np.full(n, src, dtype=object))
        for k, key in (('domains', 'domains'), ('generators', 'generators'),
                       ('attacks', 'attacks'), ('decodings', 'decodings')):
            bucket[k].append(col(d, key, len(d['labels']))[idx])

    # ------------------------------------------------------ RAID -------
    log('\n  RAID  (grouped split on source_id)')
    ha = load(RAID_H)
    ai = load(RAID_AI)
    ids = load(RAID_AI_IDS)
    if ha is None or ai is None or ids is None:
        sys.exit('  FATAL: RAID caches missing.')

    h_sid = np.asarray(ha['adv_source_ids']).astype(str)   # == source_id
    h_dom = np.asarray(ha['domains']).astype(str)
    a_sid = np.asarray(ids['source_ids']).astype(str)
    a_known = np.asarray(ids['known'])
    a_dom = np.asarray(ai['domains']).astype(str)

    # article -> domain, from the human side (authoritative)
    art_dom = {}
    for s, dd in zip(h_sid, h_dom):
        art_dom.setdefault(s, dd)

    # deterministic: sort ids within each domain, take every VAL_EVERY-th
    val_ids = set()
    by_dom = defaultdict(list)
    for s, dd in art_dom.items():
        by_dom[dd].append(s)
    for dd, lst in by_dom.items():
        lst.sort()
        val_ids.update(lst[::VAL_EVERY])
    log(f'    articles {len(art_dom):,}   -> val {len(val_ids):,} '
        f'({len(val_ids)/len(art_dom)*100:.1f}%)')

    hv = np.array([s in val_ids for s in h_sid])
    av = np.array([bool(k) and (s in val_ids)
                   for s, k in zip(a_sid, a_known)])
    a_use = a_known                      # unmatched AI rows cannot be split
    log(f'    human  train {int((~hv).sum()):>7,}  val {int(hv.sum()):>7,}')
    log(f'    AI     train {int((a_use & ~av).sum()):>7,}  '
        f'val {int(av.sum()):>7,}   '
        f'(dropped unmatched {int((~a_known).sum()):,})')

    add(train, 'RAID_human', ha, np.where(~hv)[0])
    add(val, 'RAID_human', ha, np.where(hv)[0])
    add(train, 'RAID_AI', ai, np.where(a_use & ~av)[0])
    add(val, 'RAID_AI', ai, np.where(av)[0])

    # -------------------------------------------- full sources ---------
    log('\n  FULL SOURCES (used entirely; val only where official)')
    official = {s for s, _ in OFFICIAL_VAL}
    for src, fn in FULL_SOURCES:
        d = load(fn)
        if d is None:
            continue
        n = len(d['labels'])
        lab = np.asarray(d['labels'])
        if src in official:
            add(train, src, d, np.arange(n))       # official val handled below
            log(f'    {src:<12}{fn:<44} train {n:>8,}  (official val)')
        else:
            order = stable_order(d['texts'])
            v = order[::VAL_EVERY]
            t = np.setdiff1d(np.arange(n), v, assume_unique=False)
            add(train, src, d, t)
            add(val, src, d, v)
            log(f'    {src:<12}{fn:<44} train {len(t):>8,}  val {len(v):>7,}')

    for src, fn in OFFICIAL_VAL:
        d = load(fn)
        if d is None:
            continue
        add(val, src, d, np.arange(len(d['labels'])))
        log(f'    {src:<12}{fn:<44} val   {len(d["labels"]):>8,}  [official]')

    # --------------------------------------------- balance filler ------
    def counts(b):
        if not b['labels']:
            return 0, 0
        y = np.concatenate(b['labels'])
        return int((y == 0).sum()), int((y == 1).sum())

    th, ta = counts(train)
    log(f'\n  before filler:  train human {th:,}  AI {ta:,}  '
        f'(deficit {ta-th:+,})')

    need = ta - th
    if need > 0:
        log('  adding human filler in order until balanced:')
        for src, fn in FILLER:
            if need <= 0:
                break
            d = load(fn)
            if d is None:
                continue
            lab = np.asarray(d['labels'])
            hidx = np.where(lab == 0)[0]
            order = hidx[stable_order(np.asarray(d['texts'],
                                                 dtype=object)[hidx])]
            take = order[:need]
            add(train, src, d, take)
            log(f'    {src:<16}{len(take):>9,}')
            need -= len(take)
        if need > 0:
            log(f'    STILL SHORT {need:,} human rows -- add another filler '
                f'source or reduce RAID AI.')
    elif need < 0:
        log(f'  human EXCESS {-need:,}: AI side is the smaller one.')

    # ------------------------------------------------- report ----------
    def report(name, b):
        y = np.concatenate(b['labels'])
        src = np.concatenate(b['sources'])
        dom = np.concatenate(b['domains'])
        gen = np.concatenate(b['generators'])
        atk = np.concatenate(b['attacks'])
        dec = np.concatenate(b['decodings'])
        n = len(y)
        log('\n' + '=' * 78)
        log(f'{name}:  {n:,} rows   human {int((y==0).sum()):,}  '
            f'AI {int((y==1).sum()):,}  '
            f'({(y==0).mean()*100:.1f}% / {(y==1).mean()*100:.1f}%)')
        log('=' * 78)
        log('  by source:')
        for k, v in Counter(src.tolist()).most_common():
            m = src == k
            log(f'    {str(k):<16}{v:>9,}  {v/n*100:5.2f}%   '
                f'h {int((y[m]==0).sum()):>8,}  ai {int((y[m]==1).sum()):>8,}')
        raid = np.isin(src, ['RAID_human', 'RAID_AI'])
        log(f'\n  RAID share: {raid.mean()*100:.2f}%')
        log('  AI by generator (top 12):')
        ga = Counter(gen[(y == 1) & (gen != '')].tolist())
        tg = sum(ga.values())
        for k, v in ga.most_common(12):
            log(f'    {str(k):<16}{v:>9,}  {v/max(tg,1)*100:5.2f}%')
        log('  by attack:')
        for k, v in Counter(atk[atk != ''].tolist()).most_common():
            log(f'    {str(k):<26}{v:>9,}')
        log('  by decoding (RAID AI):')
        for k, v in Counter(dec[dec != ''].tolist()).most_common():
            log(f'    {str(k):<16}{v:>9,}')
        log('  by domain (top 12):')
        for k, v in Counter(dom[dom != ''].tolist()).most_common(12):
            log(f'    {str(k):<16}{v:>9,}')
        return n

    # ---- val hygiene: drop leaks, then balance ------------------------
    # Two sources of leakage, both deterministic to remove:
    #  * HC3's official val shares 422 QA + 44 SI texts with hc3plus_train
    #  * the every-10th slices can put a text in train and its exact
    #    duplicate in val, when a source contains internal duplicates
    # Dropping from VAL (never from train) keeps the training set intact
    # and guarantees the val score is measured on unseen text.
    tr_texts = set(np.concatenate(train['texts']).tolist())
    vt = np.concatenate(val['texts'])
    keep = np.array([t not in tr_texts for t in vt])
    n_drop = int((~keep).sum())
    if n_drop:
        log(f'\n  val hygiene: dropping {n_drop:,} rows whose text also '
            f'appears in train')
        off = 0
        for i in range(len(val['labels'])):
            n = len(val['labels'][i])
            k = keep[off:off+n]
            for key in val:
                val[key][i] = val[key][i][k]
            off += n

    # val label balance: train is 50/50 but val is not, because RAID AI
    # contributes far more rows than RAID human. The threshold tau* is
    # chosen on val, so an imbalanced val moves the operating point.
    vh, va_ = counts(val)
    log(f'  val before balancing: human {vh:,}  AI {va_:,}')
    if va_ > vh:
        need_v = va_ - vh
        log(f'  adding {need_v:,} human rows to val from filler '
            f'(disjoint from the train filler slice):')
        for src, fn in FILLER:
            if need_v <= 0:
                break
            d = load(fn)
            if d is None:
                continue
            lab = np.asarray(d['labels'])
            hidx = np.where(lab == 0)[0]
            order = hidx[stable_order(np.asarray(d['texts'],
                                                 dtype=object)[hidx])]
            avail = [i for i in order if str(d['texts'][i]) not in tr_texts]
            take = np.array(avail[:need_v], dtype=int)
            if len(take):
                add(val, src, d, take)
                log(f'    {src:<16}{len(take):>9,}')
                need_v -= len(take)
        if need_v > 0:
            log(f'    still short {need_v:,} -- val will stay imbalanced')

    ntr = report('TRAIN', train)
    nva = report('VAL', val)

    # leakage assertion
    log('\n' + '=' * 78)
    log('LEAKAGE CHECK')
    log('=' * 78)
    tr_t = set(np.concatenate(train['texts']).tolist())
    va_t = set(np.concatenate(val['texts']).tolist())
    ov = len(tr_t & va_t)
    log(f'  identical texts in both splits: {ov:,}')
    if ov:
        log('  (expected >0 only if a non-RAID source repeats text across')
        log('   its own official train and val files)')

    if not args.build:
        log('\n  PLAN ONLY -- nothing written. Re-run with --build.')
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for name, b in (('train', train), ('val', val)):
        arrs = {k: (np.concatenate(v) if k != 'features'
                    else np.concatenate(v).astype(np.float32))
                for k, v in b.items()}
        p = OUT / f'v218_{name}_62features.npz'
        np.savez(p, **arrs)
        log(f'  wrote {p}  {p.stat().st_size/1e6:.1f} MB')


if __name__ == '__main__':
    main()
