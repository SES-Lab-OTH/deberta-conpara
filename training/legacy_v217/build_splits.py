#!/usr/bin/env python3
"""
v2.17 Stage 2 (patched) — leakage-free stratified split assembly.

Fixes over the first version
----------------------------
1. wikihow_v217 humans DROPPED (2,730/2,731 duplicate M4 humans)
2. m4_human_full deduplicated against every other human cache
3. RAID human split GROUPED by adv_source_id (13,371 articles x ~11 variants)
4. RAID AI    split GROUPED by adv_source_id (same structure)

adv_source_id is recovered by exact match on production-normalized text.

Usage
-----
  python3 build_v217_splits_v2.py --plan-only
  python3 build_v217_splits_v2.py
  python3 build_v217_splits_v2.py --export [--csv]
"""
import os, sys, re, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

TEXT_ROOT = Path(os.environ.get('CONPARA_ROOT',
                                Path(__file__).resolve().parent.parent))
FEAT      = TEXT_ROOT / 'features/v1_62feat'
OUT_DIR   = TEXT_ROOT / 'features/v217'
EXPORT    = TEXT_ROOT / 'data/v217_export'

SEED      = 42
VAL_FRAC  = 0.10
TARGET_H  = 500_000
TARGET_AI = 500_000

sys.path.insert(0, str(TEXT_ROOT))
from src.unicode_preprocessing_v2 import unicode_normalize as pnorm


def norm(t):
    if not t:
        return ''
    return re.sub(r'\s+', ' ', pnorm(str(t))).strip()


def load(fname):
    p = FEAT / fname
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files} | {'_file': fname}


def recover_adv_ids(cache, gen2adv, label):
    """Attach adv_source_ids by exact normalized-text match."""
    if 'adv_source_ids' in cache:
        return cache['adv_source_ids']
    ids, miss = [], 0
    for t in cache['texts']:
        a = gen2adv.get(str(t))
        if a is None:
            miss += 1
            a = f'_um_{miss}'
        ids.append(a)
    ids = np.array(ids)
    uniq = len(set(x for x in ids if not x.startswith('_um_')))
    print(f"    {label}: matched {len(ids)-miss:,}/{len(ids):,}"
          f"  groups={uniq:,}  variants/group={(len(ids)-miss)/max(uniq,1):.1f}")
    return ids


def grouped_split(groups, val_frac, rng):
    """Split so all rows sharing a group id land on the SAME side."""
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[str(g)].append(i)
    keys = sorted(by_g)
    rng.shuffle(keys)
    n_val_g = int(len(keys) * val_frac)
    val_keys = set(keys[:n_val_g])
    tr = [i for g in keys if g not in val_keys for i in by_g[g]]
    va = [i for g in val_keys for i in by_g[g]]
    return np.array(tr, dtype=int), np.array(va, dtype=int)


def strat_split(n, keys, val_frac, rng):
    """Stratified row split by tuple(keys)."""
    if not keys:
        idx = rng.permutation(n)
        nv = int(n * val_frac)
        return idx[nv:], idx[:nv]
    cells = defaultdict(list)
    for i in range(n):
        cells[tuple(str(k[i]) for k in keys)].append(i)
    tr, va = [], []
    for _, mem in sorted(cells.items()):
        m = np.array(mem)
        rng.shuffle(m)
        nv = int(len(m) * val_frac)
        va.extend(m[:nv]); tr.extend(m[nv:])
    return np.array(tr, dtype=int), np.array(va, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--csv', action='store_true')
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)

    print("=" * 76)
    print("  v2.17 STAGE 2 (patched) — leakage-free splits")
    print("=" * 76)

    # ---------------------------------------------------------- load ----
    print("\n[1] LOADING")
    C = {}
    SPEC = [
        ('hc3_tr',  'hc3plus_train_62features.npz'),
        ('hc3_vq',  'combined_eval_HC3_val_QA_62features.npz'),
        ('hc3_vs',  'combined_eval_HC3_val_SI_62features.npz'),
        ('mage_tr', 'mage_train_62features.npz'),
        ('mage_va', 'mage_val_62features.npz'),
        ('m4',      'm4_v217_62features.npz'),
        ('peer',    'm4gt_peerread_train_62features.npz'),
        ('wh',      'wikihow_v217_62features.npz'),
        ('raid_h',  'raid_human_v217_62features.npz'),
        ('raid_ai', 'raid_ai_v217_62features.npz'),
        ('fill',    'm4_human_full_v217_62features.npz'),
    ]
    for k, f in SPEC:
        c = load(f)
        if c is None:
            print(f"  MISSING {f}"); sys.exit(1)
        C[k] = c
        lab = c['labels']
        print(f"  {k:<9}{f:<44}n={len(lab):>8,} "
              f"H={int((lab==0).sum()):>7,} AI={int((lab==1).sum()):>7,}")

    # ------------------------------------------------- patch 1: wh -----
    print("\n[2] PATCH 1 — drop wikihow humans (duplicate M4)")
    m = C['wh']['labels'] == 1
    for k in ('features', 'labels', 'texts', 'generators'):
        if k in C['wh']:
            C['wh'][k] = C['wh'][k][m]
    print(f"  wikihow now AI-only: {len(C['wh']['labels']):,}")

    # ------------------------------------------------ patch 2: fill ----
    print("\n[3] PATCH 2 — dedup filler against other human caches")
    other_h = set()
    for k in ('hc3_tr', 'mage_tr', 'm4', 'peer', 'raid_h'):
        lab = C[k]['labels']
        other_h |= {str(t)[:250] for t, l in zip(C[k]['texts'], lab) if l == 0}
    print(f"  human keys elsewhere: {len(other_h):,}")
    keep = np.array([str(t)[:250] not in other_h for t in C['fill']['texts']])
    for k in ('features', 'labels', 'texts', 'domains'):
        if k in C['fill']:
            C['fill'][k] = C['fill'][k][keep]
    print(f"  filler: {len(keep):,} -> {int(keep.sum()):,} "
          f"({int((~keep).sum()):,} removed)")

    # ------------------------------------------- patch 3+4: raid ids ---
    print("\n[4] PATCH 3+4 — recover adv_source_id for grouped RAID splits")
    import pandas as pd
    from raid.utils import load_data
    print("  normalizing RAID (few min)...")
    raid = pd.DataFrame(load_data('train'))
    raid['_n'] = raid.generation.map(norm)
    gen2adv = dict(zip(raid['_n'], raid.adv_source_id.astype(str)))
    del raid
    C['raid_h']['adv']  = recover_adv_ids(C['raid_h'],  gen2adv, 'raid_human')
    C['raid_ai']['adv'] = recover_adv_ids(C['raid_ai'], gen2adv, 'raid_ai')

    # ---------------------------------------------------- splitting ----
    print("\n[5] SPLITTING")
    pools = defaultdict(lambda: {'train': [], 'val': []})

    def add(src, cache, tr, va):
        if len(tr): pools[src]['train'].append((cache, tr))
        if len(va): pools[src]['val'].append((cache, va))

    add('HC3_Plus', C['hc3_tr'], np.arange(len(C['hc3_tr']['labels'])), [])
    add('HC3_Plus', C['hc3_vq'], [], np.arange(len(C['hc3_vq']['labels'])))
    add('HC3_Plus', C['hc3_vs'], [], np.arange(len(C['hc3_vs']['labels'])))
    add('MAGE', C['mage_tr'], np.arange(len(C['mage_tr']['labels'])), [])
    add('MAGE', C['mage_va'], [], np.arange(len(C['mage_va']['labels'])))

    tr, va = strat_split(len(C['m4']['labels']),
                         [C['m4']['domains'], C['m4']['generators']],
                         VAL_FRAC, rng)
    add('M4', C['m4'], tr, va); print(f"  M4        {len(tr):,}/{len(va):,}")

    tr, va = strat_split(len(C['peer']['labels']),
                         [C['peer']['generators']], VAL_FRAC, rng)
    add('PeerRead', C['peer'], tr, va); print(f"  PeerRead  {len(tr):,}/{len(va):,}")

    tr, va = strat_split(len(C['wh']['labels']),
                         [C['wh']['generators']], VAL_FRAC, rng)
    add('WikiHow', C['wh'], tr, va); print(f"  WikiHow   {len(tr):,}/{len(va):,}")

    tr, va = grouped_split(C['raid_h']['adv'], VAL_FRAC, rng)
    add('RAID_human', C['raid_h'], tr, va)
    print(f"  RAID_h    {len(tr):,}/{len(va):,}  [GROUPED]")

    tr, va = grouped_split(C['raid_ai']['adv'], VAL_FRAC, rng)
    add('RAID_AI', C['raid_ai'], tr, va)
    print(f"  RAID_AI   {len(tr):,}/{len(va):,}  [GROUPED]")

    tr, va = strat_split(len(C['fill']['labels']),
                         [C['fill']['domains']], VAL_FRAC, rng)
    add('m4_human_full', C['fill'], tr, va)
    print(f"  filler    {len(tr):,}/{len(va):,}")

    # ------------------------------------------------------ tally ------
    def tally(split):
        t = defaultdict(lambda: {'h': 0, 'ai': 0})
        for src, sp in pools.items():
            for cache, idx in sp[split]:
                lab = cache['labels'][idx]
                t[src]['h']  += int((lab == 0).sum())
                t[src]['ai'] += int((lab == 1).sum())
        return t

    T, V = tally('train'), tally('val')

    fix_tr_h  = sum(v['h']  for k, v in T.items() if k != 'm4_human_full')
    fix_tr_ai = sum(v['ai'] for k, v in T.items() if k != 'RAID_AI')
    fill_tr   = max(0, min(T['m4_human_full']['h'], TARGET_H - fix_tr_h))
    raid_tr   = max(0, min(T['RAID_AI']['ai'], TARGET_AI - fix_tr_ai))
    tr_h, tr_ai = fix_tr_h + fill_tr, fix_tr_ai + raid_tr

    fix_va_h  = sum(v['h']  for v in V.values())
    fix_va_ai = sum(v['ai'] for k, v in V.items() if k != 'RAID_AI')
    raid_va   = max(0, min(V['RAID_AI']['ai'], fix_va_h - fix_va_ai))
    va_h, va_ai = fix_va_h, fix_va_ai + raid_va
    if va_ai < va_h:
        va_h = va_ai

    for name, tal, h_tot, ai_tot, fl, rd in (
            ('TRAIN', T, tr_h, tr_ai, fill_tr, raid_tr),
            ('VAL',   V, va_h, va_ai, None,    raid_va)):
        print(f"\n{'='*76}\n  {name}\n{'='*76}")
        print(f"  {'source':<18}{'human':>10}{'AI':>10}{'H%':>8}{'AI%':>8}")
        print("  " + "-" * 56)
        for src in sorted(tal, key=lambda s: -(tal[s]['h'] + tal[s]['ai'])):
            h, ai = tal[src]['h'], tal[src]['ai']
            if src == 'm4_human_full' and fl is not None: h = fl
            if src == 'RAID_AI': ai = rd
            if not h and not ai: continue
            print(f"  {src:<18}{h:>10,}{ai:>10,}"
                  f"{h/max(h_tot,1)*100:>7.1f}%{ai/max(ai_tot,1)*100:>7.1f}%")
        print("  " + "-" * 56)
        print(f"  {'TOTAL':<18}{h_tot:>10,}{ai_tot:>10,}   = {h_tot+ai_tot:,}")

    print(f"\n{'='*76}")
    print(f"  TRAIN {tr_h+tr_ai:,}   VAL {va_h+va_ai:,}")
    print(f"{'='*76}")

    if args.plan_only:
        print("\n  PLAN ONLY — nothing written.")
        return

    # ------------------------------------------------------ write ------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[6] WRITING -> {OUT_DIR}")
    for split in ('train', 'val'):
        F, L, TX, SR, DM, GN, AT = [], [], [], [], [], [], []
        for src, sp in pools.items():
            for cache, idx in sp[split]:
                if src == 'm4_human_full' and split == 'train':
                    idx = idx[:fill_tr]
                if src == 'RAID_AI':
                    idx = idx[:(raid_tr if split == 'train' else raid_va)]
                n = len(idx)
                F.append(cache['features'][idx])
                L.append(cache['labels'][idx])
                TX.extend(cache['texts'][idx].tolist())
                SR.extend([src] * n)
                def g(key, dflt='-'):
                    return ([str(x) for x in cache[key][idx]]
                            if key in cache else [dflt] * n)
                DM.extend(g('domains', g('sources')[0] if 'sources' in cache else '-'))
                GN.extend(g('generators'))
                AT.extend(g('attacks', 'none'))
        F = np.concatenate(F); L = np.concatenate(L)
        out = OUT_DIR / f'v217_{split}_62features.npz'
        np.savez(out, features=F, labels=L,
                 texts=np.array(TX, dtype=object), sources=np.array(SR),
                 domains=np.array(DM), generators=np.array(GN),
                 attacks=np.array(AT))
        print(f"  {split:<6}n={len(L):,} H={int((L==0).sum()):,} "
              f"AI={int((L==1).sum()):,} -> {out.name} "
              f"({out.stat().st_size/1e6:.0f} MB)")

    if args.export:
        import pandas as pd
        EXPORT.mkdir(parents=True, exist_ok=True)
        print(f"\n[7] EXPORT -> {EXPORT}")
        for split in ('train', 'val'):
            d = np.load(OUT_DIR / f'v217_{split}_62features.npz', allow_pickle=True)
            df = pd.DataFrame({'text': d['texts'], 'label': d['labels'],
                               'source': d['sources'], 'domain': d['domains'],
                               'generator': d['generators'],
                               'attack': d['attacks'], 'split': split})
            pq = EXPORT / f'v217_{split}.parquet'
            df.to_parquet(pq, index=False)
            print(f"  {pq.name} {pq.stat().st_size/1e6:.0f} MB ({len(df):,} rows)")
            if args.csv:
                cv = EXPORT / f'v217_{split}.csv'
                df.to_csv(cv, index=False)
                print(f"  {cv.name} {cv.stat().st_size/1e6:.0f} MB")

    print("\n  STAGE 2 DONE ✓")


if __name__ == '__main__':
    main()
