#!/usr/bin/env python3
"""v2.17 Stage 1: extract all missing feature caches WITH metadata."""
import os, sys, re, json, unicodedata, argparse
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

TEXT_ROOT = Path(os.environ.get('CONPARA_ROOT',
                                Path(__file__).resolve().parent.parent))
FEAT = TEXT_ROOT/'features/v1_62feat'
RAW  = TEXT_ROOT/'data/raw'
SEED = 42
RAID_AI_N = 300_000
FILLER_N  = 200_000            # 163,271 train + val headroom
FILLER_W  = {'eli5':0.45,'wikihow':0.30,'arxiv':0.15,'wikipedia':0.10}

sys.path.insert(0, str(TEXT_ROOT))
from train_conpara_v24 import FeatureExtractor

from train_conpara_v24 import unicode_normalize as _prod_norm

def norm(t):
    """Production 4-layer normalization + whitespace collapse."""
    if not t: return ''
    return re.sub(r'\s+', ' ', _prod_norm(str(t))).strip()

def english(t):
    s = str(t)[:400]
    if re.search(r'[\u0600-\u06FF\u0400-\u04FF\u0900-\u097F\u4e00-\u9fff]', s):
        return False
    return sum(c.isascii() for c in s)/max(len(s),1) > 0.85

def save(name, texts, labels, **meta):
    out = FEAT/f'{name}_62features.npz'
    print(f"\n  [{name}] extracting {len(texts):,} texts "
          f"(~{max(1,len(texts)//1700)} min)")
    ex = FeatureExtractor()
    f = ex.extract_batch(texts, desc=name)
    print(f"    shape={f.shape} NaN={np.isnan(f).sum()} Inf={np.isinf(f).sum()}")
    np.savez(out, features=f, labels=np.array(labels,dtype=int),
             texts=np.array(texts,dtype=object),
             **{k:np.array(v) for k,v in meta.items()})
    print(f"    saved {out.name}  {out.stat().st_size/1e6:.1f} MB")

ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='all',
                choices=['all','m4','wikihow','raid_h','raid_ai','filler'])
a = ap.parse_args()
S = a.stage

# ---- M4 pairs (domain x generator) ----
if S in ('all','m4'):
    print("\n" + "="*66 + "\n  M4 pairs\n" + "="*66)
    T,L,D,G = [],[],[],[]
    seen_h = set()
    for dom in ['arxiv','peerread','reddit','wikihow','wikipedia']:
        for fp in sorted((RAW/'m4').glob(f'{dom}_*.jsonl')):
            gen = fp.stem.rsplit('_',1)[-1]
            with open(fp, encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    if not line.strip(): continue
                    try: r = json.loads(line)
                    except: continue
                    m = norm(r.get('machine_text',''))
                    if len(m.split())>=30 and english(m):
                        T.append(m); L.append(1); D.append(dom); G.append(gen)
                    h = norm(r.get('human_text',''))
                    if len(h.split())>=30 and english(h) and h[:200] not in seen_h:
                        seen_h.add(h[:200])
                        T.append(h); L.append(0); D.append(dom); G.append('human')
    print(f"  AI={sum(L):,}  H={len(L)-sum(L):,}  cells={len(set(zip(D,G)))}")
    save('m4_v217', T, L, domains=D, generators=G)

# ---- WikiHow (clean, deduped) ----
if S in ('all','wikihow'):
    print("\n" + "="*66 + "\n  WikiHow (clean)\n" + "="*66)
    T,L,G = [],[],[]
    seen_h = set()
    for fp in sorted((RAW/'wikihow_generated_clean').glob('*.jsonl')):
        with open(fp, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if not line.strip(): continue
                try: r = json.loads(line)
                except: continue
                gid = r.get('generator_id','?')
                m = norm(r.get('machine_text',''))
                if len(m.split())>=30:
                    T.append(m); L.append(1); G.append(gid)
                h = norm(r.get('human_text',''))
                if len(h.split())>=30 and h[:200] not in seen_h:
                    seen_h.add(h[:200]); T.append(h); L.append(0); G.append('human')
    print(f"  AI={sum(L):,} (expect 13,958)  H={len(L)-sum(L):,}")
    save('wikihow_v217', T, L, generators=G)

# ---- RAID ----
if S in ('all','raid_h','raid_ai'):
    from raid.utils import load_data
    print("\n" + "="*66 + "\n  RAID\n" + "="*66)
    raid = pd.DataFrame(load_data('train'))

    if S in ('all','raid_h'):
        rh = raid[raid.model=='human'].drop_duplicates(subset=['generation'])
        T,D = [],[]
        for _,r in rh.iterrows():
            t = norm(r['generation'])
            if len(t.split())>=30:
                T.append(t); D.append(r['domain'])
        print(f"  human: {len(T):,}")
        save('raid_human_v217', T, [0]*len(T), domains=D)

    if S in ('all','raid_ai'):
        rai = raid[raid.model!='human'].copy()
        rai['_c'] = (rai.domain.astype(str)+'|'+rai.model.astype(str)
                     +'|'+rai.attack.astype(str))
        per = max(1, RAID_AI_N//rai['_c'].nunique())
        picks=[]
        for _,sub in rai.groupby('_c'):
            picks.append(sub.sample(n=min(per,len(sub)), random_state=SEED))
        rai = pd.concat(picks).drop_duplicates(subset=['generation'])
        T,D,M,A = [],[],[],[]
        for _,r in rai.iterrows():
            t = norm(r['generation'])
            if len(t.split())>=30:
                T.append(t); D.append(r['domain'])
                M.append(r['model']); A.append(str(r['attack']))
        print(f"  AI: {len(T):,}  cells={len(set(zip(D,M,A)))}")
        save('raid_ai_v217', T, [1]*len(T), domains=D, generators=M, attacks=A)

# ---- m4_human_full filler ----
if S in ('all','filler'):
    print("\n" + "="*66 + "\n  m4_human_full filler\n" + "="*66)
    rng = np.random.RandomState(SEED)
    T,D = [],[]
    for dom,w in FILLER_W.items():
        want = int(FILLER_N*w)
        fp = RAW/'m4_human_full'/f'{dom}_human_FULL.csv'
        if not fp.exists():
            print(f"  MISSING {fp.name}"); continue
        got=set()
        for chunk in pd.read_csv(fp, chunksize=50_000):
            col = 'text' if 'text' in chunk.columns else chunk.columns[0]
            for t in chunk[col].dropna():
                t = norm(t)
                if len(t.split())>=30 and english(t) and t[:200] not in got:
                    got.add(t[:200]); T.append(t); D.append(dom)
                    if len(got)>=want: break
            if len(got)>=want: break
        print(f"  {dom:<12}{len(got):>8,} / {want:,}")
    print(f"  filler total: {len(T):,}")
    save('m4_human_full_v217', T, [0]*len(T), domains=D)

print("\n  STAGE 1 DONE ✓")
