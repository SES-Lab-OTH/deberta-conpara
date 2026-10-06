#!/usr/bin/env python3
"""
=============================================================================
DeBERTa-ConPara v2.18  --  training
=============================================================================

Single variable versus v2.17: THE DATA.
Architecture, optimiser, LR, schedule, batch, accumulation, class weights,
seed and epoch count are all unchanged, so a v2.18-vs-v2.17 difference is
attributable to the corpus rather than to the recipe.

What changed in the data (features/v218/)
-----------------------------------------
  1.56M train (from 1.00M), 171k val, both exactly 50/50.

  * GROUPED ON source_id, NOT adv_source_id.
    RAID has both. source_id is the ARTICLE (13,371 ids, shared by human
    and AI rows). adv_source_id groups a row with its own attack variants
    -- 13,371 ids on the human side but 454,614 on the AI side, sharing
    NOTHING. v2.17 split RAID AI on adv_source_id, so generations derived
    from a val article could sit in train. v2.18 has 0 shared texts.

  * RAID reweighted by hidden-test difficulty rather than uniformly.
    cohere 22.25% of AI rows (was ~9%), cohere-chat 9.61%, mistral 8.67%.
    The hidden test put cohere at 76.56 TPR@1% against llama-chat 99.57.

  * paraphrase 56,678 rows (was 1.16% of training) -- it cost 8.83 points
    on the hidden test, up to -17.3 on mistral.

  * decoding stratified: 220,179 sampling / 178,182 greedy. Sampling is
    6.2 points harder than greedy on the hidden test.

  * human filler drawn evenly from all five m4_human_full CSVs by
    reservoir sampling, instead of 90% arXiv abstracts.

Checkpoint selection
--------------------
Every epoch is saved. v2.17's epoch 1 beat epoch 5 on TPR@1% while losing
on balanced accuracy, and TPR@1% is the deployment metric -- so this logs
BOTH per epoch, plus TPR@5%, and reports which epoch wins on each. Pick
deliberately rather than by whatever BA happened to peak.

Usage
  python3 train_conpara_v218.py --dry_run      # 2000 rows, verifies the path
  python3 train_conpara_v218.py                # full run
=============================================================================
"""

import os, sys, json, time, argparse, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from transformers import DebertaV2Tokenizer, DebertaV2Model, get_linear_schedule_with_warmup
from tqdm import tqdm

TEXT_ROOT  = Path.home() / 'Text'
DATA_DIR   = TEXT_ROOT / 'features/v218'
MODEL_DIR  = TEXT_ROOT / 'models/conpara_v218'
LOG_PATH   = TEXT_ROOT / 'train_v218_log.txt'

BACKBONE   = 'microsoft/deberta-v3-large'
HIDDEN     = 1024
N_FEATURES = 30
MAX_LEN    = 512
BATCH      = 4
ACCUM      = 8
LR         = 1e-5
EPOCHS     = 5
PATIENCE   = 3
SEED       = 42
MI_SAMPLE  = 100_000
CLASS_W    = [1.5, 1.0]          # unchanged from v2.17 (human, AI)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAID_DOMAINS = ['abstracts', 'books', 'news', 'poetry',
                'recipes', 'reddit', 'reviews', 'wiki']
_log = []


def p(msg=''):
    print(msg, flush=True)
    _log.append(str(msg))


def flush_log():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('\n'.join(_log), encoding='utf-8')


class ConParaDetectorLarge(nn.Module):
    """Identical to v2.17."""
    def __init__(self, nf=N_FEATURES):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(BACKBONE)
        hs = HIDDEN
        self.feat_proj = nn.Sequential(
            nn.Linear(nf, hs), nn.GELU(), nn.Dropout(0.1))
        self.feat_gate = nn.Sequential(
            nn.Linear(hs * 2, hs), nn.Tanh(), nn.Linear(hs, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(hs * 2, hs // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hs // 2, 2))

    def forward(self, input_ids, attention_mask, features):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc.last_hidden_state[:, 0, :]
        f_ = self.feat_proj(features)
        gate = self.feat_gate(torch.cat([cls, f_], dim=1))
        return self.classifier(torch.cat([cls, gate * f_], dim=1)), None


class TextDS(Dataset):
    def __init__(self, texts, feats, labels, tok, ml=MAX_LEN):
        self.texts, self.feats, self.labels = texts, feats, labels
        self.tok, self.ml = tok, ml

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i] or '', max_length=self.ml,
                       truncation=True, padding='max_length',
                       return_tensors='pt')
        return (enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0),
                torch.tensor(self.feats[i], dtype=torch.float32),
                torch.tensor(int(self.labels[i]), dtype=torch.long))


def load_split(name):
    fp = DATA_DIR / f'v218_{name}_62features.npz'
    if not fp.exists():
        p(f'  MISSING: {fp}')
        sys.exit(1)
    d = np.load(fp, allow_pickle=True)
    return {k: d[k] for k in d.files}


def evaluate(model, loader):
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for iids, amask, feats, y in tqdm(loader, ncols=70, leave=False,
                                          desc='  val'):
            with autocast():
                logits, _ = model(iids.to(DEVICE, non_blocking=True),
                                  amask.to(DEVICE, non_blocking=True),
                                  feats.to(DEVICE, non_blocking=True))
            # keep the raw margin too: softmax saturates past ~16 logits and
            # ties destroy the ranking that TPR@FPR depends on
            lg = logits.float()
            probs.extend((lg[:, 1] - lg[:, 0]).cpu().numpy())
            labs.extend(y.tolist())
    return np.array(probs), np.array(labs)


def tpr_at_fpr(margin, y, target):
    h, a = margin[y == 0], margin[y == 1]
    if len(h) < 20 or len(a) < 20:
        return float('nan')
    return float((a >= np.quantile(h, 1 - target)).mean() * 100)


def report_val(margin, labels, sources, domains, ep):
    """BA at the best threshold, plus the low-FPR metrics that actually
    decide leaderboard placement."""
    cand = np.quantile(margin, np.linspace(0.001, 0.999, 400))
    bas = [balanced_accuracy_score(labels, (margin >= t).astype(int))
           for t in cand]
    bi = int(np.argmax(bas))
    thr_m = float(cand[bi])
    ba = bas[bi] * 100
    ba50 = balanced_accuracy_score(labels, (margin >= 0).astype(int)) * 100
    au = roc_auc_score(labels, margin)
    t1 = tpr_at_fpr(margin, labels, 0.01)
    t5 = tpr_at_fpr(margin, labels, 0.05)
    p(f'    BA@0 {ba50:.2f}  BA* {ba:.2f} (margin {thr_m:+.3f})  '
      f'AUROC {au:.5f}')
    p(f'    TPR@1% {t1:.2f}   TPR@5% {t5:.2f}')

    for nm, col, keys in (('source', sources, None),
                          ('RAID domain', domains, RAID_DOMAINS)):
        if col is None:
            continue
        p(f'    per {nm}:')
        vals = keys if keys else sorted(set(col.tolist()))
        for v in vals:
            m = col == v
            if m.sum() < 100 or len(set(labels[m].tolist())) < 2:
                continue
            p(f'      {str(v):<16}n={int(m.sum()):>7,}  '
              f'BA {balanced_accuracy_score(labels[m], (margin[m] >= thr_m).astype(int))*100:6.2f}  '
              f'TPR@1% {tpr_at_fpr(margin[m], labels[m], 0.01):6.2f}')
    return {'epoch': ep, 'ba': ba, 'ba50': ba50, 'auroc': float(au),
            'tpr1': t1, 'tpr5': t5, 'threshold_margin': thr_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--raw', action='store_true',
                    help='ablation: train on the no-preprocess corpus')
    args = ap.parse_args()

    global DATA_DIR, MODEL_DIR, LOG_PATH
    if args.raw:
        DATA_DIR  = TEXT_ROOT / 'features/v218_raw'
        MODEL_DIR = TEXT_ROOT / 'models/conpara_v218_raw'
        LOG_PATH  = TEXT_ROOT / 'train_v218_raw_log.txt'

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    p('=' * 70)
    p('  DeBERTa-ConPara v2.18')
    p(f'  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   device {DEVICE}')
    p('=' * 70)

    p('\n  PHASE 1: load pre-built splits')
    tr, va = load_split('train'), load_split('val')
    X_tr, y_tr = tr['features'], tr['labels'].astype(int)
    X_va, y_va = va['features'], va['labels'].astype(int)
    t_tr = list(tr['texts'])
    t_va = list(va['texts'])
    if args.dry_run:
        n = 2000
        rs = np.random.RandomState(SEED)

        def bal(X, y, texts, extra=None):
            take = np.concatenate([
                rs.choice(np.where(y == c)[0], n // 2, replace=False)
                for c in (0, 1)])
            rs.shuffle(take)
            out = (X[take], y[take], [texts[i] for i in take])
            return out + ({k: v[take] for k, v in extra.items()},) \
                if extra is not None else out

        X_tr, y_tr, t_tr = bal(X_tr, y_tr, t_tr)
        X_va, y_va, t_va, va = bal(X_va, y_va, t_va, va)
        p(f'  DRY RUN: {n} rows each, class-balanced '
          f'(the file is ordered by source, so a head slice is all human)')
    p(f'  train {len(y_tr):,}  (h {int((y_tr==0).sum()):,} / '
      f'ai {int((y_tr==1).sum()):,})')
    p(f'  val   {len(y_va):,}  (h {int((y_va==0).sum()):,} / '
      f'ai {int((y_va==1).sum()):,})')

    p('\n  PHASE 2: refit RobustScaler + MI selection on v2.18 train')
    scaler = RobustScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)
    idx = np.random.RandomState(SEED).choice(
        len(Xs), min(MI_SAMPLE, len(Xs)), replace=False)
    mi = mutual_info_classif(Xs[idx], y_tr[idx], random_state=SEED)
    mi_idx = np.argsort(mi)[::-1][:N_FEATURES]
    p(f'  selected {N_FEATURES}: {sorted(mi_idx.tolist())}')
    p(f'  MI range {mi[mi_idx].min():.4f} .. {mi[mi_idx].max():.4f}')
    X_tr_sel = Xs[:, mi_idx]
    X_va_sel = scaler.transform(X_va)[:, mi_idx]

    p('\n  PHASE 3: train')
    tok = DebertaV2Tokenizer.from_pretrained(BACKBONE)
    model = ConParaDetectorLarge(nf=N_FEATURES).to(DEVICE)
    dl_tr = DataLoader(TextDS(t_tr, X_tr_sel, y_tr, tok), batch_size=BATCH,
                       shuffle=True, num_workers=4, pin_memory=True,
                       drop_last=True)
    dl_va = DataLoader(TextDS(t_va, X_va_sel, y_va, tok),
                       batch_size=BATCH * 4, shuffle=False, num_workers=4,
                       pin_memory=True)

    steps = len(dl_tr) // ACCUM * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.06), steps)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_W, dtype=torch.float32).to(DEVICE))
    amp = GradScaler()
    p(f'  steps/epoch {len(dl_tr)//ACCUM:,}  total {steps:,}')

    hist, best_ba, best_t1, bad = [], -1, -1, 0
    ckpt_path = MODEL_DIR / (f'4ds_deberta_v3_large_v218'
                             f'{"_raw" if args.raw else ""}'
                             f'_{stamp}_s{SEED}.pt')

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = run_corr = run_n = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for i, (iids, amask, feats, y) in enumerate(
                tqdm(dl_tr, ncols=70, desc=f'  ep{ep}')):
            iids = iids.to(DEVICE, non_blocking=True)
            amask = amask.to(DEVICE, non_blocking=True)
            feats = feats.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with autocast():
                logits, _ = model(iids, amask, feats)
                loss = crit(logits, y) / ACCUM
            amp.scale(loss).backward()
            if (i + 1) % ACCUM == 0:
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt); amp.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * ACCUM * len(y)
            run_corr += (logits.argmax(1) == y).sum().item()
            run_n += len(y)

        p(f'\n  epoch {ep}  loss {run_loss/run_n:.4f}  '
          f'acc {run_corr/run_n:.4f}  ({(time.time()-t0)/3600:.2f} h)')
        margin, labs = evaluate(model, dl_va)
        r = report_val(margin, labs, va.get('sources'), va.get('domains'), ep)
        hist.append(r)

        # every epoch saved -- selection happens afterwards, deliberately
        torch.save({'model_state_dict': model.state_dict(),
                    'mi_selected_indices': mi_idx,
                    'scaler_center': scaler.center_,
                    'scaler_scale': scaler.scale_,
                    'epoch': ep, 'version': 'v2.18',
                    'val_bal_acc': r['ba'], 'tpr1': r['tpr1'],
                    'threshold_margin': r['threshold_margin'],
                    'n_train': len(y_tr), 'n_val': len(y_va)},
                   MODEL_DIR / f'epoch{ep}_{stamp}_s{SEED}.pt')

        if r['ba'] > best_ba:
            best_ba, bad = r['ba'], 0
            torch.save({'model_state_dict': model.state_dict(),
                        'mi_selected_indices': mi_idx,
                        'scaler_center': scaler.center_,
                        'scaler_scale': scaler.scale_,
                        'epoch': ep, 'version': 'v2.18',
                        'val_bal_acc': r['ba'], 'tpr1': r['tpr1'],
                        'threshold_margin': r['threshold_margin'],
                        'n_train': len(y_tr), 'n_val': len(y_va)}, ckpt_path)
            p(f'    saved best-BA -> {ckpt_path.name}')
        else:
            bad += 1
            if bad >= PATIENCE:
                p('    early stop'); break
        best_t1 = max(best_t1, r['tpr1'])
        flush_log()

    p('\n' + '=' * 70)
    p('  SUMMARY  (choose the checkpoint deliberately)')
    p('=' * 70)
    p(f'  {"epoch":>6}{"BA*":>9}{"AUROC":>10}{"TPR@1%":>9}{"TPR@5%":>9}')
    for r in hist:
        p(f'  {r["epoch"]:>6}{r["ba"]:>9.2f}{r["auroc"]:>10.5f}'
          f'{r["tpr1"]:>9.2f}{r["tpr5"]:>9.2f}')
    if hist:
        bb = max(hist, key=lambda r: r['ba'])
        bt = max(hist, key=lambda r: r['tpr1'])
        p(f'\n  best BA    : epoch {bb["epoch"]}  ({bb["ba"]:.2f})')
        p(f'  best TPR@1%: epoch {bt["epoch"]}  ({bt["tpr1"]:.2f})')
        if bb['epoch'] != bt['epoch']:
            p('  THESE DISAGREE. v2.17 had the same split -- epoch 1 won on')
            p('  TPR@1% while epoch 5 won on BA. TPR@1% is the deployment')
            p('  metric; epoch checkpoints are all on disk, so submit the')
            p('  one that wins on the metric you are judged by.')
    json.dump(hist, open(MODEL_DIR / f'history_{stamp}.json', 'w'), indent=2)
    flush_log()
    p(f'\n  checkpoints in {MODEL_DIR}')


if __name__ == '__main__':
    main()
