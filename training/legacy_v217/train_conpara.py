#!/usr/bin/env python3
"""
DeBERTa-ConPara v2.17 — training on the pre-built 1M leakage-free dataset.

Differences from v2.16
----------------------
* Data is PRE-BUILT (features/v217/) — no construction phase at train time
* RobustScaler + MI feature selection REFIT on the v2.17 training set
* Trained FROM SCRATCH (no warm start) so the data change is isolated
* Per-source validation reporting (8 sources) + RAID broken out by domain
* Corrected unicode_normalize (U+0440 -> 'p') used throughout

Usage
-----
  python3 train_conpara_v217.py --dry_run
  python3 train_conpara_v217.py
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

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
torch.set_float32_matmul_precision('high')

# ----------------------------------------------------------------- config --
TEXT_ROOT  = Path(os.environ.get('CONPARA_ROOT',
                                 Path(__file__).resolve().parent.parent))
DATA_DIR   = TEXT_ROOT / 'features/v217'
MODEL_DIR  = TEXT_ROOT / 'models/conpara_v217'
LOG_PATH   = TEXT_ROOT / 'train_v217_log.txt'

BACKBONE   = 'microsoft/deberta-v3-large'
HIDDEN     = 1024
N_FEATURES = 30          # selected by MI from the 62 extracted
MAX_LEN    = 512
BATCH      = 4
ACCUM      = 8           # effective batch 32
LR         = 1e-5
EPOCHS     = 5
PATIENCE   = 3
SEED       = 42
MI_SAMPLE  = 100_000     # subsample for MI estimation
CLASS_W    = [1.5, 1.0]  # weighted loss (human, AI) — v2.x convention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAID_DOMAINS = ['abstracts', 'books', 'news', 'poetry',
                'recipes', 'reddit', 'reviews', 'wiki']

_log_lines = []


def p(msg=''):
    print(msg, flush=True)
    _log_lines.append(str(msg))


def flush_log():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_log_lines))


# ------------------------------------------------------------------ model --
class ConParaDetectorLarge(nn.Module):
    """DeBERTa-v3-large + gated linguistic-feature fusion."""

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
        self.texts = texts
        self.feats = feats
        self.labels = labels
        self.tok = tok
        self.ml = ml

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i] or '', max_length=self.ml,
                       truncation=True, padding='max_length',
                       return_tensors='pt')
        return (enc['input_ids'].squeeze(0),
                enc['attention_mask'].squeeze(0),
                torch.tensor(self.feats[i], dtype=torch.float32),
                torch.tensor(int(self.labels[i]), dtype=torch.long))


# ------------------------------------------------------------------- data --
def load_split(name):
    fp = DATA_DIR / f'v217_{name}_62features.npz'
    if not fp.exists():
        p(f'  MISSING: {fp}')
        sys.exit(1)
    d = np.load(fp, allow_pickle=True)
    return {k: d[k] for k in d.files}


def evaluate(model, loader, scaler_amp=None):
    """Return (probs, labels) over a loader."""
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for iids, amask, feats, y in tqdm(loader, ncols=70, leave=False,
                                          desc='  val'):
            with autocast():
                logits, _ = model(iids.to(DEVICE, non_blocking=True),
                                  amask.to(DEVICE, non_blocking=True),
                                  feats.to(DEVICE, non_blocking=True))
            probs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
            labs.extend(y.tolist())
    return np.array(probs), np.array(labs)


def report_val(probs, labels, sources, domains, thr=0.5):
    """Overall + per-source + RAID-per-domain balanced accuracy."""
    pred = (probs >= thr).astype(int)
    overall = balanced_accuracy_score(labels, pred) * 100
    p(f'    Val BA (tau={thr:.2f}): {overall:.2f}%'
      f'   AUROC: {roc_auc_score(labels, probs):.4f}')

    p(f'    {"source":<16}{"n":>9}{"BA":>9}')
    for src in sorted(set(sources)):
        m = sources == src
        if m.sum() < 50 or len(np.unique(labels[m])) < 2:
            continue
        ba = balanced_accuracy_score(labels[m], pred[m]) * 100
        p(f'    {src:<16}{int(m.sum()):>9,}{ba:>8.2f}%')

    # RAID by domain — human and AI pooled per domain
    raid_m = np.isin(sources, ['RAID_human', 'RAID_AI'])
    if raid_m.sum() > 0:
        p(f'    {"RAID domain":<16}{"n":>9}{"BA":>9}')
        for dom in RAID_DOMAINS:
            m = raid_m & (domains == dom)
            if m.sum() < 50 or len(np.unique(labels[m])) < 2:
                continue
            ba = balanced_accuracy_score(labels[m], pred[m]) * 100
            flag = '  <<<' if dom in ('reviews', 'poetry') else ''
            p(f'    {"  "+dom:<16}{int(m.sum()):>9,}{ba:>8.2f}%{flag}')
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    p('=' * 100)
    p('  DeBERTa-ConPara v2.17 — 1M leakage-free stratified dataset')
    p(f'  Backbone={BACKBONE} | hidden={HIDDEN} | '
      f'batch={BATCH}x{ACCUM}={BATCH*ACCUM}eff | lr={LR} | seed={SEED}')
    p(f'  Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | Device: {DEVICE}')
    p('=' * 100)

    # ---------------------------------------------------- PHASE 1: load --
    p('\n' + '=' * 100)
    p('  PHASE 1: Load pre-built splits')
    p('=' * 100)
    tr = load_split('train')
    va = load_split('val')

    for nm, d in (('train', tr), ('val', va)):
        lab = d['labels']
        p(f'  {nm:<6} n={len(lab):>9,}  H={int((lab==0).sum()):>8,}  '
          f'AI={int((lab==1).sum()):>8,}')
        cnt = defaultdict(int)
        for s in d['sources']:
            cnt[str(s)] += 1
        for s, n in sorted(cnt.items(), key=lambda x: -x[1]):
            p(f'    {s:<18}{n:>9,}  ({n/len(lab)*100:.1f}%)')

    # -------------------------------------------- PHASE 2: scale + MI ---
    p('\n' + '=' * 100)
    p('  PHASE 2: Refit RobustScaler + MI feature selection on v2.17 train')
    p('=' * 100)

    X_tr = tr['features'].astype(np.float32)
    y_tr = tr['labels'].astype(int)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler()
    scaler.fit(X_tr)
    Xs = scaler.transform(X_tr)
    p(f'  scaler fit on {len(Xs):,} x {Xs.shape[1]} features')

    idx = np.random.RandomState(SEED).choice(
        len(Xs), min(MI_SAMPLE, len(Xs)), replace=False)
    p(f'  MI on {len(idx):,} samples...')
    mi = mutual_info_classif(Xs[idx], y_tr[idx], random_state=SEED)
    mi_idx = np.argsort(mi)[::-1][:N_FEATURES]
    p(f'  Selected {N_FEATURES}: {sorted(mi_idx.tolist())}')
    p(f'  MI range: {mi[mi_idx].min():.4f} .. {mi[mi_idx].max():.4f}')

    X_tr_sel = Xs[:, mi_idx]
    X_va = np.nan_to_num(va['features'].astype(np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    X_va_sel = scaler.transform(X_va)[:, mi_idx]
    y_va = va['labels'].astype(int)

    if args.dry_run:
        p('\n  DRY RUN complete. Run without --dry_run to train.')
        flush_log()
        return

    # ------------------------------------------------- PHASE 3: train ----
    p('\n' + '=' * 100)
    p('  PHASE 3: Training')
    p('=' * 100)

    tok = DebertaV2Tokenizer.from_pretrained(BACKBONE)
    model = ConParaDetectorLarge(nf=N_FEATURES).to(DEVICE)
    n_par = sum(x.numel() for x in model.parameters())
    p(f'  Params: {n_par:,}')

    ds_tr = TextDS(tr['texts'].tolist(), X_tr_sel, y_tr, tok)
    ds_va = TextDS(va['texts'].tolist(), X_va_sel, y_va, tok)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH * 4, shuffle=False,
                       num_workers=4, pin_memory=True)

    steps = len(dl_tr) // ACCUM * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.06), steps)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_W, dtype=torch.float32).to(DEVICE))
    amp = GradScaler()

    p(f'  steps/epoch={len(dl_tr)//ACCUM:,}  total={steps:,}')
    p(f'  val batches={len(dl_va):,} ({len(y_va):,} samples)')

    best_ba, bad = 0.0, 0
    ckpt_path = MODEL_DIR / f'4ds_deberta_v3_large_v217_{stamp}_s{SEED}.pt'

    for ep in range(1, args.epochs + 1):
        p(f'\n  -- Epoch {ep}/{args.epochs} --')
        model.train()
        run_loss, run_corr, run_n = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        for step, (iids, amask, feats, y) in enumerate(
                tqdm(dl_tr, ncols=70, desc=f'  Ep{ep}')):
            iids = iids.to(DEVICE, non_blocking=True)
            amask = amask.to(DEVICE, non_blocking=True)
            feats = feats.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast():
                logits, _ = model(iids, amask, feats)
                loss = crit(logits, y) / ACCUM
            amp.scale(loss).backward()

            if (step + 1) % ACCUM == 0:
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt)
                amp.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            run_loss += loss.item() * ACCUM * len(y)
            run_corr += (logits.argmax(1) == y).sum().item()
            run_n += len(y)

        p(f'    Train: loss={run_loss/run_n:.4f}  acc={run_corr/run_n:.4f}'
          f'  ({(time.time()-t0)/3600:.2f}h)')

        probs, labs = evaluate(model, dl_va)
        ba = report_val(probs, labs, va['sources'], va['domains'])

        # always snapshot this epoch so selection can be revisited later
        torch.save({
            'model_state_dict': model.state_dict(),
            'mi_selected_indices': mi_idx,
            'scaler_center': scaler.center_,
            'scaler_scale': scaler.scale_,
            'epoch': ep, 'val_bal_acc': ba, 'version': 'v2.17',
        }, MODEL_DIR / f'epoch{ep}_{stamp}_s{SEED}.pt')

        if ba > best_ba:
            best_ba, bad = ba, 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'mi_selected_indices': mi_idx,
                'scaler_center': scaler.center_,
                'scaler_scale': scaler.scale_,
                'epoch': ep,
                'val_bal_acc': ba,
                'version': 'v2.17',
                'n_train': len(y_tr),
                'n_val': len(y_va),
            }, ckpt_path)
            p(f'    * New best: {ba:.2f}%  -> {ckpt_path.name}')
        else:
            bad += 1
            p(f'    No improvement ({bad}/{PATIENCE})')
            if bad >= PATIENCE:
                p('    Early stopping.')
                break
        flush_log()

    # -------------------------------------------- PHASE 4: threshold ----
    p('\n' + '=' * 100)
    p('  PHASE 4: Threshold selection on val')
    p('=' * 100)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.to(DEVICE)
    probs, labs = evaluate(model, dl_va)

    best_t, best_v = 0.5, 0.0
    for t in np.arange(0.05, 0.99, 0.01):
        v = balanced_accuracy_score(labs, (probs >= t).astype(int))
        if v > best_v:
            best_v, best_t = v, t
    p(f'  tau* = {best_t:.2f}   val BA = {best_v*100:.2f}%')
    p(f'  (tau=0.50 gives '
      f'{balanced_accuracy_score(labs,(probs>=0.5).astype(int))*100:.2f}%)')

    ck['threshold'] = float(best_t)
    torch.save(ck, ckpt_path)

    p('\n  Final per-source at tau*:')
    report_val(probs, labs, va['sources'], va['domains'], thr=best_t)

    p(f'\n  Checkpoint: {ckpt_path}')
    p(f'  Finished: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    p('=' * 100)
    flush_log()


if __name__ == '__main__':
    main()
