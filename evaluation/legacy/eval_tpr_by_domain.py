"""Per-domain TPR@1% and TPR@5% on v2.17 val — the metric that matters."""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
sys.path.insert(0, os.environ.get('CONPARA_ROOT',
                                  str(Path(__file__).resolve().parent.parent)))
from train_conpara_v217 import (ConParaDetectorLarge, TextDS, load_split,
                                evaluate, RAID_DOMAINS, BACKBONE, N_FEATURES)
from transformers import DebertaV2Tokenizer

import os
ck_path = Path(os.environ.get('CKPT',
    str(sorted((Path.home()/'Text/models/conpara_v217').glob('*.pt'))[-1])))
ck = torch.load(ck_path, map_location='cpu', weights_only=False)
print(f"checkpoint: {ck_path.name}  epoch={ck.get('epoch')}  "
      f"val_BA={ck.get('val_bal_acc',0):.2f}%\n")

va = load_split('val')
sc = RobustScaler(); sc.center_ = ck['scaler_center']; sc.scale_ = ck['scaler_scale']
X = np.nan_to_num(va['features'].astype(np.float32), nan=0., posinf=0., neginf=0.)
X = sc.transform(X)[:, ck['mi_selected_indices']]

model = ConParaDetectorLarge(nf=N_FEATURES).cuda()
model.load_state_dict(ck['model_state_dict']); model.eval()
tok = DebertaV2Tokenizer.from_pretrained(BACKBONE)
dl = DataLoader(TextDS(va['texts'].tolist(), X, va['labels'], tok),
                batch_size=32, shuffle=False, num_workers=4)
probs, labs = evaluate(model, dl)

def tpr_at(p, y, fpr):
    """Threshold set on humans to hit target FPR, then TPR on AI."""
    h = p[y == 0]
    if len(h) < 20: return None, None
    t = np.quantile(h, 1 - fpr)
    ai = p[y == 1]
    return (ai >= t).mean()*100, t

src, dom = va['sources'], va['domains']
raid = np.isin(src, ['RAID_human','RAID_AI'])

print(f"{'RAID domain':<14}{'n_H':>7}{'n_AI':>8}{'TPR@5%':>9}{'TPR@1%':>9}{'gap':>8}")
print("-"*56)
for d in RAID_DOMAINS:
    m = raid & (dom == d)
    if m.sum() < 100: continue
    p5,_ = tpr_at(probs[m], labs[m], 0.05)
    p1,_ = tpr_at(probs[m], labs[m], 0.01)
    if p5 is None: continue
    flag = '  <<<' if d in ('reviews','poetry') else ''
    print(f"{d:<14}{int((labs[m]==0).sum()):>7,}{int((labs[m]==1).sum()):>8,}"
          f"{p5:>8.1f}%{p1:>8.1f}%{p5-p1:>7.1f}pp{flag}")

p5,_ = tpr_at(probs[raid], labs[raid], 0.05)
p1,_ = tpr_at(probs[raid], labs[raid], 0.01)
print("-"*56)
print(f"{'RAID overall':<14}{'':>7}{'':>8}{p5:>8.1f}%{p1:>8.1f}%{p5-p1:>7.1f}pp")
print(f"\nv2.16 on RAID hidden test: TPR@5%=98.88%  TPR@1%=87.20%")
print("(val-set numbers, not directly comparable — but the reviews/poetry")
print(" gap should shrink if attack augmentation is working)")
