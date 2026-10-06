from pathlib import Path
#!/usr/bin/env python3
"""Full OOD evaluation of v2.16 on HC3, MAGE, SemEval, OUTFOX, M4GT-Bench"""
import os, sys, json, pickle, torch, torch.nn as nn
import numpy as np, re, unicodedata, pandas as pd
from transformers import DebertaV2Tokenizer, DebertaV2Model
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('high')
DEVICE    = torch.device('cuda')
TEXT_ROOT = os.environ.get('CONPARA_ROOT',
                           str(Path(__file__).resolve().parent.parent))
CKPT      = TEXT_ROOT + '/models/conpara_v216/4ds_deberta_v3_large_v216_20260717_020602_s42.pt'
sys.path.insert(0, TEXT_ROOT)
from train_conpara_v24 import FeatureExtractor

def unicode_normalize(t):
    if not t: return ''
    t = unicodedata.normalize('NFKC', str(t))
    t = re.sub(r'[\u200b-\u200f\u202a-\u202f\u2060-\u2064\ufeff\u00ad]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

class ConParaDetectorLarge(nn.Module):
    def __init__(self, nf=30):
        super().__init__()
        self.encoder   = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        hs = 1024
        self.feat_proj = nn.Sequential(nn.Linear(nf,hs), nn.GELU(), nn.Dropout(0.1))
        self.feat_gate = nn.Sequential(nn.Linear(hs*2,hs), nn.Tanh(), nn.Linear(hs,1), nn.Sigmoid())
        self.classifier= nn.Sequential(nn.Linear(hs*2,hs//2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hs//2,2))
    def forward(self, iids, amask, feats):
        enc=self.encoder(input_ids=iids, attention_mask=amask)
        cls=enc.last_hidden_state[:,0,:]
        f_=self.feat_proj(feats)
        g=self.feat_gate(torch.cat([cls,f_],dim=1))
        return self.classifier(torch.cat([cls,g*f_],dim=1)), None

class InfDS(Dataset):
    def __init__(self, texts, feats, tok, ml=512):
        self.texts=texts; self.feats=feats; self.tok=tok; self.ml=ml
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc=self.tok(self.texts[i] or '',max_length=self.ml,
                     truncation=True,padding='max_length',return_tensors='pt')
        return enc['input_ids'].squeeze(0),enc['attention_mask'].squeeze(0),\
               torch.tensor(self.feats[i],dtype=torch.float32)

print("Loading v2.16...")
ckpt=torch.load(CKPT,map_location='cpu',weights_only=False)
mi_idx=ckpt['mi_selected_indices']
scaler=RobustScaler(); scaler.center_=ckpt['scaler_center']; scaler.scale_=ckpt['scaler_scale']
model=ConParaDetectorLarge(nf=len(mi_idx)).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'],strict=False)
model.eval()
tokenizer=DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
extractor=FeatureExtractor()

def run_eval(texts, labels, desc=''):
    clean=[unicode_normalize(t) for t in texts]
    raw_f=extractor.extract_batch(clean, desc=desc)
    sel_f=scaler.transform(raw_f)[:,mi_idx]
    dl=DataLoader(InfDS(clean,sel_f,tokenizer),batch_size=16,shuffle=False,num_workers=0)
    probs=[]
    with torch.no_grad():
        for iids,amask,feats in tqdm(dl,ncols=65,leave=False):
            logits,_=model(iids.to(DEVICE),amask.to(DEVICE),feats.to(DEVICE))
            probs.extend(torch.softmax(logits,dim=1)[:,1].cpu().tolist())
    probs=np.array(probs); labels=np.array(labels)
    ba5=balanced_accuracy_score(labels,(probs>=0.5).astype(int))*100
    best_ba,best_t=0,0.5
    for t in np.arange(0.01,0.99,0.005):
        b=balanced_accuracy_score(labels,(probs>=t).astype(int))*100
        if b>best_ba: best_ba,best_t=b,t
    auc=roc_auc_score(labels,probs)
    fp=((labels==0)&(probs>=0.5)).sum()
    fn=((labels==1)&(probs<0.5)).sum()
    return ba5, best_ba, best_t, auc, int(fp), int(fn)

DATA = os.path.expanduser('~/Text/data/raw')
results = {}

print('\n' + '='*70)
print('  v2.16 FULL OOD EVALUATION')
print('='*70)
print(f'\n  {"Dataset":<30} {"τ=0.5":>8} {"τ=opt":>8} {"AUROC":>8}')
print('  ' + '─'*60)

# ── HC3 test sets ──────────────────────────────────────────────────────────
for name, csv_p, tc, lc, flip in [
    ('HC3-QA (test)',  f'{DATA}/hc3_plus/test_hc3_QA.csv', 'Text','Is_AI',False),
    ('HC3-SI (test)',  f'{DATA}/hc3_plus/test_hc3_si.csv', 'Text','Is_AI',False),
]:
    df=pd.read_csv(csv_p)
    labels=df[lc].values.astype(int)
    if flip: labels=1-labels
    ba5,best_ba,best_t,auc,fp,fn=run_eval(df[tc].tolist(),labels,name)
    results[name]=(ba5,best_ba,auc)
    print(f'  {name:<30} {ba5:>7.2f}% {best_ba:>7.2f}% {auc:>8.4f}')

# ── MAGE test ──────────────────────────────────────────────────────────────
df=pd.read_csv(f'{DATA}/mage/test.csv')
labels=1-df['label'].values.astype(int)
ba5,best_ba,best_t,auc,fp,fn=run_eval(df['text'].tolist(),labels,'MAGE-test')
results['MAGE (test)']=(ba5,best_ba,auc)
print(f'  {"MAGE (test)":<30} {ba5:>7.2f}% {best_ba:>7.2f}% {auc:>8.4f}')

avg=np.mean([results[k][0] for k in ['HC3-QA (test)','HC3-SI (test)','MAGE (test)']])
print(f'\n  HC3+MAGE average (τ=0.5): {avg:.2f}%')
print(f'  (v2.14: 93.80% | v2.15: 93.51% | v2.16: {avg:.2f}%)')

# ── SemEval 2024 ───────────────────────────────────────────────────────────
print('\n  ' + '─'*60)
semeval_p=os.path.expanduser('~/Text/SemEval2024_gold/subtaskA_monolingual.jsonl')
if os.path.exists(semeval_p):
    texts,labels=[],[]
    with open(semeval_p,encoding='utf-8',errors='replace') as f:
        for line in f:
            try:
                d=json.loads(line)
                texts.append(d['text']); labels.append(int(d['label']))
            except: pass
    ba5,best_ba,best_t,auc,fp,fn=run_eval(texts,labels,'SemEval')
    results['SemEval-2024 mono']=(ba5,best_ba,auc)
    print(f'  {"SemEval-2024 mono":<30} {ba5:>7.2f}% {best_ba:>7.2f}% {auc:>8.4f}')

# ── OUTFOX ─────────────────────────────────────────────────────────────────
outfox_p=os.path.expanduser('~/Text/OUTFOX/data')
for name, fname in [('OUTFOX-clean','outfox_clean.pkl'),
                     ('OUTFOX-attacked','outfox_attacked.pkl')]:
    fp_path=f'{outfox_p}/{fname}'
    if not os.path.exists(fp_path): continue
    with open(fp_path,'rb') as f: d=pickle.load(f)
    texts=d['texts']; labels=np.array(d['labels'])
    ba5,best_ba,best_t,auc,fp,fn=run_eval(texts,labels,name)
    results[name]=(ba5,best_ba,auc)
    print(f'  {name:<30} {ba5:>7.2f}% {best_ba:>7.2f}% {auc:>8.4f}')

# ── M4GT-Bench overall ─────────────────────────────────────────────────────
print('\n  ' + '─'*60)
m4gt_texts,m4gt_labels=[],[]
with open(os.path.expanduser('~/Text/M4GT-Bench/data/SubtaskA.jsonl'),
          encoding='utf-8',errors='replace') as f:
    for line in f:
        try:
            d=json.loads(line)
            m4gt_texts.append(d['text'])
            m4gt_labels.append(int(d['label']))
        except: pass
ba5,best_ba,best_t,auc,fp,fn=run_eval(m4gt_texts,m4gt_labels,'M4GT-Bench')
results['M4GT-Bench']=(ba5,best_ba,auc)
print(f'  {"M4GT-Bench (overall)":<30} {ba5:>7.2f}% {best_ba:>7.2f}% {auc:>8.4f}')
print(f'  (v2.14: 84.62% | v2.15: ~87.5% est | v2.16: {ba5:.2f}%)')

# ── Final summary ──────────────────────────────────────────────────────────
print('\n' + '='*70)
print('  FINAL SUMMARY — v2.16')
print('='*70)
print(f'\n  {"Dataset":<30} {"τ=0.5 BA":>10} {"τ=opt BA":>10} {"AUROC":>8}')
print('  ' + '─'*62)
for name,(ba5,best_ba,auc) in results.items():
    print(f'  {name:<30} {ba5:>9.2f}% {best_ba:>9.2f}% {auc:>8.4f}')
print('\n  DONE ✓')
