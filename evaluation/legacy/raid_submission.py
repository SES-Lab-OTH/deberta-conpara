#!/usr/bin/env python3
"""RAID submission script for DeBERTa-ConPara v2.14 (DeBERTa-v3-large, 1024-dim)."""
import os, sys, re, unicodedata, json, argparse
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2Model
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

TEXT_ROOT = Path(os.environ.get('CONPARA_ROOT',
                                Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(TEXT_ROOT))
from train_conpara_v24 import FeatureExtractor
print("✓ Imported from train_conpara_v24")

BACKBONE_ID  = 'microsoft/deberta-v3-large'
LARGE_HIDDEN = 1024
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Identical model class to train_conpara_v217.py ────────────────────────────
class ConParaDetectorLarge(nn.Module):
    def __init__(self, nf=30, dropout=0.1):
        super().__init__()
        self.encoder   = DebertaV2Model.from_pretrained(BACKBONE_ID)
        hs             = LARGE_HIDDEN
        self.feat_proj = nn.Sequential(nn.Linear(nf, hs), nn.GELU(), nn.Dropout(dropout))
        self.feat_gate = nn.Sequential(nn.Linear(hs*2, hs), nn.Tanh(), nn.Linear(hs, 1), nn.Sigmoid())
        self.classifier= nn.Sequential(nn.Linear(hs*2, hs//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hs//2, 2))
    def forward(self, input_ids, attention_mask, features):
        enc  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls  = enc.last_hidden_state[:, 0, :]
        f_   = self.feat_proj(features)
        gate = self.feat_gate(torch.cat([cls, f_], dim=1))
        logits = self.classifier(torch.cat([cls, gate*f_], dim=1))
        return logits, None

def load_model(model_path):
    ckpt   = torch.load(model_path, map_location='cpu', weights_only=False)
    thr    = 0.98  # v2.14: proven RAID threshold
    mi_idx = ckpt['mi_selected_indices']
    nf     = len(mi_idx)
    scaler = RobustScaler()
    scaler.center_ = ckpt['scaler_center']
    scaler.scale_  = ckpt['scaler_scale']
    model  = ConParaDetectorLarge(nf=nf).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  MI indices: from 'mi_selected_indices' ({nf} features)")
    print(f"  Scaler: loaded ✓ (shape=({len(scaler.center_)},))")
    print(f"  NF={nf} | τ*={thr:.3f} | MI[0:5]={mi_idx[:5]}")
    return model, thr, mi_idx, scaler

try:
    from src.unicode_preprocessing_v2 import unicode_normalize
except ImportError:
    def unicode_normalize(t):
        if not t: return ''
        t = unicodedata.normalize('NFKC', str(t))
        t = re.sub(r'[\u200b-\u200d\ufeff\u2060\u00ad]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

extractor = FeatureExtractor()

class InferenceDataset(Dataset):
    def __init__(self, texts, features, tokenizer, max_len=512):
        self.texts = texts; self.features = features
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i] or '', max_length=self.max_len,
                       truncation=True, padding='max_length', return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'features': torch.tensor(self.features[i], dtype=torch.float32)}

def run_detection_internal(texts, model, scaler, mi_idx, threshold, tokenizer, batch_size=8):
    # Normalize texts — critical for robustness against Unicode attacks
    # (zero_width_space, homoglyph) — must match training-time normalization
    texts = [unicode_normalize(t) for t in tqdm(texts, desc='  Preprocess', ncols=80)]
    raw_feats = extractor.extract_batch(texts, desc='RAID inference')
    scaled    = scaler.transform(raw_feats)[:, mi_idx]
    print(f"  ({len(texts)}, {raw_feats.shape[1]}) → scaled → selected ({len(texts)}, {len(mi_idx)})")
    ds     = InferenceDataset(texts, scaled, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs  = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Inference', ncols=60):
            iids  = batch['input_ids'].to(DEVICE)
            amask = batch['attention_mask'].to(DEVICE)
            feats = batch['features'].to(DEVICE)
            logits, _ = model(iids, amask, feats)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return [1 if p >= threshold else 0 for p in probs], probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=str(
        TEXT_ROOT / 'models/conpara_v217/4ds_deberta_v3_large_v217_20260806_221724_s42.pt'))
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print('='*60)
    print(f'  DeBERTa-ConPara v2.17 — RAID Submission')
    from datetime import datetime; print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Device: {DEVICE}'); print('='*60)

    print(f'  Loading: {Path(args.model_path).name}')
    model, thr, mi_idx, scaler = load_model(args.model_path)

    print('  Loading RAID test set...')
    from raid.utils import load_data
    data = load_data('test')
    if args.test: data = data[:100]; print('  TEST MODE: using first 100 samples')
    print(f'  Test samples: {len(data):,}')

    print(f'\n  Running detection (τ*={thr:.3f})...')

    # Use RAID's run_detection interface (same as v2.6)
    from raid import run_detection
    tokenizer = DebertaV2Tokenizer.from_pretrained(BACKBONE_ID)

    def our_detector(texts):
        _, probs = run_detection_internal(texts, model, scaler, mi_idx, thr, tokenizer)
        return probs

    predictions = run_detection(our_detector, data)

    out_dir   = Path(args.model_path).parent.parent.parent / 'data/raw/raid'
    pred_path = out_dir / 'predictions.json'
    meta_path = out_dir / 'metadata.json'

    with open(pred_path, 'w') as f: json.dump(predictions, f)
    n_ai = sum(1 for p in predictions if p['score'] >= thr)
    print(f'  Predictions → {pred_path}')
    print(f'  Total: {len(predictions):,}')
    print(f'  AI detected: {n_ai:,} ({n_ai/len(predictions)*100:.1f}%) at τ*={thr:.3f}')

    meta = {"date_released":"2026-06-24","detector_name":"DeBERTa-ConPara-v2.14",
            "website":"","paper_link":"","huggingface_link":"","github_link":"",
            "contact_info":"Email Address: mohamed.mady@oth-regensburg.de"}
    with open(meta_path, 'w') as f: json.dump(meta, f, indent=2)
    print(f'  Metadata  → {meta_path}')
    print('='*60)

if __name__ == '__main__':
    main()
