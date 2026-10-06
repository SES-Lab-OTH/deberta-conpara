"""
Inference API for the DeBERTa-ConPara AI-generated-text detector.

The released checkpoint is the no-feature, raw-trained cell of the 2x2x2
factorial reported in the paper, scored with inference-time Unicode
normalisation ("raw training -> normalised inference"). Architecture:

    DeBERTa-v3-large encoder -> CLS token -> Linear(1024, 512) -> GELU
                             -> Dropout(0.1) -> Linear(512, 2)

The score is the logit margin  logit[AI] - logit[human]  (higher = more
machine-like). It is an unbounded real number, NOT a probability: the model is
poorly calibrated at the extremes, so threshold it rather than reading it as a
confidence. The default threshold is the one stored in the checkpoint
(`threshold_margin`), selected on the source validation split.

Usage
-----
    from conpara import ConPara

    det = ConPara.from_pretrained()            # downloads from the Hub
    scores = det.score(["some text", "another text"])
    flags  = det.predict(["some text"])         # bool, at the stored threshold

CLI
---
    python conpara.py --text "paste a document here"
    python conpara.py --file docs.txt --jsonl out.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:  # the repo's normaliser; falls back to the packaged copy
    from unicode_preprocessing_v2 import unicode_normalize
except ImportError:  # pragma: no cover
    from src.unicode_preprocessing_v2 import unicode_normalize

HF_REPO = "SES-Lab-OTH/deberta-conpara"
HF_FILE = "rawguard.pt"
BACKBONE = "microsoft/deberta-v3-large"
MAX_LEN = 512

# The paper's evaluation normalises then collapses whitespace. Both steps matter:
# the detector is only robust to Unicode attacks when this runs at inference.
import re as _re


def normalise(text: str) -> str:
    """Inference-time preprocessing, identical to the paper's evaluation."""
    return _re.sub(r"\s+", " ", unicode_normalize(str(text))).strip()


class NoFeat(nn.Module):
    """Released architecture. Kept byte-identical to the training definition."""

    def __init__(self, backbone: str = BACKBONE, hs: int = 1024, config_only: bool = False):
        super().__init__()
        if config_only:  # weights come from the checkpoint; skip the download
            from transformers import AutoConfig
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        else:
            self.encoder = AutoModel.from_pretrained(backbone)
        self.classifier = nn.Sequential(
            nn.Linear(hs, hs // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hs // 2, 2)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(cls)


@dataclass
class ConPara:
    model: NoFeat
    tokenizer: object
    threshold: float
    device: str
    meta: dict

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(cls, path: Optional[str] = None, device: Optional[str] = None,
                        repo_id: str = HF_REPO, filename: str = HF_FILE,
                        use_fast_tokenizer: bool = True) -> "ConPara":
        """`path` is a local .pt; without it the checkpoint is fetched from the Hub."""
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        if any(k.startswith("feat_proj") for k in sd):
            raise ValueError("this checkpoint has a feature branch; ConPara is the no-feature cell")
        model = NoFeat(config_only=True)
        model.load_state_dict(sd, strict=True)      # strict: a silent mismatch is a wrong model
        model.to(device).eval()
        tok = AutoTokenizer.from_pretrained(BACKBONE, use_fast=use_fast_tokenizer)
        meta = {k: v for k, v in ck.items()
                if k not in ("model_state_dict", "scaler_center", "scaler_scale",
                             "mi_selected_indices")}
        return cls(model=model, tokenizer=tok, threshold=float(ck.get("threshold_margin", 0.0)),
                   device=device, meta=meta)

    # ----------------------------------------------------------------- score
    @torch.no_grad()
    def score(self, texts: Sequence[str], batch_size: int = 16,
              normalise_input: bool = True, progress: bool = False) -> np.ndarray:
        """Logit margins, one per text, in the order given. Higher = more machine-like."""
        texts = [str(t) for t in texts]
        prepped = [normalise(t) for t in texts] if normalise_input else texts
        order = np.argsort([len(t) for t in prepped])[::-1]     # length-sorted batching
        out = np.empty(len(prepped), dtype=np.float64)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tokenizer([prepped[j] for j in idx], return_tensors="pt",
                                 padding=True, truncation=True, max_length=MAX_LEN).to(self.device)
            enc.pop("token_type_ids", None)
            logits = self.model(enc["input_ids"], enc["attention_mask"]).float()
            out[idx] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            if progress:
                print(f"  {min(i + batch_size, len(order))}/{len(order)}", file=sys.stderr, flush=True)
        return out

    def predict(self, texts: Sequence[str], threshold: Optional[float] = None, **kw) -> List[bool]:
        """True = flagged as machine-generated, at `threshold` (default: the stored one)."""
        t = self.threshold if threshold is None else threshold
        return [bool(s > t) for s in self.score(texts, **kw)]

    def band(self, texts: Sequence[str], **kw) -> List[str]:
        """Coarse verbal bands. Preferred over percentages: scores are not calibrated."""
        t = self.threshold
        labels = []
        for s in self.score(texts, **kw):
            d = s - t
            labels.append("likely machine" if d > 4 else
                          "leaning machine" if d > 0 else
                          "leaning human" if d > -4 else "likely human")
        return labels


# ---------------------------------------------------------------------- CLI
def _read_inputs(a) -> List[str]:
    if a.text:
        return [a.text]
    if a.file == "-":
        return [l.rstrip("\n") for l in sys.stdin if l.strip()]
    with open(a.file, encoding="utf-8") as fh:
        return [l.rstrip("\n") for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="DeBERTa-ConPara AI-generated-text detector")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="a single document")
    g.add_argument("--file", help="one document per line, or - for stdin")
    ap.add_argument("--ckpt", help="local checkpoint (default: download from the Hub)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-normalise", action="store_true",
                    help="skip inference-time Unicode normalisation (not recommended)")
    ap.add_argument("--jsonl", help="write results here instead of stdout")
    a = ap.parse_args()

    det = ConPara.from_pretrained(path=a.ckpt)
    texts = _read_inputs(a)
    scores = det.score(texts, batch_size=a.batch_size, normalise_input=not a.no_normalise)
    thr = det.threshold if a.threshold is None else a.threshold
    rows = [{"score": float(s), "flagged": bool(s > thr), "words": len(t.split()),
             "text": t[:80]} for t, s in zip(texts, scores)]
    for r in rows:
        if r["words"] < 25:
            r["warning"] = "below 25 words: not reliable"
        elif r["words"] < 75:
            r["warning"] = "below 75 words: treat with caution"
    if a.jsonl:
        with open(a.jsonl, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {a.jsonl} ({len(rows)} rows), threshold {thr:+.3f}")
    else:
        for r in rows:
            w = f"  [{r['warning']}]" if "warning" in r else ""
            print(f"{r['score']:+8.3f}  {'MACHINE' if r['flagged'] else 'human  '}  {r['text']!r}{w}")


if __name__ == "__main__":
    main()
