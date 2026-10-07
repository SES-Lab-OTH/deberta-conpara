# DeBERTa-ConPara: robust detection of AI-generated text

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-deberta--conpara-blue)](https://huggingface.co/SES-Lab-OTH/deberta-conpara)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-orange)](https://huggingface.co/spaces/mohamedmady/deberta-conpara)
[![Paper](https://img.shields.io/badge/Paper-AACL--IJCNLP%202026-1f6feb)](https://arxiv.org/abs/2610.00883)
[![arXiv](https://img.shields.io/badge/arXiv-2610.00883-b31b1b)](https://arxiv.org/abs/2610.00883)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23198565.svg)](https://doi.org/10.5281/zenodo.23198565)

**Paper:** [DeBERTa-ConPara: Attack-Aware and Deployment-Realistic Detection of AI-Generated Text](https://arxiv.org/abs/2610.00883),
AACL-IJCNLP 2026 (main conference). Developed at the [Smart Embedded Systems Lab](https://github.com/SES-Lab-OTH), OTH Regensburg.

DeBERTa-ConPara is a DeBERTa-v3-large detector for machine-generated text, trained on a
1.55M-document leakage-free corpus and hardened against the adversarial edits
that break most detectors: homoglyph substitution, zero-width insertions,
whitespace and typographic attacks.

Its central finding is that **Unicode normalisation acts in opposite directions
depending on where you apply it**. Normalising the *training* corpus silently
deduplicates it: 35.4% of RAID rows collapse into byte-identical copies of their
clean siblings, deleting exactly the adversarial supervision the model needs.
Normalising at *inference* is an effective defence. DeBERTa-ConPara is the cell of a
complete 2×2×2 factorial (feature branch × train-time normalisation ×
inference-time normalisation) that trains on raw text and normalises only at
inference. All eight cells were submitted individually to the RAID hidden test.

## Results

**RAID hidden test** (672,000 documents, 11 adversarial attacks):

| metric | DeBERTa-ConPara |
|---|---|
| AUROC | 99.61 |
| TPR @ 5% FPR | 99.01 |
| TPR @ 1% FPR | 96.57 |

**Against every RAID leaderboard system that publishes a checkpoint.** Balanced
accuracy at a single threshold calibrated once on our source validation split
and then held fixed; RAID column is TPR @ 5% FPR from the leaderboard.

| system | HC3-QA | HC3-SI | MAGE | avg | M4 | RAID |
|---|---|---|---|---|---|---|
| **DeBERTa-ConPara** | **99.69** | **83.50** | **96.23** | **93.14** | **98.27** | 99.01 |
| MELD | 94.64 | 66.86 | 96.17 | 85.89 | 94.26 | **99.78** |
| ModernBERT (raid-mage) | 95.40 | 52.54 | 93.51¹ | 80.48 | 84.07 | 94.14 |
| Desklib v1.01 | 97.87 | 56.38 | 83.44 | 79.23 | 90.56 | 91.17 |
| SuperAnnotate | 99.04 | 56.25 | 60.01 | 71.77 | 83.08 | 64.87 |
| e5-small-lora | 87.29 | 59.45 | 67.12 | 71.29 | 78.66 | 85.69 |
| TMR | 84.85 | 55.92 | 70.99 | 70.59 | 79.77 | 95.79 |
| BERT-tiny-4M | 69.84 | 55.60 | 61.49 | 62.31 | 68.98 | 84.18 |
| ADAL | 59.89 | 44.89 | 61.23 | 55.34 | 65.62 | 96.25 |
| RADAR | 53.32 | 49.09 | 60.40 | 54.27 | 58.25 | 63.91 |

¹ ModernBERT's released checkpoint was trained on MAGE, so that column is
in-distribution for it.

**Read these numbers with two caveats.** First, MELD leads DeBERTa-ConPara on RAID
itself (99.78 vs 99.01); on adversarial robustness alone it is the stronger
open system. Second, HC3, MAGE and M4 are sources in DeBERTa-ConPara's own training
corpus, so those columns are held-out splits for DeBERTa-ConPara but genuinely external
data for every other system. The comparison shows how far each detector travels
from *its* training distribution to *ours*, which favours DeBERTa-ConPara by
construction. Several commercial systems score above 99 on RAID but publish no
checkpoint and cannot be evaluated anywhere else.

Reproduce the table with `evaluation/eval_competitors_external.py`; it
downloads each competitor from the Hub and applies the identical protocol.

## Install

```bash
git clone https://github.com/SES-Lab-OTH/deberta-conpara.git
cd deberta-conpara
pip install -r requirements.txt          # inference only
pip install -r requirements-dev.txt      # + reproduction of the paper's tables
```

## Use it

```python
from src.conpara import ConPara

det = ConPara.from_pretrained()          # ~1.7 GB on first call
print(det.score(["The quick brown fox ..."]))     # logit margin, higher = machine
print(det.predict(["The quick brown fox ..."]))   # bool, at the stored threshold
print(det.band(["The quick brown fox ..."]))      # coarse verbal band
```

From the command line:

```bash
python src/conpara.py --text "paste a document here"
python src/conpara.py --file documents.txt --jsonl scores.jsonl
```

Inference-time Unicode normalisation is applied by default and is what makes the
detector robust to homoglyph and zero-width attacks. Turning it off
(`--no-normalise`) reproduces the undefended condition from the ablation.

## How to read a score

The output is a **logit margin**, not a probability. The model is badly
calibrated at the extremes, so a large margin does not mean a high probability of
being machine-written. Use the stored threshold, or the coarse bands from
`det.band()`, and treat the result as one piece of evidence for a human decision.

Known limits, measured:

- **Short text is unreliable.** Below ~60 words errors are enriched 4–5×. The CLI
  warns below 75 words and the demo refuses below 25.
- **Academic prose draws false positives** at rates between 13% and 67%
  depending on the subcorpus. Do not use this to accuse a student.
- **English only.** The training corpus is English; other languages are untested.
- **Generators move.** Detectors decay as new models appear; a 2026 checkpoint
  is not a permanent instrument.

## What is in this repository

```
src/          conpara.py (inference), unicode_preprocessing_v2.py (the normaliser),
              features.py (the 30/62-feature extractors used by the ablations)
training/     the v2.18 pipeline behind the released checkpoint, published as run
              (README with commands, settings and md5 manifest); legacy_v217/
evaluation/   the fixed-threshold protocol, competitor evaluation, RAID submission
              (byte-identical to the runs behind the paper); legacy/ older versions
results/      per-cell metrics behind the paper's tables; training_logs/ of the
              released run
figures/      figure sources, regenerable
docs/         the protocol in prose, and the negative results
```

Two negative results are included deliberately, because they cost real compute
and the field should not repeat them: paraphrase augmentation with supervised
contrastive learning (ConPara) does not help, and the 30-feature fusion branch is
inert in distribution and harmful outside it, including a 49.8-point loss on RAID
poetry.

## Questions and feedback

Questions about using the model, reproducing the paper or the data are welcome in
[Discussions](https://github.com/SES-Lab-OTH/deberta-conpara/discussions). For bugs,
please open an [issue](https://github.com/SES-Lab-OTH/deberta-conpara/issues).

## Citation

```bibtex
@inproceedings{mady2026conpara,
  title         = {{DeBERTa-ConPara}: Attack-Aware and Deployment-Realistic
                   Detection of {AI}-Generated Text},
  author        = {Mady, Mohamed and Li, Yupei and Reschke, Johannes and Schuller, Bj\"orn W.},
  booktitle     = {Proceedings of the 14th International Joint Conference on Natural
                   Language Processing and the 4th Conference of the Asia-Pacific Chapter
                   of the Association for Computational Linguistics (AACL-IJCNLP 2026)},
  year          = {2026},
  publisher     = {Association for Computational Linguistics},
  eprint        = {2610.00883},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.00883}
}
```

## License

MIT, matching the DeBERTa-v3-large backbone. The released weights are a
derivative of `microsoft/deberta-v3-large`.

## Acknowledgements

Developed at the Smart Embedded Systems Lab, OTH Regensburg, in a cooperative
doctorate with the Technical University of Munich. Built on RAID (Dugan et al.,
ACL 2024), HC3 Plus, MAGE and M4. Compute provided by the RCAI cluster at OTH
Regensburg.
