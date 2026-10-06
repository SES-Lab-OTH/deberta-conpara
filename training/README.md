# Training: the v2.18 pipeline behind the released checkpoint

The released checkpoint (`rawguard.pt` on the Hub, epoch 2 of run
`20260819_181215`, seed 42) was produced by the scripts in this folder,
**published exactly as they ran** on our cluster. They are not refactored: the
point of this folder is provenance, so paths and file names are those of the
original run (a `~/Text` working directory). Each file's md5 and last
modification time are listed in the manifest below.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `v218/extract_v218_raid.py` | RAID AI rows, weighted by generator, attack and decoding difficulty (`raid_ai_v218*_62features.npz`) |
| 2 | `v218/extract_human_filler.py` | balanced human filler, one fifth from each of the five M4 human CSVs |
| 3 | `v218/attach_adv_ids_v218.py` | RAID source ids on the AI rows (exact text match), needed for a leakage-free split |
| 4 | `v218/build_v218_splits.py` | `features/v218_raw/v218_{train,val}_62features.npz` (deterministic, no RNG; grouped by RAID `source_id`) |
| 5 | `train_conpara_v218_nofeat.py` | the released model |

The raw corpus (no Unicode normalisation at training time, the setting of the
released model) is built with `--no-preprocess` in steps 1 and 2, `--raw` in
steps 3 and 4. The released model was trained with

```bash
python train_conpara_v218_nofeat.py --raw
```

Each script documents its design decisions in its docstring; the extractors
and the split builder print their plan before writing (`--plan`).

**Inputs.** RAID train via the `raid-bench` package (`raid.utils.load_data`);
the HC3 Plus, MAGE, M4, PeerRead and WikiHow feature caches from earlier
pipeline versions (see `legacy_v217/`); the M4 human CSVs. The scripts import
the normaliser as `src.unicode_preprocessing_v2` (this repository's
`src/unicode_preprocessing_v2.py`, md5 `b9adbe2b56d6f2905df809d82f2befff`) and
the 62-feature extractor as `train_conpara_v24.FeatureExtractor`, published here
as `src/features.py`. The features are stored in the corpus for the ablations;
the released no-feature model reads only text and labels (its script still
fits the RobustScaler and MI selection and stores them in the checkpoint, where
they are unused).

## The released run

| | |
|---|---|
| Corpus | `features/v218_raw`: train 1,549,358 (774,679 per class), val 172,826 (86,413 per class) |
| Model | `microsoft/deberta-v3-large`, CLS → Linear(1024, 512) → GELU → Dropout(0.1) → Linear(512, 2) |
| Optimisation | AdamW, LR 1e-5, weight decay 0.01, linear schedule with 6 % warm-up, batch 4 × 8 accumulation (effective 32), max length 512, class weights 1.5 (human) / 1.0 (AI), mixed precision |
| Schedule | 5 epochs, 48,417 steps per epoch, about 14.5 h per epoch on one GPU |
| Selection | every epoch saved; **epoch 2** chosen on validation TPR@1 % FPR (91.78; BA* 96.20, AUROC 0.99400, threshold margin 2.942). Balanced accuracy peaked at epoch 5 (96.51); TPR@1 % is the deployment metric. |

The full log and per-epoch history are in `results/training_logs/`.

## The factorial of the paper

`train_conpara_v218.py` is the same recipe with the gated feature-fusion
branch (30 MI-selected features); the two scripts differ only in the model
class, the classifier input width and output paths. Each runs on the
normalised corpus (default) or the raw corpus (`--raw`). Inference-time
normalisation, the third factor, is applied at evaluation
(`evaluation/eval_cells_external.py`).

## Manifest

| File | md5 (first 12) | Last modified (UTC) |
|---|---|---|
| `v218/extract_v218_raid.py` | `237bfdf6f92b` | 2026-08-14 16:00 |
| `v218/extract_human_filler.py` | `7210edff0da6` | 2026-08-14 16:06 |
| `v218/attach_adv_ids_v218.py` | `b6d6897f66b4` | 2026-08-14 15:42 |
| `v218/build_v218_splits.py` | `14b5cb76a62f` | 2026-08-14 21:38 |
| `train_conpara_v218_nofeat.py` | `6c4181016d09` | 2026-08-18 22:46 |
| `train_conpara_v218.py` | `0f091d95de27` | 2026-08-14 15:42 |
| `../results/training_logs/train_v218_nofeat_raw_log.txt` | `8df73e295229` | 2026-08-22 21:39 |
| `../results/training_logs/history_20260819_181215.json` | `aebc91c3a9cc` | 2026-08-22 21:39 |

Timeline check: the raw corpus was written on 2026-08-15 18:09, after the last
modification of every builder, and the released run started on 2026-08-19
18:12, after the last modification of its training script. The *normalised*
corpus (`features/v218`, used only by the normalised-training cells of the
factorial) was written on 2026-08-12, before the builders' final edits of
2026-08-14; those cells may therefore have been built by an earlier revision
of the same scripts. The raw validation file has md5
`0a02624e0504f55bf2179ff55bde2bef`.

## Legacy

`legacy_v217/` holds the v2.17 pipeline (corpus extraction, split builder and
the feature-branch training script) that preceded v2.18. It is kept because
the v2.18 builders reuse caches it produced; it does not reproduce the
released model.
