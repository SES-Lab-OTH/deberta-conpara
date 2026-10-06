# Legacy: the v2.17 pipeline

These scripts preceded v2.18 and are kept for transparency, because the v2.18
builders reuse feature caches that this pipeline produced. They do **not**
reproduce the released checkpoint; see `../README.md` for the v2.18 pipeline.

| File | Original name | What it is |
|---|---|---|
| `extract_sources.py` | v2.17 stage 1 | extracts the per-source feature caches with metadata |
| `build_splits.py` | `build_v217_splits_v2.py` | v2.17 split assembly, grouped by RAID `adv_source_id` (v2.18 groups by `source_id`, which also keeps generations of a validation article out of training) |
| `train_conpara.py` | `train_conpara_v217.py` | v2.17 training with the gated feature branch, on `features/v217` |
