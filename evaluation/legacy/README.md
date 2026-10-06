# Legacy evaluation scripts

Earlier versions, kept for transparency. They do not produce the numbers in
the paper. The scripts behind the paper's tables are in `../`:
`eval_cells_external.py` (cross-dataset cells and the fixed-threshold
protocol), `eval_competitors_external.py` (competitor comparison) and
`raid_submit_margin.py` (the RAID leaderboard submission); each is
byte-identical to the file that produced the published numbers.

| File | What it is |
|---|---|
| `raid_submission.py` | RAID submission script of v2.14 (feature-branch model) |
| `eval_cross_dataset.py` | cross-dataset evaluation of v2.16 (HC3, MAGE, SemEval, OUTFOX, M4GT-Bench) |
| `eval_tpr_by_domain.py` | per-domain TPR@1 % / 5 % on the v2.17 validation split |
