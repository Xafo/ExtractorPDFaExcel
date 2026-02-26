# Gold Set Manual v1

This folder contains a manual annotation scaffold to optimize extraction for
"filled and correct" values.

## Files

- `goldset_manual_v1.csv`: sampled pages and labeling template.
- `goldset_manual_v1_seed.csv`: same sample prefilled with current model output.
- `pages/`: rendered images (`sample_id.png`) for fast review.
- `batches/`: prioritized annotation batches (`annotation_batch_XX.csv`).

## How to generate

1. Create scaffold CSV:

```bash
python scripts/generate_goldset_scaffold.py
```

2. Export sample pages as PNG:

```bash
python scripts/export_goldset_images.py
```

3. (Optional) prefill with current extractor predictions:

```bash
python scripts/prefill_goldset_seed.py
```

4. Build prioritized manual batches (hardest first):

```bash
python scripts/build_annotation_batches.py
```

## Sampling used

- PDFs with >= 20 pages: 8 samples each (1,2,0.28n,0.50n,0.56n,0.75n,n-1,n).
- PDFs with < 20 pages: 6 samples each (1,2,0.33n,0.58n,n-1,n).
- Current dataset size expected: 94 pages total.

## Manual labeling rules

Annotate these fields in `goldset_manual_v1.csv`:

- `certificado_no`, `marca`, `chasis`, `linea`, `pasajeros`, `modelo`,
  `clase`, `placa`, `motor`, `color`

Conventions:

- Use uppercase.
- Leave empty if unreadable.
- Keep plate normalized (example: `C-633BNT`).
- Keep `chasis` as 17 chars only when it is actually visible.

Workflow columns:

- `reviewer`: your name/initials.
- `status`: `todo`, `in_progress`, `done`, `qa_done`.
- `quality_flag`: optional (`clean`, `difficult`, `illegible`, `low_contrast`).
- `notes`: optional comments.

## Next step after annotation

Run prediction + evaluator to report quality on labeled rows:

```bash
python scripts/predict_on_goldset_sample.py
python scripts/evaluate_against_goldset.py --gold goldset/goldset_manual_v1_seed.csv --pred goldset/goldset_predictions_v1.csv --require-status qa_done --out-mismatches goldset/mismatches_v1.csv
```

Metrics reported:

- Fill rate
- Exact match accuracy per field
- Exact row accuracy (10/10 fields)
- Optional mismatch CSV for targeted fixes
