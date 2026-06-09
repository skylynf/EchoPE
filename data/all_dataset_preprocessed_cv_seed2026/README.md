# All-Dataset Preprocessed 5-Fold CV Splits

- Source data: `data/all-dataset/dataset/preprocessed`
- Source manifest: `data/all-dataset/dataset/preprocessed/manifest_preprocessed.csv`
- Rows: `4069`
- Seed: `2026`
- Folds: `5`
- Validation fraction of total dataset per fold: `0.1`
- Stratification key: `label + parsed_view`
- Training CSV format: `path,label` with `normal=0`, `pe=1`

Regenerate with:

```bash
python script/make_preprocessed_cv_splits.py
```

Use a fold in training by overriding the Hydra data CSVs, for example:

```bash
python src/train.py experiment=1_baseline/resnet3d \
  data.train_csv=data/splits/all_dataset_preprocessed_cv_seed2026/fold_0/train.csv \
  data.val_csv=data/splits/all_dataset_preprocessed_cv_seed2026/fold_0/val.csv \
  data.test_csv=data/splits/all_dataset_preprocessed_cv_seed2026/fold_0/test.csv
```
