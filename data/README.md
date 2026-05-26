# Data Layout

This repository does not redistribute meme images or restricted benchmark files. Download each dataset from its official source and place the files under`data/raw/<DATASET>/`, then run `scripts/prepare_datasets.py`.

Expected raw layout:

```text
data/raw/
  FHM/
    train.jsonl
    test.jsonl
    images/
  Harm-C/                 # also used when --dataset HarM is passed
    train_v1.jsonl
    val_v1.jsonl
    test_v1.jsonl
    images/
  Harm-P/
    train_v1.jsonl
    val_v1.jsonl
    test_v1.jsonl
    images/
  MultiOFF/
    Training_meme_dataset.csv
    Validation_meme_dataset.csv
    Testing_meme_dataset.csv
    images/
  PrideMM/
    PrideMM.csv
    images/
```

After preparation, the framework expects:

```text
data/<DATASET>/
  train.jsonl
  val.jsonl       # optional
  test.jsonl
  images/
```

`HarM` is treated as an alias for `Harm-C` for compatibility with the paper
notation.
