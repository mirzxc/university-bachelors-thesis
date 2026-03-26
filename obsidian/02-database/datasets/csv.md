---
type: dataset
repo: thesis-elm
status: active
dataset_kind: user_csv
cli_name: csv
source_files:
  - "../../../src/thesis_elm/data.py"
tags:
  - dataset
  - csv
  - custom-data
---

# CSV Dataset

## Status

User-supplied tabular classification data loaded from a CSV file.

## Required arguments

- `--dataset csv`
- `--csv-path <path>`
- `--target-column <column>`

## Notes

- Labels are encoded with `LabelEncoder`.
- The dataset name in result rows becomes the CSV stem.

## Example command

```bash
uv run thesis-elm run --model elm --dataset csv --csv-path data/my_dataset.csv --target-column label --L 500
```
