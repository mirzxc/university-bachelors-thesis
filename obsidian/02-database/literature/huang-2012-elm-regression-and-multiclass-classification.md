---
type: literature_note
repo: thesis-elm
status: active
reading_status: to_read
priority: high
recommended_order: 3
authors:
  - Guang-Bin Huang
  - Hongming Zhou
  - Xiaojian Ding
  - Rui Zhang
year: 2012
venue: IEEE Transactions on Systems, Man, and Cybernetics, Part B
topic:
  - elm
  - multiclass-classification
  - theory
paper_link: https://doi.org/10.1109/TSMCB.2011.2168604
doi: 10.1109/TSMCB.2011.2168604
model_focus:
  - elm
tags:
  - literature
  - thesis
  - elm
  - multiclass
---

# Extreme Learning Machine for Regression and Multiclass Classification

## Why read now

- The thesis is about tabular classification, so this paper is useful for the multiclass case beyond the original 2006 formulation.

## Claim

- Positions ELM as a general learning framework that also handles regression and multiclass classification directly.

## Method

- Extends the ELM framework and discusses its relation to LS-SVM and related regularization-based formulations.

## Evidence

- Reports experiments suggesting favorable scalability and strong performance, especially in multiclass settings.

## Relevance to this repo

- Useful when describing why ELM is a plausible baseline on datasets like `iris`, `wine`, and `digits`.
- Connects directly to the thesis comparison between classical baselines and ELM-based models.

## Questions to answer while reading

- Which multiclass coding strategy is assumed?
- What theoretical or optimization arguments are most defensible in the thesis text?
- Which claims remain relevant when using standardized tabular datasets?

## Quotes or paraphrases

- 

## Follow-up

- Use it when drafting the model-comparison chapter and interpreting multiclass results from [data.py](../../../src/thesis_elm/data.py).
