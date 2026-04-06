---
type: literature_note
repo: thesis-elm
status: active
reading_status: to_read
priority: medium
recommended_order: 6
authors:
  - Marc Masana
  - Xialei Liu
  - Bartlomiej Twardowski
  - Mikel Menta
  - Andrew D. Bagdanov
  - Joost van de Weijer
year: 2023
venue: IEEE Transactions on Pattern Analysis and Machine Intelligence
topic:
  - class-incremental-learning
  - continual-learning
  - evaluation
paper_link: https://doi.org/10.1109/TPAMI.2022.3213473
doi: 10.1109/TPAMI.2022.3213473
model_focus:
  - os_elm
tags:
  - literature
  - thesis
  - continual-learning
  - dynamic-scenario
---

# Class-Incremental Learning: Survey and Performance Evaluation on Image Classification

## Why read now

- Not ELM-specific, but useful for designing the thesis dynamic scenario and for avoiding a naive continual-learning evaluation.

## Claim

- Surveys class-incremental learning methods and compares them under multiple evaluation scenarios, with emphasis on catastrophic forgetting and protocol design.

## Method

- Survey plus broad empirical comparison of class-incremental setups and metrics.

## Evidence

- Strong on evaluation design and scenario framing; less directly relevant to tabular models than the core ELM papers.

## Relevance to this repo

- Useful for shaping the `sequential` experiment logic in [experiments.py](../../../src/thesis_elm/experiments.py).
- Helps define what should be measured and discussed in the dynamic scenario chapter even if the paper itself is image-focused.

## Questions to answer while reading

- Which evaluation metrics transfer well from image class-incremental learning to tabular classification?
- Which parts are worth adapting and which are domain-specific noise?
- How should forgetting or distribution shift be measured in this thesis?

## Quotes or paraphrases

- 

## Follow-up

- Use this as a checkpoint before freezing the final sequential experiment protocol.
