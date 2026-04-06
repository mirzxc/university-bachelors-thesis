---
type: literature_note
repo: thesis-elm
status: active
reading_status: to_read
priority: high
recommended_order: 2
authors:
  - Nan-Ying Liang
  - Guang-Bin Huang
  - P. Saratchandran
  - N. Sundararajan
year: 2006
venue: IEEE Transactions on Neural Networks
topic:
  - os-elm
  - online-learning
  - sequential-updates
paper_link: https://doi.org/10.1109/TNN.2006.880583
doi: 10.1109/TNN.2006.880583
model_focus:
  - os_elm
tags:
  - literature
  - thesis
  - os-elm
  - foundation
---

# A Fast and Accurate Online Sequential Learning Algorithm for Feedforward Networks

## Why read now

- This is the key paper for the dynamic part of the thesis because it defines OS-ELM and its online update rule.

## Claim

- Proposes OS-ELM as a sequential learning algorithm for SLFNs that can update one sample or one chunk at a time while keeping the ELM philosophy.

## Method

- Initializes on a first batch and then updates the model recursively on incoming data instead of retraining from scratch.

## Evidence

- Evaluates regression, classification, and time-series tasks against other sequential learning algorithms.

## Relevance to this repo

- Direct background for [os_elm.py](../../../src/thesis_elm/models/os_elm.py).
- Important for justifying `partial_fit`, `initial_batch_size`, and sequential experiment design in [experiments.py](../../../src/thesis_elm/experiments.py).

## Questions to answer while reading

- What exact matrix update is used for the recursive solve?
- What assumptions does the method make about chunk size and initialization?
- Which evaluation setup can be adapted for the thesis dynamic scenario?

## Quotes or paraphrases

- 

## Follow-up

- Map the notation from the paper to [modeling_guide.md](../../../docs/modeling_guide.md) and the sequential CLI workflow in [experiments.md](../../../docs/experiments.md).
