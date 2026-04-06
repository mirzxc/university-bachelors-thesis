---
type: literature_note
repo: thesis-elm
status: active
reading_status: to_read
priority: high
recommended_order: 1
authors:
  - Guang-Bin Huang
  - Qin-Yu Zhu
  - Chee Kheong Siew
year: 2006
venue: Neurocomputing
topic:
  - elm
  - theory
  - classification
paper_link: https://doi.org/10.1016/j.neucom.2005.12.126
doi: 10.1016/j.neucom.2005.12.126
model_focus:
  - elm
tags:
  - literature
  - thesis
  - elm
  - foundation
---

# Extreme learning machine: Theory and applications

## Why read now

- Foundational ELM paper. This is the minimum required citation for explaining why the repo includes a closed-form random-feature baseline.

## Claim

- Introduces ELM for single-hidden-layer feedforward networks with randomly chosen hidden parameters and analytically solved output weights.

## Method

- Batch learning on SLFNs with a frozen hidden layer and least-squares solve for the output layer.

## Evidence

- Reports approximation and classification experiments showing strong training-speed gains against conventional iterative neural training.

## Relevance to this repo

- Direct background for [elm.py](../../../src/thesis_elm/models/elm.py).
- Useful for the theory chapter and for motivating why `L` is the main structural hyperparameter in this repository.

## Questions to answer while reading

- Which assumptions about activation functions are required by the theory?
- Which claims are proved formally and which are only empirical?
- How strong is the evidence on classification tasks similar to the thesis setup?

## Quotes or paraphrases

- 

## Follow-up

- Compare the paper's closed-form formulation with the implementation in [modeling_guide.md](../../../docs/modeling_guide.md).
