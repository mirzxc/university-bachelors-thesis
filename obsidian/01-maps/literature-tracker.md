---
type: map
area: literature
repo: thesis-elm
status: active
tags:
  - literature
  - map
  - reading-plan
---

# Literature Tracker

## Reading status legend

- `to_read`: selected and queued
- `reading`: currently reading and extracting notes
- `read`: finished and summarized
- `skimmed`: only quick pass done

## Recommended first pass

1. [[../02-database/literature/huang-2006-extreme-learning-machine-theory-and-applications]]
2. [[../02-database/literature/liang-2006-online-sequential-learning-algorithm]]
3. [[../02-database/literature/huang-2012-elm-regression-and-multiclass-classification]]
4. [[../02-database/literature/huang-et-al-2015-trends-in-extreme-learning-machines-review]]
5. [[../02-database/literature/huang-2014-insight-into-extreme-learning-machines]]
6. [[../02-database/literature/masana-et-al-2023-class-incremental-learning-survey]]

## High-priority papers for now

- [ ] [[../02-database/literature/huang-2006-extreme-learning-machine-theory-and-applications]]
- [ ] [[../02-database/literature/liang-2006-online-sequential-learning-algorithm]]
- [ ] [[../02-database/literature/huang-2012-elm-regression-and-multiclass-classification]]

## Why these papers

- The first three cover the core thesis objects directly: ELM, OS-ELM, and multiclass classification.
- The review papers help frame the theory chapter and position the thesis relative to existing variants.
- The class-incremental survey helps when defining the dynamic scenario and evaluation protocol.

## Workflow

1. Duplicate [[../03-templates/literature-note]] for every newly selected paper.
2. Set `reading_status` and `priority` in the note frontmatter immediately.
3. When finished, fill in Claim, Method, Evidence, Relevance, and Questions.
4. Link the paper into experiment notes or thesis chapter notes once it becomes relevant.
