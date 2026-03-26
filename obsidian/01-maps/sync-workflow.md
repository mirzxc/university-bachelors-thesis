---
type: map
area: workflow
repo: thesis-elm
status: active
source_files:
  - "../../scripts/obsidian_sync_on_save.sh"
tags:
  - obsidian
  - git
  - sync
---

# Sync Workflow

## What is configured

- The vault uses `03-templates` as the templates folder.
- Daily Notes are disabled.
- Obsidian Sync is disabled because this repo uses GitHub-based sync instead.

## Save-to-GitHub flow

The watcher script at [scripts/obsidian_sync_on_save.sh](../../scripts/obsidian_sync_on_save.sh) watches the `obsidian/` vault for file saves and then:

1. stages only `obsidian/`
2. creates a commit with a timestamped message
3. rebases on top of the current remote branch
4. pushes to `origin`

## Scope

Only changes inside `obsidian/` are auto-committed. Code changes elsewhere in the repo stay under manual Git control.

## Operational note

If the remote branch changes in a conflicting way, the script logs the failure and stops pushing until the Git conflict is resolved manually.
