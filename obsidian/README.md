# Obsidian Vault for thesis-elm

This folder is a dedicated Obsidian vault for the repository.

Open `obsidian/` as a vault in Obsidian.

## Layout

- `00-dashboard.md`: entry point and navigation hub
- `01-maps/`: repo-level maps and architecture notes
- `02-database/`: frontmatter-driven notes for models, commands, datasets, docs, and literature
- `03-templates/`: reusable note templates for experiments and literature

## Database approach

The vault uses note properties as the database layer. Each seeded note includes structured frontmatter such as:

- `type`
- `repo`
- `status`
- `source_files`
- `tags`

Type-specific notes add fields such as `cli_name`, `handler`, `dataset_kind`, or `supports_partial_fit`.

## Suggested workflow

1. Open [[00-dashboard]].
2. Use note properties to filter notes by `type`, `area`, or `status`.
3. Duplicate a template from `03-templates/` when you start a new experiment or literature note.
4. Link new notes back to repo files with relative markdown links.
5. Track paper progress through `reading_status` and `priority` so the literature backlog stays actionable.
