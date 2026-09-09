# ChatGPT deliverables export

This branch adds the generated learning materials alongside the repository's existing course.

## Local-to-repository layout

| Original deliverable | Repository location |
|---|---|
| `outputs/robot_learning_loops/` | `robot_learning_loops/` |
| `outputs/START_HERE.md` and three setup/practice scripts | `day0/chatgpt/` |
| Two portable ZIP packages | `downloads/` (rebuilt from the publication copies) |

All existing recorded runs, raw synthetic trajectories, small checkpoints, plots, reports and animations are included. New run directories are ignored. A directory with only `manifest.json` represents an incomplete interactive run, not a successful experiment.

Excluded: virtual environments, dependency caches, Python bytecode, notebook autosave checkpoints, macOS metadata, private resume source files and research scratch directories. These are not needed to reproduce the exercises.

## Publication edits

- Portable relative setup paths replace machine-specific paths.
- Private Drive references and personal resume attribution are omitted from the initial guide. Local originals remain unchanged.
- The Notebook uses the verified delivery snapshot with 16 sequentially executed code cells; its lesson/code sources were checked against the live local copy and were unchanged. Numeric results and images are preserved.
- HTML is regenerated from that same Notebook; both archives are rebuilt from repository-ready copies.
- Historical manifests retain their original source hashes as experiment provenance. `export_inventory.json` records the SHA256 of each published file, so publication edits are not confused with the original runtime sources.
- No license is added on the author's behalf. Upstream environment and algorithm references remain in their respective lessons.

## Validation

Notebook schema, executed-cell order, absence of error outputs, embedded plots/GIFs, Python/shell syntax, and local Markdown links are checked before publishing. Seven contract tests cover dynamics, episode boundaries, checkpoint round trips and experiment configuration. See `robot_learning_loops/VALIDATION.md` for actual training validation and limitations.
