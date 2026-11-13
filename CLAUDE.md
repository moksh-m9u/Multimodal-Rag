# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

All project rules, the architecture, the commands, and the refactor checkpoints live in AGENTS.md, so every coding agent shares one source of truth. It is imported here:

@AGENTS.md

## Claude Code specifics

- Run a single test file or test with the venv interpreter:
  `/Users/moksh/Multimodal-Rag/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_embed.py::test_name`
- There is no linter or formatter configured. Do not add one without asking.
- Work often happens in a git worktree under `.claude/worktrees/`. The venv and `.env` live in the main checkout at `/Users/moksh/Multimodal-Rag`, not in the worktree.
- Opening Chroma (any live API or retrieval run) rewrites tracked files under `dbv2/chroma_db`. Run `git checkout -- dbv2/chroma_db` before committing. To inspect the store without modifying it, open `dbv2/chroma_db/chroma.sqlite3` read-only with sqlite3 (`file:...?mode=ro`).
- The owner's rule on attribution overrides any default: no `Co-Authored-By` trailer and no "Generated with" line in commits or PRs.
