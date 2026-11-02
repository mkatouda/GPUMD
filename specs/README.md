# specs/ (fork-only)

Source of truth for dielectric-response development in this fork. See `AGENTS.md`.

- `requirements/` — one file per requirement, ID `DR-NNN`, with acceptance criteria.
- `design/` — design notes (equations, data flow, GPU layout).
- `decisions/` — ADRs (`ADR-NNNN-title.md`); required for any change to existing
  upstream keywords, file formats, or default numerics.
- `traceability.yaml` — requirement → design/ADR → tests → status.

Changes here require human approval. Do not include this directory in upstream PRs.
