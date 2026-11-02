# ADR-0001: Spec-driven, test-first development process for this fork

- Status: Accepted
- Date: 2026-11-02

## Context
Dielectric-response features are developed in the fork mkatouda/GPUMD and later
proposed upstream. Changes to force fields and electrostatics are easy to get subtly
wrong and hard to verify after the fact.

## Decision
- `specs/` (requirements, design, ADRs, `traceability.yaml`) is the source of truth;
  rules are in `AGENTS.md`.
- Tests are written before or with implementation; regression vs. the `master`
  binary (`tests_regression --suite full`) is required for every change.
- Force/physics/kernel changes get independent reviews by the agents in
  `.claude/agents/`.
- `specs/`, `.claude/`, and the process rules in `AGENTS.md` are fork-only and are
  excluded from upstream PRs.

## Consequences
Extra up-front effort per task, in exchange for traceable requirements, reproducible
verification, and upstream PRs that contain only code, tests, examples, and docs.
