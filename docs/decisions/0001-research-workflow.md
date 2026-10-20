# 0001: Requirements-driven research workflow

Status: accepted, 2026-10-20

## Context

Theory and design happen in chat, implementation in Claude Code. State lived in conversations:
an implementation session saw the repository plus whatever was pasted into it, and needs that
AttentionDowker places on pyDowker were recorded nowhere. The first run of a brute-force
reference suite against pyDowker found that the main sublevel path crashes on NumPy 2
(DWK-001), the unfiltered path ignores the relation (DWK-002), the Euler profile crashes
(DWK-003) and the constructor drops its cutoff (DWK-004), plus one edge case (DWK-005).

## Decision

1. The repository is the shared state. `CLAUDE.md` holds conventions and invariants,
   `requirements.yaml` what is built and why, `docs/` the specs, decisions and reports.
   Conversations are disposable; a chat result counts once it is a committed spec or test.
2. A requirement binds through its contract tests. Its status drives the expected outcome
   (strict xfail until implemented), so the registry cannot drift from the code.
3. Roles. Spec sessions (frontier model, high effort) write specs, contracts, oracles and
   decisions. The `implementer` subagent (Sonnet, own worktree) implements one accepted
   requirement and may not change what it is graded against; a hook enforces this. The
   `reviewer` subagent (Opus, read-only) checks the branch against the spec and the
   mathematics before merge.
4. Pull, don't push. A consumer need enters the registry when an experiment for a live
   hypothesis needs it. A budget is accepted only with a measurement that misses it.
5. Pure mathematics stays in chat and enters the repository only as a spec or a test.

## Consequences

- Correctness is decided by oracles and contracts written in spec sessions, which is where
  the expensive model is spent. Implementation runs on a cheaper model.
- Contracts must fail as a whole while unmet, which shapes how they are written
  (`tests/CLAUDE.md`).
- Worktrees branch from `main`, so spec and contract changes are merged before an
  implementer starts.
- The advisor tool and Claude Code Projects are left out until this loop has carried a few
  requirements end to end.
