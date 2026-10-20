# 0002: Scope and timing of the native engine (ENG-001)

Status: proposed, needs Niklas's decision

## Context

- The engine was designed in chat for non-filtered homology with Z/p coefficients: Dowker
  duality, cohomology-first reduction, clearing, apparent pairs via discrete Morse theory,
  bitset-AND rows.
- AttentionDowker needs filtered persistence per head (H1, H3) and a bifiltration (H4,
  DWK-006). A non-filtered engine serves none of these directly.
- pyDowker's Python sublevel path passes every oracle (2026-10-20) and is the reference any
  engine has to match.
- Measured on synthetic attention at max_dimension 2 (`docs/reports/2026-10-20-baseline-cost.md`):
  without a cutoff the Python path needs about 17 s per head at n = 256 and, extrapolated,
  30 to 35 minutes per 12-head layer at n = 512, against DWK-007's 30 s. With the sink dropped
  and a 5% cutoff, prototypes of three Python-level changes (DWK-008) bring n = 512 to about
  4 s per head including persistence, about 48 s per layer on one core.

## Options

A. Build the non-filtered engine first, as designed, and add filtration later.
B. Make the first version filtered (sublevel, one parameter), keeping the techniques that carry
   over: duality to put the short side in the rows, cohomology with clearing, apparent pairs.
C. Build nothing native yet. Implement DWK-008 in Python if Phase 0 needs the speed, and pull
   ENG-001 only if the measured layer time on Niklas's laptop still misses DWK-007.

## Recommendation

C. Option A produces an engine its only consumer cannot use. On these measurements the Python
path with DWK-008 lands within a small factor of the budget, and sink handling (open decision 1
in `docs/specs/attention-interface.md`) moves cost by more than that factor. If ENG-001 is pulled
later, B is the scope that serves AttentionDowker.

## Decision

Pending.
