# Tests

- `oracles/`: invariants of the construction. Unmarked, so they must always pass. Each file's
  docstring states the theorem it checks and why exact comparison is valid.
  `reference.py` holds brute-force definitions, `corpus.py` the shared inputs.
- `contracts/`: one file per requirement, `test_<id>_<topic>.py`, with
  `pytestmark = pytest.mark.req("<ID>")`. The requirement's status decides the expected
  outcome (`conftest.py`).
- `tooling/`: tests for the guard hook.
- `unit/`: implementers' own tests, the only part of `tests/` an implementer may edit.

When an oracle or contract fails, the implementation is wrong until shown otherwise. If a test
itself looks wrong, an implementer reports a minimal counterexample instead of editing it.

## Writing a contract (spec sessions)

- It must fail as a whole while the requirement is unmet: strict xfail turns any passing case
  into a failure. Loop over the corpus inside one test instead of parametrizing cases that
  already pass.
- Encode only what the spec decides. Undecided points go in the requirement's notes.
- Before accepting the requirement, check in a scratch copy that a plausible fix makes the
  contract pass.
- List the file in the requirement's `contract` field; `scripts/check_requirements.py` checks
  the link in both directions.

## Writing an oracle

State the theorem and its hypotheses in the docstring, say where exactness holds, and keep
inputs at 8 rows or fewer so the reference can stay brute force.
