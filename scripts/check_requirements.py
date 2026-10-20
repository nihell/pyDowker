#!/usr/bin/env python3
"""Validate requirements.yaml, show its dependency graph, and optionally run the guard.

    python scripts/check_requirements.py            validate + status and dependency view
    python scripts/check_requirements.py --mermaid  also print a Mermaid flowchart
    python scripts/check_requirements.py --guard [--base REF]
                                                     also check that this branch changed no
                                                     protected path (implementer branches)

Exit status: 0 OK, 1 registry errors, 2 guard violations.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "requirements.yaml"
TESTS = ROOT / "tests"

STATUSES = ["implemented", "in-progress", "accepted", "proposed", "dropped"]
NEEDS_CONTRACT = {"accepted", "in-progress", "implemented"}
KINDS = {"bug", "feature", "budget", "infra"}
REQUIRED = {"id", "title", "kind", "origin", "status"}
OPTIONAL = {"serves", "spec", "contract", "budget", "blocked_by", "notes"}
ID_RE = re.compile(r"^[A-Z]{2,6}-\d{3}$")
MARK_RE = re.compile(r"""mark\.req\(\s*["']([A-Z]{2,6}-\d{3})["']""")


def load(path=REGISTRY):
    data = yaml.safe_load(path.read_text())
    return data.get("requirements", []) if isinstance(data, dict) else []


def markers_in_tests():
    """{test file (repo-relative) -> set of requirement IDs it marks}."""
    found = {}
    for path in sorted(TESTS.rglob("*.py")):
        ids = set(MARK_RE.findall(path.read_text()))
        if ids:
            found[path.relative_to(ROOT).as_posix()] = ids
    return found


def validate(reqs, markers):
    errors = []
    by_id = {}
    for i, r in enumerate(reqs):
        rid = r.get("id", f"entry #{i + 1}")
        missing, unknown = REQUIRED - r.keys(), r.keys() - REQUIRED - OPTIONAL
        if missing:
            errors.append(f"{rid}: missing fields {sorted(missing)}")
        if unknown:
            errors.append(f"{rid}: unknown fields {sorted(unknown)}")
        if not ID_RE.match(str(rid)):
            errors.append(f"{rid}: id must look like ABC-123")
        if rid in by_id:
            errors.append(f"{rid}: duplicate id")
        by_id[rid] = r
        if r.get("kind") not in KINDS:
            errors.append(f"{rid}: kind must be one of {sorted(KINDS)}")
        if r.get("status") not in STATUSES:
            errors.append(f"{rid}: status must be one of {STATUSES}")
        if r.get("kind") == "budget" and not r.get("budget"):
            errors.append(f"{rid}: a budget requirement needs a `budget` field")
        if r.get("spec") and not (ROOT / r["spec"]).exists():
            errors.append(f"{rid}: spec {r['spec']} does not exist")
        contract = r.get("contract") or []
        if r.get("status") in NEEDS_CONTRACT and not contract:
            errors.append(f"{rid}: status {r['status']} requires a contract")
        for c in contract:
            if not (ROOT / c).exists():
                errors.append(f"{rid}: contract {c} does not exist")
            elif rid not in markers.get(c, set()):
                errors.append(f'{rid}: contract {c} has no @pytest.mark.req("{rid}")')

    for rid, r in by_id.items():
        for b in r.get("blocked_by") or []:
            if b not in by_id:
                errors.append(f"{rid}: blocked_by {b}, which does not exist")
            elif r.get("status") == "implemented" and by_id[b].get("status") != "implemented":
                errors.append(f"{rid}: implemented, but its blocker {b} is {by_id[b].get('status')}")
    errors += [f"dependency cycle: {' -> '.join(c)}" for c in _cycles(by_id)]

    for path, ids in markers.items():
        for rid in sorted(ids):
            if rid not in by_id:
                errors.append(f"{path}: marks unknown requirement {rid}")
            elif path not in (by_id[rid].get("contract") or []):
                errors.append(f"{path}: marks {rid}, but {rid}'s contract list does not include it")
    return errors


def _cycles(by_id):
    state, cycles = {}, []

    def visit(n, stack):
        state[n] = "open"
        for b in by_id[n].get("blocked_by") or []:
            if b not in by_id:
                continue
            if state.get(b) == "open":
                cycles.append(stack[stack.index(b):] + [b])
            elif b not in state:
                visit(b, stack + [b])
        state[n] = "done"

    for n in by_id:
        if n not in state:
            visit(n, [n])
    return cycles


def ready(r, by_id):
    return r["status"] in ("accepted", "in-progress") and all(
        by_id.get(b, {}).get("status") == "implemented" for b in r.get("blocked_by") or [])


def report(reqs):
    by_id = {r["id"]: r for r in reqs}
    counts = ", ".join(f"{sum(r['status'] == s for r in reqs)} {s}" for s in STATUSES)
    lines = [f"pyDowker requirements: {len(reqs)} ({counts})", ""]
    width = max(len(r["title"]) for r in reqs) if reqs else 0
    for r in sorted(reqs, key=lambda r: (STATUSES.index(r["status"]), r["id"])):
        blockers = [b for b in r.get("blocked_by") or [] if by_id.get(b, {}).get("status") != "implemented"]
        note = "ready" if ready(r, by_id) else (f"blocked by {', '.join(blockers)}" if blockers else "")
        lines.append(f"  {r['status']:<12} {r['id']:<10} {r['title']:<{width}}  {note}".rstrip())
    ready_ids = [r["id"] for r in reqs if ready(r, by_id)]
    lines += ["", f"Ready to implement: {', '.join(ready_ids) or 'none'}"]

    dependents = {}
    for r in reqs:
        for b in r.get("blocked_by") or []:
            dependents.setdefault(b, []).append(r["id"])
    chains = []

    def walk(n, path):
        if n not in dependents:
            chains.append(" -> ".join(path))
        for d in dependents.get(n, []):
            walk(d, path + [d])

    for r in reqs:
        if not r.get("blocked_by") and r["id"] in dependents:
            walk(r["id"], [r["id"]])
    if chains:
        lines += ["Dependency chains:"] + [f"  {c}" for c in chains]
    return "\n".join(lines)


def mermaid(reqs):
    def node(rid):
        return rid.replace("-", "_")

    out = ["flowchart LR"]
    for r in reqs:
        title = r["title"].replace('"', "'")
        out.append(f'  {node(r["id"])}["{r["id"]}: {title}"]:::{r["status"].replace("-", "_")}')
    for r in reqs:
        for b in r.get("blocked_by") or []:
            out.append(f"  {node(b)} --> {node(r['id'])}")
    out += [
        "  classDef implemented fill:#d9f2e3,stroke:#2f7a4f",
        "  classDef in_progress fill:#fff1cc,stroke:#a07800",
        "  classDef accepted fill:#dbe9fb,stroke:#2f5f9e",
        "  classDef proposed fill:#eeeeee,stroke:#888888",
        "  classDef dropped fill:#f6dada,stroke:#9e2f2f",
    ]
    return "\n".join(out)


def run_guard(base):
    path = ROOT / ".claude/hooks/guard_protected_paths.py"
    spec = importlib.util.spec_from_file_location("guard_protected_paths", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard.main(["check", "--repo", str(ROOT)] + (["--base", base] if base else []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mermaid", action="store_true", help="print a Mermaid flowchart")
    parser.add_argument("--guard", action="store_true", help="check protected paths against the base")
    parser.add_argument("--base", help="base ref for --guard (default: the default branch)")
    args = parser.parse_args(argv)

    reqs = load()
    errors = validate(reqs, markers_in_tests())
    if errors:
        print("requirements.yaml has problems:", *[f"  - {e}" for e in errors], sep="\n", file=sys.stderr)
        return 1
    print(report(reqs))
    if args.mermaid:
        print("\n" + mermaid(reqs))
    if args.guard:
        print()
        return run_guard(args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
