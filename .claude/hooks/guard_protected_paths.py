#!/usr/bin/env python3
"""Keep implementer sessions from changing what they are graded against.

Protected paths (edited only in spec sessions):
    tests/            except tests/unit/, where implementers may add their own tests
    .claude/          agent definitions and this hook
    docs/specs/, docs/decisions/
    scripts/check_requirements.py
requirements.yaml: implementers may change only `status:` lines, and only
    accepted -> in-progress -> implemented.

Modes (Claude Code passes the hook event as JSON on stdin):
    pre-tool-use  PreToolUse hook for Edit|Write|NotebookEdit|Bash. Exact for the file
                  tools; a heuristic for Bash (redirections, tee, sed -i, cp/mv targets ...).
    stop          Stop/SubagentStop hook. The authoritative check: diffs the working tree
                  at the event's cwd against its merge base with the default branch.
    check         Same check from the command line: [--repo DIR] [--base REF].

Exit status: 0 allow, 2 block (reason on stderr), 1 could not check (non-blocking).
Standard library only, so it runs under whichever python3 the hook finds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from functools import lru_cache

PROTECTED_PREFIXES = ("tests/", ".claude/", "docs/specs/", "docs/decisions/")
PROTECTED_FILES = ("scripts/check_requirements.py",)
IMPLEMENTER_PREFIXES = ("tests/unit/",)
REGISTRY = "requirements.yaml"
ALLOWED_STATUS_FLIPS = {
    ("accepted", "in-progress"),
    ("accepted", "implemented"),
    ("in-progress", "implemented"),
}

# Bash commands that mention none of these cannot touch a protected path by name.
_FRAGMENTS = ("tests", "oracles", "contracts", "conftest", ".claude", "specs", "decisions",
              "check_requirements")
_SEPARATORS = {"&&", "||", ";", "|", "|&", "&", "(", ")", ";;"}
_WRAPPERS = {"sudo", "env", "command", "nohup", "time", "nice", "xargs"}


class GuardError(Exception):
    """The repository state could not be determined."""


def is_protected(rel: str) -> bool:
    """`rel` is a POSIX path relative to the repository root."""
    rel = rel.replace(os.sep, "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if rel in ("", "."):
        return False
    probe = rel if rel.endswith("/") else rel + "/"
    if any(probe.startswith(p) for p in IMPLEMENTER_PREFIXES):
        return False
    return rel in PROTECTED_FILES or any(probe.startswith(p) for p in PROTECTED_PREFIXES)


def _git(args, cwd, check=True):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise GuardError(f"git {' '.join(args)}: {proc.stderr.strip() or proc.returncode}")
    return proc


@lru_cache(maxsize=256)
def _toplevel(directory: str):
    proc = _git(["rev-parse", "--show-toplevel"], cwd=directory, check=False)
    return os.path.realpath(proc.stdout.strip()) if proc.returncode == 0 else None


def repo_relative(path: str, cwd: str):
    """Path relative to the root of the git checkout (or worktree) containing it."""
    path = os.path.expanduser(path)
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
    probe = full
    while not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    top = _toplevel(probe)
    if top is None:
        return None
    rel = os.path.relpath(full, top)
    return None if rel.startswith("..") else rel.replace(os.sep, "/")


# ---------------------------------------------------------------- pre-tool-use

def _protected_args(words, cwd):
    out = []
    for w in words:
        if w.startswith("-") and "=" in w:
            w = w.split("=", 1)[1]
        elif w.startswith("-"):
            continue
        rel = repo_relative(w, cwd)
        if rel is not None and is_protected(rel):
            out.append(rel)
    return out


def _segments(tokens):
    seg = []
    for t in tokens:
        if t in _SEPARATORS:
            if seg:
                yield seg
            seg = []
        else:
            seg.append(t)
    if seg:
        yield seg


def bash_block_reason(command: str, cwd: str):
    """Why `command` would write a protected path, or None. Heuristic by design."""
    if not any(f in command for f in _FRAGMENTS):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes, heredocs ...: let the stop check decide
        return None

    for seg in _segments(tokens):
        words, i = [], 0
        while i < len(seg):
            tok = seg[i]
            if set(tok) <= set("<>&|") and ">" in tok:  # redirection operator
                target = seg[i + 1] if i + 1 < len(seg) else ""
                fd_dup = tok in (">&", "<&") and (target.isdigit() or target == "-")
                if not fd_dup and _protected_args([target], cwd):
                    return f"redirects output into protected path {target}"
                i += 2
                continue
            words.append(tok)
            i += 1
        while words and (re.match(r"^\w+=", words[0]) or words[0] in _WRAPPERS):
            words = words[1:]
        if not words:
            continue
        name, args = os.path.basename(words[0]), words[1:]

        if name == "tee":
            hits = _protected_args(args, cwd)
        elif name in ("sed", "perl") and any(
                re.match(r"^-[A-Za-z0-9]*i", a) or a.startswith("--in-place") for a in args):
            hits = _protected_args(args, cwd)
        elif name in ("mv", "truncate", "patch"):
            hits = _protected_args(args, cwd)
        elif name == "dd":
            hits = _protected_args([a[3:] for a in args if a.startswith("of=")], cwd)
        elif name in ("cp", "install", "rsync", "ln"):
            positional = [a for a in args if not a.startswith("-")]
            targets = positional[-1:]
            for j, a in enumerate(args):
                if a in ("-t", "--target-directory") and j + 1 < len(args):
                    targets.append(args[j + 1])
                elif a.startswith("--target-directory="):
                    targets.append(a.split("=", 1)[1])
            hits = _protected_args(targets, cwd)
        elif name.startswith("python") and "-c" in args:
            code = args[args.index("-c") + 1] if args.index("-c") + 1 < len(args) else ""
            writes = re.search(r"open\(|write|unlink|rmtree|rename|replace\(", code)
            hits = ["(python -c)"] if writes and any(f in code for f in _FRAGMENTS) else []
        else:
            hits = []
        if hits:
            return f"`{name}` would modify protected path {hits[0]}"
    return None


def pre_tool_use(event) -> int:
    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}
    cwd = event.get("cwd") or os.getcwd()
    if tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        rel = repo_relative(path, cwd) if path else None
        if rel is not None and is_protected(rel):
            return _block(f"{rel} is protected.")
        return 0
    if tool == "Bash":
        reason = bash_block_reason(tool_input.get("command", ""), cwd)
        if reason:
            return _block(f"This command {reason}.")
    return 0


def _block(reason: str) -> int:
    print(
        f"Blocked by the protected-path guard: {reason}\n"
        "Oracle and contract tests, specs, decisions and the agent setup are written in spec "
        "sessions; implementers do not edit them. Add your own tests under tests/unit/. If a "
        "protected test or spec looks wrong, stop and report it with a minimal counterexample "
        "(input, expected, actual) instead of changing it.",
        file=sys.stderr,
    )
    return 2


# ------------------------------------------------------------------ stop / check

def default_base(repo: str) -> str:
    candidates = []
    if os.environ.get("GUARD_BASE"):
        candidates.append(os.environ["GUARD_BASE"])
    head = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], repo, check=False)
    if head.returncode == 0:
        remote_ref = head.stdout.strip()                     # refs/remotes/origin/main
        candidates += [remote_ref.rsplit("/", 1)[-1], remote_ref]
    candidates += ["main", "master", "origin/main", "origin/master"]
    for ref in candidates:
        if _git(["rev-parse", "--verify", "--quiet", ref + "^{commit}"], repo, check=False).returncode == 0:
            return ref
    raise GuardError("no default branch found (set GUARD_BASE or pass --base)")


def _registry_problems(diff_text: str):
    problems = []
    status = re.compile(r"^\s*(?:-\s+)?status:\s*([\w-]+)\s*(?:#.*)?$")
    for hunk in re.split(r"^@@[^\n]*\n", diff_text, flags=re.M)[1:]:
        removed = [line[1:] for line in hunk.splitlines() if line.startswith("-")]
        added = [line[1:] for line in hunk.splitlines() if line.startswith("+")]
        non_status = [line for line in removed + added if not status.match(line)]
        if non_status:
            problems.append(f"{REGISTRY}: only `status:` lines may change, found: {non_status[0].strip()!r}")
            continue
        if len(removed) != len(added):
            problems.append(f"{REGISTRY}: requirements may not be added or removed")
            continue
        for old, new in zip(removed, added):
            flip = (status.match(old).group(1), status.match(new).group(1))
            if flip not in ALLOWED_STATUS_FLIPS:
                problems.append(f"{REGISTRY}: status change {flip[0]} -> {flip[1]} is reserved for spec sessions")
    return problems


def check_repo(repo: str, base=None):
    """Return (merge_base, problems) for the checkout containing `repo`."""
    top = _toplevel(os.path.realpath(repo))
    if top is None:
        raise GuardError(f"{repo} is not inside a git checkout")
    base = base or default_base(top)
    merge_base = _git(["merge-base", "HEAD", base], top).stdout.strip()

    changed = _git(["diff", "--name-status", "--no-renames", merge_base], top).stdout.splitlines()
    untracked = _git(["ls-files", "--others", "--exclude-standard"], top).stdout.splitlines()
    problems = []
    for line in changed:
        kind, path = line.split("\t", 1)
        if is_protected(path):
            verb = {"A": "added", "D": "deleted", "M": "modified"}.get(kind[0], "changed")
            problems.append(f"protected file {verb}: {path}")
    problems += [f"protected file added (untracked): {p}" for p in untracked if is_protected(p)]
    registry_diff = _git(["diff", "-U0", merge_base, "--", REGISTRY], top).stdout
    problems += _registry_problems(registry_diff)
    return merge_base, problems


def _report(merge_base: str, problems) -> str:
    return "\n".join(
        ["The protected-path guard found changes this session may not make:"]
        + [f"  - {p}" for p in problems]
        + [
            f"Restore tracked files with `git checkout {merge_base[:12]} -- <path>`, delete files "
            "you added under protected paths, and revert any registry edit other than a status flip.",
            "If a protected test or spec is wrong, say so in your report with a minimal "
            "counterexample instead of changing it.",
        ]
    )


def stop(event) -> int:
    # Claude Code ends a run of consecutive blocks after a few attempts, so this keeps
    # blocking while violations remain instead of honouring stop_hook_active.
    cwd = event.get("cwd") or os.getcwd()
    try:
        merge_base, problems = check_repo(cwd)
    except GuardError as exc:
        print(f"protected-path guard could not run: {exc}", file=sys.stderr)
        return 1
    if problems:
        print(_report(merge_base, problems), file=sys.stderr)
        return 2
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["pre-tool-use", "stop", "check"])
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--base")
    args = parser.parse_args(argv)
    if args.mode == "check":
        try:
            merge_base, problems = check_repo(args.repo, args.base)
        except GuardError as exc:
            print(f"guard: {exc}", file=sys.stderr)
            return 1
        if problems:
            print(_report(merge_base, problems), file=sys.stderr)
            return 2
        print(f"guard: OK (no protected changes since {merge_base[:12]})")
        return 0
    event = json.load(sys.stdin)
    return pre_tool_use(event) if args.mode == "pre-tool-use" else stop(event)


if __name__ == "__main__":
    sys.exit(main())
