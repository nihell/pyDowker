"""Tests for .claude/hooks/guard_protected_paths.py, the implementer's protected-path guard."""

import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("guard", ROOT / ".claude/hooks/guard_protected_paths.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

REGISTRY = """\
requirements:
  - id: DWK-001
    title: first
    status: accepted
  - id: DWK-002
    title: second
    status: in-progress
"""


def git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for rel, text in {
        "src/pkg.py": "x = 1\n",
        "tests/oracles/test_a.py": "def test_a(): pass\n",
        "tests/contracts/test_c.py": "def test_c(): pass\n",
        ".claude/agents/implementer.md": "agent\n",
        "requirements.yaml": REGISTRY,
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(root, "init", "-q")
    git(root, "checkout", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    git(root, "checkout", "-q", "-b", "feature")
    return root


def pre(tool, cwd, **tool_input):
    return guard.pre_tool_use({"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)})


@pytest.mark.parametrize("rel, protected", [
    ("tests/oracles/test_a.py", True),
    ("tests/contracts/test_c.py", True),
    ("tests/conftest.py", True),
    ("tests", True),
    ("tests/unit/test_mine.py", False),
    (".claude/agents/implementer.md", True),
    ("docs/specs/x.md", True),
    ("docs/reports/x.md", False),
    ("scripts/check_requirements.py", True),
    ("scripts/bench.py", False),
    ("src/pyDowker/DowkerComplex.py", False),
    ("./tests/oracles/x.py", True),
    ("requirements.yaml", False),
])
def test_is_protected(rel, protected):
    assert guard.is_protected(rel) is protected


def test_file_tools(repo):
    assert pre("Edit", repo, file_path=str(repo / "tests/oracles/test_a.py")) == 2
    assert pre("Write", repo, file_path=str(repo / "tests/oracles/new_dir/new.py")) == 2
    assert pre("Write", repo, file_path=str(repo / "tests/unit/test_mine.py")) == 0
    assert pre("Edit", repo, file_path=str(repo / "src/pkg.py")) == 0
    assert pre("NotebookEdit", repo, notebook_path=str(repo / "tests/oracles/nb.ipynb")) == 2
    assert pre("Edit", repo, file_path="tests/contracts/test_c.py") == 2  # relative to cwd


def test_worktree_under_dot_claude_is_judged_by_its_own_root(repo):
    wt = repo / ".claude/worktrees/w1"
    git(repo, "worktree", "add", "-q", str(wt), "-b", "w1", "main")
    assert pre("Edit", wt, file_path=str(wt / "src/pkg.py")) == 0
    assert pre("Edit", wt, file_path=str(wt / "tests/oracles/test_a.py")) == 2
    assert pre("Edit", wt, file_path=str(repo / "tests/oracles/test_a.py")) == 2


@pytest.mark.parametrize("command, blocked", [
    ("pytest tests/oracles -q 2>&1 | tail -3", False),
    ("cat tests/oracles/test_a.py | head", False),
    ("cp tests/oracles/test_a.py /tmp/copy.py", False),
    ("git checkout main -- tests/oracles/test_a.py", False),
    ("git diff main -- tests/contracts", False),
    ("echo x > tests/oracles/test_a.py", True),
    ("echo x >> tests/contracts/test_c.py", True),
    ("printf x | tee tests/oracles/test_a.py", True),
    ("sed -i 's/a/b/' tests/contracts/test_c.py", True),
    ("cp /tmp/x.py tests/oracles/", True),
    ("mv tests/oracles/test_a.py tests/unit/test_a.py", True),
    ("cat <<EOF > tests/oracles/new.py\nprint(1)\nEOF", True),
    ("python3 -c 'open(\"tests/oracles/test_a.py\", \"w\").write(\"\")'", True),
    ("cat > .claude/agents/implementer.md", True),
    ("echo x > tests/unit/test_mine.py", False),
])
def test_bash_heuristic(repo, command, blocked):
    assert (pre("Bash", repo, command=command) == 2) is blocked


def test_check_clean_and_allowed_changes(repo):
    (repo / "src/pkg.py").write_text("x = 2\n")
    (repo / "tests/unit").mkdir()
    (repo / "tests/unit/test_mine.py").write_text("def test_m(): pass\n")
    (repo / "requirements.yaml").write_text(
        REGISTRY.replace("status: accepted", "status: implemented")
                .replace("status: in-progress", "status: implemented"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "work")
    _, problems = guard.check_repo(str(repo))
    assert problems == []


def test_check_catches_committed_uncommitted_and_untracked(repo):
    (repo / "tests/oracles/test_a.py").write_text("def test_a(): assert True\n")
    git(repo, "commit", "-qam", "loosen oracle")
    (repo / "tests/contracts/test_c.py").unlink()
    (repo / "tests/contracts/test_new.py").write_text("")
    _, problems = guard.check_repo(str(repo))
    text = "\n".join(problems)
    assert "modified: tests/oracles/test_a.py" in text
    assert "deleted: tests/contracts/test_c.py" in text
    assert "untracked): tests/contracts/test_new.py" in text


@pytest.mark.parametrize("edit, message", [
    (lambda s: s.replace("status: accepted", "status: dropped"), "accepted -> dropped"),
    (lambda s: s.replace("status: in-progress", "status: accepted"), "in-progress -> accepted"),
    (lambda s: s.replace("title: first", "title: easier"), "only `status:` lines"),
    (lambda s: s + "  - id: DWK-003\n    title: third\n    status: implemented\n", "only `status:` lines"),
])
def test_check_registry_rules(repo, edit, message):
    path = repo / "requirements.yaml"
    path.write_text(edit(path.read_text()))
    _, problems = guard.check_repo(str(repo))
    assert any(message in p for p in problems), problems


def test_stop_hook_blocks_with_explanation(repo, capsys):
    (repo / "tests/oracles/test_a.py").write_text("")
    assert guard.stop({"cwd": str(repo), "stop_hook_active": True}) == 2
    assert "git checkout" in capsys.readouterr().err


def test_stop_hook_cli_end_to_end(repo):
    hook = ROOT / ".claude/hooks/guard_protected_paths.py"
    event = json.dumps({"hook_event_name": "SubagentStop", "cwd": str(repo)})
    ok = subprocess.run(["python3", str(hook), "stop"], input=event, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    (repo / "tests/oracles/test_a.py").write_text("")
    bad = subprocess.run(["python3", str(hook), "stop"], input=event, capture_output=True, text=True)
    assert bad.returncode == 2 and "tests/oracles/test_a.py" in bad.stderr
