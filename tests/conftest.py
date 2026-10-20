"""Links tests to requirements.yaml.

@pytest.mark.req("ID") ties a test to a requirement; the requirement's status sets the
expected outcome:

    implemented                     the test must pass
    accepted, in-progress, proposed strict xfail: the test must fail. If it passes, pytest
                                    reports XPASS(strict) as a failure: the requirement is
                                    met, so set its status to implemented.
    dropped                         the test is skipped

Tests without a req marker (the oracle suite) must always pass.
"""

import pathlib

import pytest
import yaml

REGISTRY = pathlib.Path(__file__).resolve().parents[1] / "requirements.yaml"


def _requirements():
    data = yaml.safe_load(REGISTRY.read_text())
    return {r["id"]: r for r in data["requirements"]}


def pytest_collection_modifyitems(config, items):
    reqs = _requirements()
    for item in items:
        for mark in item.iter_markers("req"):
            rid = mark.args[0]
            if rid not in reqs:
                raise pytest.UsageError(f"{item.nodeid}: unknown requirement {rid}")
            status, title = reqs[rid]["status"], reqs[rid]["title"]
            if status == "dropped":
                item.add_marker(pytest.mark.skip(reason=f"{rid} dropped: {title}"))
            elif status != "implemented":
                item.add_marker(pytest.mark.xfail(
                    strict=True,
                    reason=f"{rid} is {status}: {title} (if this passes, set {rid} to implemented)",
                ))
