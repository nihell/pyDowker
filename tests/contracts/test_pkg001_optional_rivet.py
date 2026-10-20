"""PKG-001: pyDowker installs and imports without RIVET.

numpy and gudhi are declared dependencies, pyrivet sits in an optional extra named `rivet`,
the GUDHI paths work without pyrivet, and the RIVET methods raise ImportError naming the extra.
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.req("PKG-001")
ROOT = pathlib.Path(__file__).resolve().parents[2]

WITHOUT_PYRIVET = textwrap.dedent("""
    import importlib.abc, sys

    class BlockPyrivet(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "pyrivet" or name.startswith("pyrivet."):
                raise ImportError(name + " is blocked for this test")

    sys.meta_path.insert(0, BlockPyrivet())

    import numpy as np
    from pyDowker.DowkerComplex import DowkerComplex
    import pyDowker.TwoParameterUtils

    st = DowkerComplex(np.array([[0.0, 1.0], [1.0, 0.0]])).create_simplex_tree(
        max_dimension=1, filtration="Sublevel")
    assert st.num_simplices() == 3, st.num_simplices()
    try:
        DowkerComplex(np.zeros((2, 2))).create_rivet_bifiltration(max_dimension=1)
    except ImportError as exc:
        assert "pyDowker[rivet]" in str(exc), str(exc)
    else:
        raise AssertionError("create_rivet_bifiltration should raise ImportError without pyrivet")
""")


def test_gudhi_paths_work_without_pyrivet():
    # The child must import this checkout's src, as pytest does (pythonpath = src).
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run([sys.executable, "-c", WITHOUT_PYRIVET], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_dependencies_are_declared():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    def names(specs):
        return {re.split(r"[\s<>=!~;\[@]", spec, maxsplit=1)[0].lower() for spec in specs}

    assert {"numpy", "gudhi"} <= names(project.get("dependencies", []))
    assert "pyrivet" in names(project.get("optional-dependencies", {}).get("rivet", []))
