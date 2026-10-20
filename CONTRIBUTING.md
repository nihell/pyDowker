# Contributing to pyDowker

## Building from source

You need:
- Python ≥ 3.9
- A C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019+)
- CMake ≥ 3.18

```bash
git clone https://github.com/nihell/pyDowker.git
cd pyDowker
pip install scikit-build-core pybind11
pip install --no-build-isolation -e ".[dev]"
```

The `--no-build-isolation` flag lets the editable install reuse your local pybind11 and CMake instead of re-downloading them.

## Running tests

```bash
pytest tests/
```

To also run the multipers-dependent tests (requires `pip install pyDowker[multipers]`):

```bash
pytest tests/ --no-ignore=tests/test_multipers.py
```

## Building the docs locally

```bash
pip install -e ".[docs]"
cd docs
make html
# Open docs/_build/html/index.html
```

## Rebuilding the C++ extension

After editing `src/pyDowker/_dowker_complex.cpp`, re-install in editable mode:

```bash
pip install --no-build-isolation -e .
```

## Building wheels locally

Install [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/) and run:

```bash
pip install cibuildwheel
cibuildwheel --platform linux
```

## Release checklist

1. Update version in `pyproject.toml` and `docs/changelog.md`
2. Commit and push to `main`
3. Tag the commit: `git tag v0.x.y && git push origin v0.x.y`
4. The `publish.yml` workflow builds wheels and publishes to PyPI automatically via Trusted Publisher (no token needed — configure at pypi.org → project → Publishing)
