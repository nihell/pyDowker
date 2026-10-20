# Changelog

## 0.1.0 (2026-05-23)

First public release.

- C++ backend via pybind11 for `create_simplex_tree` and `create_bifiltration`
- Python fallback implementation for all filtration modes
- `multipers` integration as an optional extra (`pip install pyDowker[multipers]`)
- Full test suite (31 tests) covering C++ vs Python agreement
- Sphinx documentation hosted on ReadTheDocs
- Binary wheels for Linux, macOS, and Windows via cibuildwheel
