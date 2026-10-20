"""Tests for DowkerComplex: Python and C++ implementations."""

import numpy as np
import pytest

from pyDowker import DowkerComplex


# ── Total-weight filtration ────────────────────────────────────────────────────

class TestTotalWeightFiltration:
    def test_vertex_filtration(self, bool_relation):
        st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="TotalWeight", max_dimension=3
        )
        # vertex [0] has 3 witnesses → filtration -3
        assert st.filtration([0]) == -3

    def test_edge_filtration(self, bool_relation):
        st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="TotalWeight", max_dimension=3
        )
        # [0,4] share witnesses at columns 0 and 1 → count 2 → filtration -2
        assert st.filtration([0, 4]) == -2

    def test_triangle_filtration(self, bool_relation):
        st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="TotalWeight", max_dimension=3
        )
        assert st.filtration([0, 1, 4]) == -1

    def test_bool_input_accepted(self, bool_relation):
        st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="TotalWeight", max_dimension=2
        )
        assert st.num_simplices() > 0

    def test_float_input_auto_converted(self, float_relation):
        """Float matrix should be thresholded at level=0 (no witnesses) → only vertices."""
        st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="TotalWeight", max_dimension=2, level=0
        )
        # level=0 means only entries ≤ 0 are considered witnesses
        # float_relation has one entry == 0 (row 2, col 1), so vertex [2] gets -1
        assert st.filtration([2]) == -1


# ── Sublevel filtration ────────────────────────────────────────────────────────

class TestSublevelFiltration:
    def test_edge_2_4(self, float_relation):
        st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3
        )
        assert st.filtration([2, 4]) == 1.0

    def test_edge_0_4(self, float_relation):
        st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3
        )
        assert st.filtration([0, 4]) == 2.0

    def test_triangle_0_2_4(self, float_relation):
        st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3
        )
        assert st.filtration([0, 2, 4]) == 2.0

    def test_edge_2_3(self, float_relation):
        st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3
        )
        assert st.filtration([2, 3]) == 2.0

    def test_requires_float(self):
        int_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
        with pytest.raises(TypeError):
            DowkerComplex(int_matrix).create_simplex_tree(
                filtration="Sublevel", max_dimension=1
            )

    def test_max_filtration_cutoff(self, float_relation):
        """max_filtration should exclude simplices appearing above the cutoff."""
        st_full = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3
        )
        st_cut = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3, max_filtration=2.0
        )
        assert st_cut.num_simplices() <= st_full.num_simplices()

    def test_m_parameter_reduces_complex(self, float_relation):
        """Higher m means fewer witnesses required → should not add simplices missing witnesses."""
        st_m1 = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=2
        )
        st_m2 = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=2, max_dimension=2
        )
        # m=2 filtration values are higher or equal → at any cutoff, fewer/same simplices
        assert st_m2.num_simplices() <= st_m1.num_simplices()


# ── No-filtration mode ─────────────────────────────────────────────────────────

class TestNoFiltration:
    def test_returns_simplex_tree(self, bool_relation):
        st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="None", max_dimension=2
        )
        assert st.num_simplices() > 0

    def test_all_filtrations_zero(self, bool_relation):
        st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="None", max_dimension=2
        )
        for simplex, filt in st.get_filtration():
            assert filt == 0.0

    def test_invalid_filtration_raises(self, float_relation):
        with pytest.raises(Exception, match="filtration"):
            DowkerComplex(float_relation).create_simplex_tree(
                filtration="BadValue", max_dimension=1
            )


# ── C++ vs Python agreement ────────────────────────────────────────────────────

class TestCppVsPython:
    def test_sublevel_agreement(self, float_relation):
        py_st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=3
        )
        cpp_st = DowkerComplex(float_relation).create_simplex_tree_cpp(
            max_dimension=3, filtration="Sublevel", m=1
        )
        # Both should return the same simplices with the same filtration values
        py_dict = {tuple(s): f for s, f in py_st.get_filtration()}
        cpp_dict = {tuple(s): f for s, f in cpp_st.get_filtration()}
        assert set(py_dict.keys()) == set(cpp_dict.keys())
        for simplex in py_dict:
            assert py_dict[simplex] == pytest.approx(cpp_dict[simplex])

    def test_total_weight_agreement(self, bool_relation):
        py_st = DowkerComplex(bool_relation).create_simplex_tree(
            filtration="TotalWeight", max_dimension=3
        )
        cpp_st = DowkerComplex(bool_relation).create_simplex_tree_cpp(
            max_dimension=3, filtration="TotalWeight"
        )
        py_dict = {tuple(s): f for s, f in py_st.get_filtration()}
        cpp_dict = {tuple(s): f for s, f in cpp_st.get_filtration()}
        assert set(py_dict.keys()) == set(cpp_dict.keys())
        for simplex in py_dict:
            assert py_dict[simplex] == pytest.approx(cpp_dict[simplex])


# ── Bifiltration (pure Python) ─────────────────────────────────────────────────

class TestBifiltration:
    def test_returns_dict(self, float_relation):
        result = DowkerComplex(float_relation).create_rivet_bifiltration(
            max_dimension=2
        )
        assert isinstance(result, dict)

    def test_keys_are_tuples(self, float_relation):
        result = DowkerComplex(float_relation).create_rivet_bifiltration(
            max_dimension=2
        )
        for key in result:
            assert isinstance(key, tuple)

    def test_appearances_are_sorted(self, float_relation):
        result = DowkerComplex(float_relation).create_rivet_bifiltration(
            max_dimension=2, m_max=3
        )
        for simplex, apps in result.items():
            rs = [a[0] for a in apps]
            assert rs == sorted(rs), f"Appearances not sorted for {simplex}"

    def test_cpp_bifiltration_agrees_with_python(self, float_relation):
        py_result = DowkerComplex(float_relation).create_rivet_bifiltration(
            max_dimension=2, m_max=3
        )
        cpp_result = DowkerComplex(float_relation).create_rivet_bifiltration_cpp(
            max_dimension=2, m_max=3
        )
        assert set(py_result.keys()) == set(cpp_result.keys())
        for simplex in py_result:
            py_apps = sorted(py_result[simplex])
            cpp_apps = sorted(cpp_result[simplex])
            assert len(py_apps) == len(cpp_apps)
            for (pr, pm), (cr, cm) in zip(py_apps, cpp_apps):
                assert pr == pytest.approx(cr)
                assert pm == cm

    def test_simplex_2_4_first_appearance(self, float_relation):
        """[2,4]: max(row2, row4) = max([2,0,2,6],[3,1,4,1]) = [3,1,4,6] → sorted [1,3,4,6] → m=1 at r=1."""
        result = DowkerComplex(float_relation).create_rivet_bifiltration(
            max_dimension=2, m_max=4
        )
        apps = dict(result[(2, 4)])
        assert apps[1] == pytest.approx(1.0)


# ── Subdivision filtration ────────────────────────────────────────────────────

class TestSubdivisionFiltration:
    """
    Tests for the subdivision filtration S(D(Y, X, R^T)) from Theorem 3.3.

    Using the 4×4 relation from Figure 5 of the preprint:
        D(Y, X, R^T) has 4 vertices {α,β,γ,δ} and 4 edges {αβ},{βγ},{βδ},{γδ}.
    After sorting by (size, lex) the simplex indices in Sd are:
        0:(0,)  1:(1,)  2:(2,)  3:(3,)   ← dim-0 duals, filtration −1
        4:(0,1) 5:(1,2) 6:(1,3) 7:(2,3)  ← dim-1 duals, filtration −2
    Edges in Sd connect each vertex to the dual-edge simplices containing it:
        (0)→4, (1)→4,5,6, (2)→5,7, (3)→6,7  ← 8 edges, all at filtration −1
    Total: 16 simplices.
    """

    def test_simplex_count(self, preprint_fig5_relation):
        st = DowkerComplex(preprint_fig5_relation).create_subdivision_simplex_tree(
            max_dimension=2
        )
        assert st.num_simplices() == 16

    def test_dim0_dual_vertex_filtration(self, preprint_fig5_relation):
        st = DowkerComplex(preprint_fig5_relation).create_subdivision_simplex_tree(
            max_dimension=2
        )
        assert st.filtration([0]) == -1.0

    def test_dim1_dual_vertex_filtration(self, preprint_fig5_relation):
        # The 5th simplex (index 4) represents the dual-edge {α,β} → filtration −2
        st = DowkerComplex(preprint_fig5_relation).create_subdivision_simplex_tree(
            max_dimension=2
        )
        assert st.filtration([4]) == -2.0

    def test_flag_edge_filtration(self, preprint_fig5_relation):
        # Flag {α}⊊{α,β} = [0,4]: bottom simplex has size 1 → filtration −1
        st = DowkerComplex(preprint_fig5_relation).create_subdivision_simplex_tree(
            max_dimension=2
        )
        assert st.filtration([0, 4]) == -1.0

    def test_simplex_count_at_m2(self, preprint_fig5_relation):
        # At filtration ≤ −2 only the 4 dual-edge vertices appear (S(D)_2)
        st = DowkerComplex(preprint_fig5_relation).create_subdivision_simplex_tree(
            max_dimension=2
        )
        m2_simplices = [s for s, f in st.get_filtration() if f <= -2.0]
        assert len(m2_simplices) == 4

    def test_float_input_thresholded(self):
        # Float matrix thresholded at level=1: entries ≤1 become witnesses
        rel = np.array([[2, 1, 3], [1, 2, 1]], dtype=np.float64)
        st = DowkerComplex(rel).create_subdivision_simplex_tree(
            max_dimension=2, level=1
        )
        assert st.num_simplices() > 0

    def test_max_dimension_zero(self, preprint_fig5_relation):
        # max_dimension=0: only isolated vertices in Sd, one per dual simplex
        st = DowkerComplex(preprint_fig5_relation).create_subdivision_simplex_tree(
            max_dimension=0
        )
        for simplex, _ in st.get_filtration():
            assert len(simplex) == 1


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_vertex(self):
        rel = np.array([[1.0, 0.5]], dtype=np.float64)
        st = DowkerComplex(rel).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=0
        )
        assert st.num_simplices() == 1

    def test_all_inf_no_simplices(self):
        rel = np.full((3, 3), np.inf)
        st = DowkerComplex(rel).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=2, max_filtration=1e9
        )
        assert st.num_simplices() == 0

    def test_dimension_0_only_vertices(self, float_relation):
        st = DowkerComplex(float_relation).create_simplex_tree(
            filtration="Sublevel", m=1, max_dimension=0
        )
        for simplex, _ in st.get_filtration():
            assert len(simplex) == 1
