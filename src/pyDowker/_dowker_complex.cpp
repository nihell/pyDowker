#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

namespace py = pybind11;

using Simplex = std::vector<int>;
using FiltrationEntry = std::pair<Simplex, double>;
using Appearance = std::pair<double, int>;


// ============================================================
// ===================== SUBLEVEL ==============================
// ============================================================

void append_sublevel(
    const Simplex& sigma,
    double r,
    int depth,
    int max_dimension,
    int m,
    int num_points,
    int num_potential_witnesses,
    double max_filtration,
    const double* data,
    std::vector<std::vector<double>>& buffers,
    std::vector<FiltrationEntry>& out)
{
    if (r > max_filtration) return;

    out.emplace_back(sigma, r);

    if ((int)sigma.size() > max_dimension) return;

    int max_idx = *std::max_element(sigma.begin(), sigma.end());

    double* current = buffers[depth].data();
    double* next    = buffers[depth + 1].data();

    for (int j = max_idx + 1; j < num_points; j++) {
        Simplex tau = sigma;
        tau.push_back(j);

        for (int i = 0; i < num_potential_witnesses; i++) {
            double jw = data[j * num_potential_witnesses + i];
            next[i] = std::max(jw, current[i]);
        }

        // selection (still requires scratch copy)
        std::vector<double> tmp(next, next + num_potential_witnesses);
        std::nth_element(tmp.begin(), tmp.begin() + (m - 1), tmp.end());
        double new_r = tmp[m - 1];

        append_sublevel(tau, new_r, depth + 1,
                        max_dimension, m,
                        num_points, num_potential_witnesses,
                        max_filtration, data,
                        buffers, out);
    }
}


// ============================================================
// ===================== TOTAL WEIGHT ==========================
// ============================================================

void append_total_weight(
    const Simplex& sigma,
    int depth,
    int max_dimension,
    int num_points,
    int num_potential_witnesses,
    const uint8_t* data,
    std::vector<std::vector<uint8_t>>& buffers,
    std::vector<FiltrationEntry>& out)
{
    uint8_t* current = buffers[depth].data();
    uint8_t* next    = buffers[depth + 1].data();

    int sum = 0;
    for (int i = 0; i < num_potential_witnesses; i++) sum += current[i];

    out.emplace_back(sigma, -double(sum));

    if ((int)sigma.size() > max_dimension) return;

    int max_idx = *std::max_element(sigma.begin(), sigma.end());

    for (int j = max_idx + 1; j < num_points; j++) {
        Simplex tau = sigma;
        tau.push_back(j);

        int nonzero = 0;
        for (int i = 0; i < num_potential_witnesses; i++) {
            uint8_t jw = data[j * num_potential_witnesses + i];
            next[i] = jw && current[i];
            nonzero += next[i];
        }

        if (nonzero > 0) {
            append_total_weight(tau, depth + 1,
                                max_dimension,
                                num_points, num_potential_witnesses,
                                data, buffers, out);
        }
    }
}


// ============================================================
// ===================== NONE =================================
// ============================================================

void append_none(
    const Simplex& sigma,
    int depth,
    int max_dimension,
    int m,
    int num_points,
    int num_potential_witnesses,
    const uint8_t* data,
    std::vector<std::vector<uint8_t>>& buffers,
    std::vector<FiltrationEntry>& out)
{
    uint8_t* current = buffers[depth].data();
    uint8_t* next    = buffers[depth + 1].data();

    int count = 0;
    for (int i = 0; i < num_potential_witnesses; i++) count += current[i];

    if (count < m) return;

    out.emplace_back(sigma, 0.0);

    if ((int)sigma.size() > max_dimension) return;

    int max_idx = *std::max_element(sigma.begin(), sigma.end());

    for (int j = max_idx + 1; j < num_points; j++) {
        Simplex tau = sigma;
        tau.push_back(j);

        int nonzero = 0;
        for (int i = 0; i < num_potential_witnesses; i++) {
            uint8_t jw = data[j * num_potential_witnesses + i];
            next[i] = jw && current[i];
            nonzero += next[i];
        }

        if (nonzero > 0) {
            append_none(tau, depth + 1,
                        max_dimension, m,
                        num_points, num_potential_witnesses,
                        data, buffers, out);
        }
    }
}


// ============================================================
// ===================== MAIN (1-param) ========================
// ============================================================

py::list create_simplex_tree(
    py::array rel_matrix,
    int max_dimension,
    std::string filtration = "None",
    int m = 1,
    double level = 0.0,
    double max_filtration = std::numeric_limits<double>::infinity())
{
    py::buffer_info buf = rel_matrix.request();

    int num_points = buf.shape[0];
    int num_potential_witnesses = buf.shape[1];

    std::vector<FiltrationEntry> out;

    // -------- SUBLEVEL --------
    if (filtration == "Sublevel") {
        auto arr = rel_matrix.cast<py::array_t<double>>();
        double* data = (double*)arr.request().ptr;

        std::vector<std::vector<double>> buffers(max_dimension + 2,
            std::vector<double>(num_potential_witnesses));

        for (int k = num_points - 1; k >= 0; k--) {
            for (int i = 0; i < num_potential_witnesses; i++)
                buffers[0][i] = data[k * num_potential_witnesses + i];

            std::vector<double> tmp = buffers[0];
            std::nth_element(tmp.begin(), tmp.begin() + (m - 1), tmp.end());
            double r = tmp[m - 1];

            append_sublevel({k}, r, 0,
                            max_dimension, m,
                            num_points, num_potential_witnesses,
                            max_filtration, data,
                            buffers, out);
        }
    }

    // -------- TOTAL WEIGHT / NONE --------
    else {
        py::array_t<uint8_t> arr_bool({num_points, num_potential_witnesses});
        uint8_t* outp = (uint8_t*)arr_bool.request().ptr;

        if (py::isinstance<py::array_t<bool>>(rel_matrix)) {
            bool* src = (bool*)rel_matrix.cast<py::array_t<bool>>().request().ptr;
            for (int i = 0; i < num_points * num_potential_witnesses; i++)
                outp[i] = src[i];
        } else {
            double* src = (double*)rel_matrix.cast<py::array_t<double>>().request().ptr;
            for (int i = 0; i < num_points * num_potential_witnesses; i++)
                outp[i] = src[i] <= level;
        }

        uint8_t* data = outp;

        std::vector<std::vector<uint8_t>> buffers(max_dimension + 2,
            std::vector<uint8_t>(num_potential_witnesses));

        for (int k = num_points - 1; k >= 0; k--) {
            for (int i = 0; i < num_potential_witnesses; i++)
                buffers[0][i] = data[k * num_potential_witnesses + i];

            if (filtration == "TotalWeight") {
                append_total_weight({k}, 0,
                                    max_dimension,
                                    num_points, num_potential_witnesses,
                                    data, buffers, out);
            } else {
                append_none({k}, 0,
                            max_dimension, m,
                            num_points, num_potential_witnesses,
                            data, buffers, out);
            }
        }
    }

    return py::cast(out);
}


// ============================================================
// ===================== BIFILTRATION ==========================
// ============================================================

void append_bifiltration(
    const Simplex& sigma,
    int depth,
    int max_dimension,
    int m_max,
    int num_points,
    int num_potential_witnesses,
    const double* data,
    std::vector<std::vector<double>>& buffers,
    std::vector<Simplex>& simplices,
    std::vector<std::vector<Appearance>>& appearances)
{
    double* current = buffers[depth].data();
    double* next    = buffers[depth + 1].data();

    simplices.push_back(sigma);

    std::vector<double> sorted(current, current + num_potential_witnesses);
    std::sort(sorted.begin(), sorted.end());

    std::vector<Appearance> app;
    for (int i = 1; i <= m_max; i++) {
        if (i-1 >= (int)sorted.size()) break;
        if (std::isinf(sorted[i-1])) break;
        app.emplace_back(sorted[i-1], i);
    }
    appearances.push_back(app);

    if ((int)sigma.size() > max_dimension) return;

    int max_idx = *std::max_element(sigma.begin(), sigma.end());

    for (int j = max_idx + 1; j < num_points; j++) {
        Simplex tau = sigma;
        tau.push_back(j);

        for (int i = 0; i < num_potential_witnesses; i++) {
            double jw = data[j * num_potential_witnesses + i];
            next[i] = std::max(jw, current[i]);
        }

        append_bifiltration(tau, depth + 1,
                            max_dimension, m_max,
                            num_points, num_potential_witnesses,
                            data, buffers,
                            simplices, appearances);
    }
}


py::tuple create_bifiltration(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr,
    int max_dimension,
    int m_max = 5)
{
    py::buffer_info buf = arr.request();

    int num_points = buf.shape[0];
    int num_potential_witnesses = buf.shape[1];

    double* data = (double*)buf.ptr;

    std::vector<Simplex> simplices;
    std::vector<std::vector<Appearance>> appearances;

    std::vector<std::vector<double>> buffers(max_dimension + 2,
        std::vector<double>(num_potential_witnesses));

    for (int k = num_points - 1; k >= 0; k--) {
        for (int i = 0; i < num_potential_witnesses; i++)
            buffers[0][i] = data[k * num_potential_witnesses + i];

        append_bifiltration({k}, 0,
                            max_dimension, m_max,
                            num_points, num_potential_witnesses,
                            data, buffers,
                            simplices, appearances);
    }

    return py::make_tuple(simplices, appearances);
}


// ============================================================
// ===================== MODULE ================================
// ============================================================

PYBIND11_MODULE(_dowker_complex, m) {
    m.doc() = R"(
C++ backend for pyDowker.

This module provides optimised implementations of the Dowker complex
construction algorithms described in:

  Hellmer & Spalinski, "Density Sensitive Bifiltered Dowker Complexes
  via Total Weight", arXiv:2405.15592.

Functions
---------
create_simplex_tree
    Build a filtered simplex tree from a relation matrix.
create_bifiltration
    Build the full bifiltration (sublevel × witness-count) from a relation matrix.
)";

    m.def(
        "create_simplex_tree",
        &create_simplex_tree,
        py::arg("rel_matrix"),
        py::arg("max_dimension"),
        py::arg("filtration") = "None",
        py::arg("m") = 1,
        py::arg("level") = 0.0,
        py::arg("max_filtration") = std::numeric_limits<double>::infinity(),
        R"(
Build a filtered Dowker simplex tree.

Parameters
----------
rel_matrix : numpy.ndarray
    2-D array of shape (n_points, n_witnesses).  Either float64 (required for
    ``filtration="Sublevel"``) or bool/uint8 (for "TotalWeight"/"None").
max_dimension : int
    Maximum skeleton dimension to compute.
filtration : {"None", "Sublevel", "TotalWeight"}, optional
    Filtration type.  Default is ``"None"`` (unfiltered complex at the given
    ``level`` threshold).
m : int, optional
    Minimum witness count parameter.  Default 1.
level : float, optional
    Threshold for converting a float matrix to boolean (used by "TotalWeight"
    and "None").  Default 0.0.
max_filtration : float, optional
    Cutoff: simplices with filtration value above this are omitted (Sublevel
    only).  Default ``inf``.

Returns
-------
list of (list[int], float)
    List of ``(simplex, filtration_value)`` pairs.
)");

    m.def(
        "create_bifiltration",
        &create_bifiltration,
        py::arg("rel_matrix"),
        py::arg("max_dimension"),
        py::arg("m_max") = 5,
        R"(
Build the density-sensitive bifiltration of a Dowker complex.

For each simplex, records its appearance bidegree ``(r, m)`` for every
``m`` from 1 to ``m_max``, where ``r`` is the sublevel filtration value at
witness count ``m``.  This is Algorithm 1 of arXiv:2405.15592.

Parameters
----------
rel_matrix : numpy.ndarray
    2-D float64 array of shape (n_points, n_witnesses).
max_dimension : int
    Maximum skeleton dimension to compute.
m_max : int, optional
    Maximum witness count to record.  Default 5.

Returns
-------
tuple[list[list[int]], list[list[tuple[float, int]]]]
    A pair ``(simplices, appearances)`` where ``simplices[i]`` is the
    ``i``-th simplex and ``appearances[i]`` is a list of ``(r, m)`` pairs.
)");
}