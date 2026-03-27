#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

namespace py = pybind11;

using Simplex = std::vector<int>;
using Appearance = std::pair<double, int>;

// Thread-safe version: no globals, everything passed explicitly
void append_upper_cofaces(
    const Simplex& sigma,
    const std::vector<double>& dist_to_neighbors,
    int max_dimension,
    int m_max,
    int num_points,
    const std::vector<std::vector<double>>& dist,
    std::vector<Simplex>& simplices,
    std::vector<std::vector<Appearance>>& appearances) {

    simplices.push_back(sigma);

    std::vector<double> sorted_dists = dist_to_neighbors;
    std::sort(sorted_dists.begin(), sorted_dists.end());

    std::vector<Appearance> appearance;
    for (int i = 1; i <= m_max; i++) {
        while (i < m_max && sorted_dists[i - 1] == sorted_dists[i]) {
            i++;
        }
        if (std::isinf(sorted_dists[i - 1])) break;
        appearance.emplace_back(sorted_dists[i - 1], i);
    }
    appearances.push_back(appearance);

    if (sigma.size() <= static_cast<size_t>(max_dimension)) {
        for (int j = *std::max_element(sigma.begin(), sigma.end()) + 1; j < num_points; j++) {
            Simplex tau = sigma;
            tau.push_back(j);

            std::vector<double> dist_to_j_neighbors(num_points);
            for (int n = 0; n < num_points; n++) {
                dist_to_j_neighbors[n] = dist[j][n];
            }            

            std::vector<double> dist_to_common_neighbors(num_points);
            for (int l = 0; l < num_points; l++) {
                dist_to_common_neighbors[l] = std::max(dist_to_neighbors[l], dist_to_j_neighbors[l]);
            }

            append_upper_cofaces(tau, dist_to_common_neighbors, max_dimension, m_max,
                                 num_points, dist, simplices, appearances);
        }
    }
}

// Core computation (thread-safe)
std::pair<std::vector<Simplex>, std::vector<std::vector<Appearance>>>
create_bifiltration_core(const std::vector<std::vector<double>>& dist,
                         int max_dimension,
                         int m_max) {

    int num_points = dist.size();

    std::vector<Simplex> simplices;
    std::vector<std::vector<Appearance>> appearances;

    for (int k = num_points - 1; k >= 0; k--) {
        std::vector<double> dist_to_neighbors(num_points);
        for (int n = 0; n < num_points; n++) {
            dist_to_neighbors[n] = dist[k][n];
        }

       
        append_upper_cofaces({k}, dist_to_neighbors, max_dimension, m_max,
                             num_points, dist, simplices, appearances);
    }

    return {simplices, appearances};
}

// NumPy-enabled wrapper (zero-copy read)
py::tuple create_bifiltration(py::array_t<double, py::array::c_style | py::array::forcecast> dist_array,
                              int max_dimension,
                              int m_max = 5) {

    py::buffer_info buf = dist_array.request();

    int n = buf.shape[0];
    int m = buf.shape[1];

    // Map NumPy buffer → C++ vector (could optimize further with direct pointer access)
    double* ptr = static_cast<double*>(buf.ptr);

    std::vector<std::vector<double>> dist(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            dist[i][j] = ptr[i * n + j];
        }
    }

    auto result = create_bifiltration_core(dist, max_dimension, m_max);

    return py::make_tuple(result.first, result.second);
}

PYBIND11_MODULE(_dowker_complex, m) {
    m.doc() = "dowker bifiltration module";

    m.def("create_bifiltration",
          &create_bifiltration,
          py::arg("dist"),
          py::arg("max_dimension"),
          py::arg("m_max") = 5,
          "Create dowker bifiltration from a NumPy relation matrix.");
}
