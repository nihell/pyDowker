#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

typedef std::vector<int> Simplex;
typedef std::pair<double, int> Appearance;

std::vector<Simplex> simplices;
std::vector<std::vector<Appearance>> appearances;

void append_upper_cofaces(const Simplex& sigma, const std::vector<double>& dist_to_neighbors, int max_dimension, int m_max, int num_points, const std::vector<std::vector<double>>& dist, bool closed) {
    simplices.push_back(sigma);
    std::vector<double> sorted_dists = dist_to_neighbors;
    std::sort(sorted_dists.begin(), sorted_dists.end());
    std::vector<Appearance> appearance;
    for (int i = 1; i <= m_max; i++) {
        while(sorted_dists[i-1]==sorted_dists[i]){//in order to make bifiltration minimal: if multiple m share the same distance,
            i++;                                  // only save the biggest of these m.
        }
        if (std::isinf(sorted_dists[i-1])){
            break;
        }
        appearance.push_back(std::make_pair(sorted_dists[i-1],i));
    }
    appearances.push_back(appearance);

    if (sigma.size() <= max_dimension) {
        for (int j = *std::max_element(sigma.begin(), sigma.end()) + 1; j < num_points; j++) {
            Simplex tau = sigma;
            tau.push_back(j);

            std::vector<double> dist_to_j_neighbors;
            for (int n = 0; n < num_points; n++) {
                dist_to_j_neighbors.push_back(dist[j][n]);
            }
            if (!closed){
                dist_to_j_neighbors[j] = std::numeric_limits<double>::infinity();
            }

            std::vector<double> dist_to_common_neighbors;
            for (int l = 0; l<num_points; l++) {
                dist_to_common_neighbors.push_back(std::max(dist_to_neighbors[l], dist_to_j_neighbors[l]));
            }

            append_upper_cofaces(tau, dist_to_common_neighbors, max_dimension, m_max, num_points, dist, closed);
        }
    }
}

std::pair<std::vector<Simplex>, std::vector<std::vector<Appearance>>> create_bifiltration(const std::vector<std::vector<double>>& dist, int max_dimension, int m_max = 5, bool closed=false) {
    int num_points = dist.size();
    simplices.clear();
    appearances.clear();

    for (int k = num_points - 1; k >= 0; k--) {
        std::vector<double> dist_to_neighbors;
        for (int n = 0; n < num_points; n++) {
            dist_to_neighbors.push_back(dist[k][n]);
        }
        if (!closed){
            dist_to_neighbors[k] = std::numeric_limits<double>::infinity();
        }
        append_upper_cofaces({k}, dist_to_neighbors, max_dimension, m_max, num_points, dist, closed);
    }

    return std::make_pair(simplices, appearances);
}

std::vector<std::vector<double>> readDistanceMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> dist;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        dist.push_back(row);
    }
    return dist;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: ./program_name <filename.csv>" << std::endl;
        return 1;
    }

    std::cout << "m-Neighbor bifiltered complex construction using distance matrix from " << argv[1] << ", max dimension = " << argv[2] << ", m_max = " << argv[3] << std::endl;

    std::string filename(argv[1]);
    std::vector<std::vector<double>> dist = readDistanceMatrixFromFile(filename);

    int max_dimension = std::stoi(argv[2]);
    int m_max = std::stoi(argv[3]);
    bool closed = false;
    if (argc > 4){
        if (std::string(argv[4])=="closed"){
            closed = true;
        }
    }

    std::pair<std::vector<Simplex>, std::vector<std::vector<Appearance>>> result = create_bifiltration(dist, max_dimension, m_max, closed);

    // std::cout << "Simplices:" << std::endl;
    // for (const Simplex& simplex : result.first) {
    //     for (int vertex : simplex) {
    //         std::cout << vertex << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Appearances:" << std::endl;
    // for (const std::vector<Appearance>& appearance : result.second) {
    //     for (const Appearance& pair : appearance) {
    //         std::cout << "(" << pair.first << ", " << pair.second << ") ";
    //     }
    //     std::cout << std::endl;
    // }
    
    std::cout << "saving RIVET format..." << std::endl;

    std::ofstream out(filename+"mneighbor.bifi");

    out << "--datatype bifiltration\n--xlabel distance\n--ylabel num_neighbors\n--yreverse\n";
    for (int i =0; i<result.first.size();i++){
        for (int vertex : result.first[i]){
                    out << vertex << ' ';

        }
        out << ";";
        for (const Appearance& pair : result.second[i]){
            out << ' ' << pair.first << ' ' << pair.second;
        }
        out << std::endl;
    }
    out.close();
    return 0;
}
