#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <random>
#include <numeric>


// A simple struct to hold atom data
struct Atom {
    int type;
    double x, y, z;
};

// ----------------------------------------------------------------------------
// 1) Read .xyz frames into a vector of frames, where each frame is a list of Atoms
// ----------------------------------------------------------------------------
std::vector< std::vector<Atom> > read_xyz_frames(const std::string& filename)
{
    std::ifstream infile(filename);
    if(!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector< std::vector<Atom> > frames;
    std::string line;

    while(true) {
        // Read the number of atoms line
        if(!std::getline(infile, line)) break; // no more lines => done
        if(line.empty()) break;

        int natoms = std::stoi(line);

        // Read the comment line (skip)
        if(!std::getline(infile, line)) break;

        // Now read the next natoms lines for atom data
        std::vector<Atom> frame_atoms;
        frame_atoms.reserve(natoms);
        for(int i=0; i < natoms; i++) {
            if(!std::getline(infile, line)) break;
            std::istringstream iss(line);

            Atom atom;
            iss >> atom.type >> atom.x >> atom.y >> atom.z;
            frame_atoms.push_back(atom);
        }
        if(!frame_atoms.empty()) {
            frames.push_back(frame_atoms);
        }

        // If we couldn't read exactly natoms lines, we probably reached EOF
        if(frame_atoms.size() < (size_t)natoms) {
            break;
        }
    }

    infile.close();
    return frames;
}

// -----------------------------------------------------------------------------
// Function to randomly select N pairs of nucleosomes
// -----------------------------------------------------------------------------
std::vector<std::pair<int, int>> select_random_pairs(int num_nucleosomes, int num_pairs) {
    std::vector<std::pair<int, int>> selected_pairs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_nucleosomes - 1);

    for (int i = 0; i < num_pairs; i++) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        while (idx1 == idx2) { // Ensure they are not the same
            idx2 = dis(gen);
        }
        selected_pairs.push_back({idx1, idx2});
    }
    return selected_pairs;
}


// ----------------------------------------------------------------------------
// 2) Split a frame's atoms into nucleosomes, as in your Python logic
// ----------------------------------------------------------------------------
std::vector< std::vector<Atom> > split_into_nucleosomes(const std::vector<Atom>& atoms)
{
    std::vector< std::vector<Atom> > nucleosomes;
    std::vector<Atom> current_nuc;
    int h3_count = 0;

    for(const auto &atom : atoms) {
        if(atom.type == 1) { // type==1 is H3
            if(current_nuc.empty()) {
                // start new nucleosome
                current_nuc.push_back(atom);
                h3_count = 1;
            } else {
                if(h3_count < 2) {
                    current_nuc.push_back(atom);
                    h3_count++;
                } else {
                    // we already have 2 H3 in current block => start new
                    nucleosomes.push_back(current_nuc);
                    current_nuc.clear();
                    current_nuc.push_back(atom);
                    h3_count = 1;
                }
            }
        } else {
            current_nuc.push_back(atom);
        }
    }
    // push the last nucleosome if any
    if(!current_nuc.empty()) {
        nucleosomes.push_back(current_nuc);
    }

    return nucleosomes;
}

// ----------------------------------------------------------------------------
// 3) Compute the center-of-mass for one nucleosome
// ----------------------------------------------------------------------------
std::vector<double> compute_com(const std::vector<Atom>& nucleosome)
{
    // We'll return {x, y, z}
    double sumx=0.0, sumy=0.0, sumz=0.0;
    for(const auto &a : nucleosome) {
        sumx += a.x;
        sumy += a.y;
        sumz += a.z;
    }
    double count = static_cast<double>(nucleosome.size());
    return {sumx / count, sumy / count, sumz / count};
}


// ----------------------------------------------------------------------------
// Function to compute Euclidean distance between two centers of mass
// ----------------------------------------------------------------------------
double compute_distance(const std::vector<double>& com1, const std::vector<double>& com2) {
    return std::sqrt(std::pow(com1[0] - com2[0], 2) +
                     std::pow(com1[1] - com2[1], 2) +
                     std::pow(com1[2] - com2[2], 2));
}


// ----------------------------------------------------------------------------
// Function to compute distances over all frames and store them
// ----------------------------------------------------------------------------
std::vector<std::vector<double>> compute_distances_over_frames(
    const std::vector<std::vector<std::vector<double>>>& com_frames,
    const std::vector<std::pair<int, int>>& selected_pairs)
{
    std::vector<std::vector<double>> distances(selected_pairs.size());

    for (size_t frame_idx = 0; frame_idx < com_frames.size(); ++frame_idx) {
        const auto& coms = com_frames[frame_idx];
        for (size_t pair_idx = 0; pair_idx < selected_pairs.size(); ++pair_idx) {
            int idx1 = selected_pairs[pair_idx].first;
            int idx2 = selected_pairs[pair_idx].second;
            if (idx1 < coms.size() && idx2 < coms.size()) {
                double dist = compute_distance(coms[idx1], coms[idx2]);
                distances[pair_idx].push_back(dist);
            }
        }
    }
    return distances;
}
 

// ----------------------------------------------------------------------------
// Compute the autocorrelation function for a single pair
// ----------------------------------------------------------------------------
std::vector<double> compute_autocorrelation(const std::vector<double>& distances) {
    size_t N = distances.size();
    std::vector<double> autocorr(N, 0.0);
    
    double mean = std::accumulate(distances.begin(), distances.end(), 0.0) / N;

    for (size_t tau = 0; tau < N; ++tau) {
        double numerator = 0.0, denominator = 0.0;
        for (size_t t = 0; t < N - tau; ++t) {
            numerator += (distances[t] - mean) * (distances[t + tau] - mean);
        }
        for (size_t t = 0; t < N; ++t) {
            denominator += (distances[t] - mean) * (distances[t] - mean);
        }
        autocorr[tau] = numerator / denominator;
    }
    return autocorr;
}


// ----------------------------------------------------------------------------
// Function to write distances to file
// ----------------------------------------------------------------------------
void write_distances_to_file(const std::string& filename, 
                             const std::vector<std::vector<double>>& distances, 
                             const std::vector<std::pair<int, int>>& selected_pairs) 
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing!\n";
        return;
    }

    // Write header
    file << "# Pair_Index Nucleosome1 Nucleosome2 Distances_per_Frame\n";

    for (size_t i = 0; i < distances.size(); ++i) {
        file << i << " " << selected_pairs[i].first << " " << selected_pairs[i].second << " ";
        for (double d : distances[i]) {
            file << d << " ";
        }
        file << "\n";
    }
    file.close();
}

// ----------------------------------------------------------------------------
// Function to write autocorrelation results to file
// ----------------------------------------------------------------------------
void write_autocorrelation_to_file(const std::string& filename, 
                                   const std::vector<std::vector<double>>& autocorr_data, 
                                   const std::vector<std::pair<int, int>>& selected_pairs) 
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing!\n";
        return;
    }

    // Write header
    file << "# Pair_Index Nucleosome1 Nucleosome2 Autocorrelation_per_Lag\n";

    for (size_t i = 0; i < autocorr_data.size(); ++i) {
        file << i << " " << selected_pairs[i].first << " " << selected_pairs[i].second << " ";
        for (double val : autocorr_data[i]) {
            file << val << " ";
        }
        file << "\n";
    }
    file.close();
}


// ----------------------------------------------------------------------------
// main analysis flow
// ----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string xyz_file = "fiber_new.xyz";
    //std::string xyz_file = "test.xyz";
    int num_pairs = 10;  // Choose 10 random nucleosome pairs

    std::cout << "Reading frames from " << xyz_file << "...\n";
    std::vector<std::vector<Atom>> frames = read_xyz_frames(xyz_file);
    if (frames.empty()) {
        std::cerr << "No frames read from file.\n";
        return 1;
    }

    // Process nucleosomes and compute centers of mass
    std::vector<std::vector<std::vector<double>>> com_frames;
    for (const auto& frame_atoms : frames) {
        auto nucs = split_into_nucleosomes(frame_atoms);
        std::vector<std::vector<double>> coms;
        for (const auto& nuc : nucs) {
            coms.push_back(compute_com(nuc));
        }
        com_frames.push_back(coms);
    }

    // Select random nucleosome pairs
    //int num_nucleosomes = com_frames[0].size();
    //std::vector<std::pair<int, int>> selected_pairs = select_random_pairs(num_nucleosomes, num_pairs);

    // Manually enter nucleosome pair indices (modify these values as needed)
    int nuc1 = 0;  
    int nuc2 = 4999;

    std::vector<std::pair<int, int>> selected_pairs = { {nuc1, nuc2} };

    // Compute distances over time
    std::vector<std::vector<double>> distances = compute_distances_over_frames(com_frames, selected_pairs);
    write_distances_to_file("distances.txt", distances, selected_pairs);

    // Compute autocorrelation
    std::vector<std::vector<double>> autocorr_data;
    for (const auto& dist_series : distances) {
        autocorr_data.push_back(compute_autocorrelation(dist_series));
    }
    write_autocorrelation_to_file("autocorrelation.txt", autocorr_data, selected_pairs);

    std::cout << "Done.\n";
    return 0;
}
