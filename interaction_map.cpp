#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

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
// 4) Binning: group COMs in blocks of nucleosomes_per_bin
// ----------------------------------------------------------------------------
std::vector< std::vector<double> > bin_coms(const std::vector< std::vector<double> >& com_array,
                                           int nucleosomes_per_bin)
{
    // com_array is shape (n, 3)
    int n = static_cast<int>(com_array.size());
    int n_bins = n / nucleosomes_per_bin; // integer division
    std::vector< std::vector<double> > binned;
    binned.reserve(n_bins);

    for(int i=0; i < n_bins; i++) {
        int start = i * nucleosomes_per_bin;
        int end   = start + nucleosomes_per_bin;
        double sx=0.0, sy=0.0, sz=0.0;
        for(int k=start; k < end; k++) {
            sx += com_array[k][0];
            sy += com_array[k][1];
            sz += com_array[k][2];
        }
        double bin_count = static_cast<double>(end - start);
        binned.push_back({sx/bin_count, sy/bin_count, sz/bin_count});
    }
    return binned;
}

// ----------------------------------------------------------------------------
// main analysis flow
// ----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // User parameters (you can adjust or make them command-line arguments)
    std::string xyz_file = "fiber_new.xyz";
    //std::string xyz_file = "ac_test.xyz";
    int nucleosomes_per_bin = 5;
    double contact_threshold = 250.0; // in angstrom
    std::string output_file = "contact_map_oe" + std::to_string(nucleosomes_per_bin) + ".txt";
    //std::string output_file = "contact_map_oe.txt";

    // Read frames from xyz
    std::cout << "Reading frames from " << xyz_file << "...\n";
    std::vector< std::vector<Atom> > frames = read_xyz_frames(xyz_file);
    if(frames.empty()) {
        std::cerr << "No frames read from file.\n";
        return 1;
    }
    //std::cout << "Number of frames: " << frames.size() << std::endl;

    // Determine number of bins from the first frame
    // 1) parse => nucleosomes => COM => bin
    {
        // We'll do a quick parse on the first frame
        auto nucs = split_into_nucleosomes(frames[0]);
        std::vector< std::vector<double> > coms;
        for(const auto &nuc : nucs) {
            coms.push_back( compute_com(nuc) );
        }
        // bin
        auto binned = bin_coms(coms, nucleosomes_per_bin);
        if(binned.empty()) {
            std::cerr << "First frame yields no bins. Check your data.\n";
            return 1;
        }
        // n_bins
        //int n_bins = binned.size();
        //std::cout << "n_bins = " << n_bins << "\n";
    }

    // We'll now do the entire pass again to accumulate contact data
    //   But we need to figure out the final n_bins from the first frame
    auto nucs_first = split_into_nucleosomes(frames[0]);
    std::vector< std::vector<double> > coms_first;
    for(const auto &nuc : nucs_first) {
        coms_first.push_back( compute_com(nuc) );
    }
    auto binned_first = bin_coms(coms_first, nucleosomes_per_bin);
    int n_bins = binned_first.size();

    // Create a 2D contact matrix
    std::vector<double> contact_matrix(n_bins * n_bins, 0.0);
    int n_frames_used = 0;

    // For each frame:
    for(const auto &frame_atoms : frames) {
        // 1) nucleosome splitting
        auto nucs = split_into_nucleosomes(frame_atoms);

        // 2) COMs
        std::vector< std::vector<double> > coms;
        coms.reserve(nucs.size());
        for(const auto &nuc : nucs) {
            coms.push_back( compute_com(nuc) );
        }

        // 3) bin them
        auto binned = bin_coms(coms, nucleosomes_per_bin);
        if(static_cast<int>(binned.size()) != n_bins) {
            // skip if mismatch
            continue;
        }

        // 4) compute pairwise distances, then contact
        // contact_frame(i,j) = 1 if dist(i,j) < threshold else 0
        // We'll accumulate in a temp vector
        std::vector<double> contact_frame(n_bins * n_bins, 0.0);

        for(int i=0; i < n_bins; i++) {
            for(int j=0; j < n_bins; j++) {
                if(i == j) {
                    // remove self-contacts => 0
                    contact_frame[i*n_bins + j] = 0.0;
                    continue;
                }
                double dx = binned[i][0] - binned[j][0];
                double dy = binned[i][1] - binned[j][1];
                double dz = binned[i][2] - binned[j][2];
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                if(dist < contact_threshold) {
                    contact_frame[i*n_bins + j] = 1.0;
                } else {
                    contact_frame[i*n_bins + j] = 0.0;
                }
            }
        }

        //std::cout << "contact map \n\n";	
        //for (const auto& elem : contact_frame) {
        //    std::cout << elem << " ";
        //}
        //std::cout << "\n\n";	

        // add to contact_matrix
        for(int idx=0; idx < n_bins*n_bins; idx++) {
            contact_matrix[idx] += contact_frame[idx];
        }
        n_frames_used++;
    }

    if(n_frames_used == 0) {
        std::cerr << "No frames used. Possibly all were skipped.\n";
        return 1;
    }

    // Average
    for(int idx=0; idx < n_bins*n_bins; idx++) {
        contact_matrix[idx] /= (double)n_frames_used;
    }

    // ------------------------------------------------------------------------
    // Observed/Expected Normalization
    // ------------------------------------------------------------------------
    // Step 1) Compute expected_by_sep for each separation
    std::vector<double> expected_by_sep(n_bins, 0.0);
    std::vector<double> counts_by_sep(n_bins,   0.0);

    for(int i=0; i < n_bins; i++) {
        for(int j=0; j < n_bins; j++) {
            int sep = std::abs(j - i);
            expected_by_sep[sep] += contact_matrix[i*n_bins + j];
            counts_by_sep[sep]   += 1.0;
        }
    }
    for(int sep=0; sep < n_bins; sep++) {
        if(counts_by_sep[sep] > 0.0) {
            expected_by_sep[sep] /= counts_by_sep[sep];
        }
    }

    // Step 2) Build O/E matrix
    std::vector<double> oe_matrix(n_bins*n_bins, 0.0);

    for(int i=0; i < n_bins; i++) {
        for(int j=0; j < n_bins; j++) {
            int sep = std::abs(j - i);
            double exp_val = expected_by_sep[sep];
            double obs_val = contact_matrix[i*n_bins + j];
            if(exp_val > 0.0) {
                oe_matrix[i*n_bins + j] = obs_val / exp_val;
            } else {
                oe_matrix[i*n_bins + j] = 0.0;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Write O/E matrix to a text file for Python plotting
    // (We will write as n_bins lines, each line has n_bins columns)
    // ------------------------------------------------------------------------
    std::ofstream out(output_file);
    if(!out.is_open()) {
        std::cerr << "Could not open output file " << output_file << " for writing.\n";
        return 1;
    }
    // Write matrix dimension on first line (optional if you prefer)
    out << n_bins << " " << n_bins << "\n";

    for(int i=0; i < n_bins; i++) {
        for(int j=0; j < n_bins; j++) {
            out << oe_matrix[i*n_bins + j];
            if(j < n_bins-1) out << " ";
        }
        out << "\n";
    }
    out.close();

    //std::cout << "Wrote O/E matrix to " << output_file << "\n";
    std::cout << "Done.\n";
    return 0;
}

