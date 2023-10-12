/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Hesam Arabzadeh, University of Missouri, hacr6@missouri.edu

#include "pair_lock_key.h"
#include "fix_rot_brownian.h"

#include "fix.h"
#include "atom.h"
#include "comm.h"
#include "utils.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <unordered_map>


using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

PairLockKey::PairLockKey(LAMMPS *lmp) : Pair(lmp), cut(nullptr)
{
    single_enable = 1;
    respa_enable = 0;
    writedata = 1;

}

/* ---------------------------------------------------------------------- */

PairLockKey::~PairLockKey()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
  if (isSiteCoordsFileOpen) {
  siteCoordsFile.close();
  }
}

/* ---------------------------------------------------------------------- */

void PairLockKey::compute(int eflag, int vflag)
{
  // Declare variables
  int i, j, ii, jj, inum, jnum, type_i, type_j;
  double delx, dely, delz, evdwl, fpair;
  double A, r0, sigma, siteCutoff, gaussValueAtCutoff, gaussValue, rsq, r, dr, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double site_i_local[3], site_j_local[3], site_i_global[3], site_j_global[3], site_i_pos[3], site_j_pos[3];
  double *quaternion_p1, *quaternion_p2;
  double *pos_i, *pos_j, *old_pos_i, *old_pos_j;

  // Initialize energy to zero
  evdwl = 0.0;

  // Initialize energy and virial computation
  ev_init(eflag, vflag);

  // Get pointers to atom properties
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  Coords coords1;
  Coords coords2;

  // A stringstream to hold the site coordinates data
  std::stringstream siteCoordsData;

  // Read the site coordinates into siteCoordsMap
  readSiteCoords();

  // Get neighbor list information
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
 
  // Outer loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    type_i = type[i];
    pos_i = x[i];

    std::string particle1_id = std::to_string(type_i);

    // Check if particle1_id exists in the siteTable
    auto it_1 = siteTable.find(particle1_id);
    if (it_1 == siteTable.end()) continue;

    // Get the sites of the first particle (in the local frame)
    auto& particle1_sites = it_1->second;

    //get the quaternions of the particle ii
    quaternion_p1 = fixRotBrownian->getQuaternion(ii);

    // Loop over all neighbors of atom i within the cutoff distance
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; ++jj) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      // Get the type and position of atom j
      type_j = type[j];
      pos_j = x[j];

      std::string particle2_id = std::to_string(type_j);

      // Check if particle2_id exists in the siteTable
      auto it_2 = siteTable.find(particle2_id);
      if (it_2 == siteTable.end()) continue;

      // Get the sites of the second particle (in the local frame)
      auto& particle2_sites = it_2->second;

      //get the quaternions of the particle jj
      quaternion_p2 = fixRotBrownian->getQuaternion(jj);  // Get quaternion for particle j

      // Iterate over all pairs of sites
      for (const auto& site1 : particle1_sites) {
        for (const auto& site2 : particle2_sites) {
          std::string interaction_id = (site1.siteID < site2.siteID) 
                                        ? (site1.siteID + "-" + site2.siteID) 
                                        : (site2.siteID + "-" + site1.siteID);      

          // Look up the interaction parameters in the interactingSitesTable
	  auto it_interaction = interactingSitesTable.find(interaction_id);
          if (it_interaction == interactingSitesTable.end()) continue;

//          std::string message = (it_interaction != interactingSitesTable.end()) ? "Key it_interaction was found" : "Key it_interaction was not found";
//          utils::logmesg(lmp, "\n     ***** " + message + " *****\n\n");

          InteractingSites interaction_params = it_interaction->second;

          r0 = interaction_params.lockKeyPrm[0];
	  A = interaction_params.lockKeyPrm[1];
          sigma = interaction_params.lockKeyPrm[2];
	  siteCutoff = interaction_params.lockKeyPrm[3];

	  // Precompute gaussValueAtCutoff
          gaussValueAtCutoff = -A * exp(-(siteCutoff - r0) * (siteCutoff - r0) / (2.0 * sigma * sigma));

          // Create unique identifiers for the sites
          std::string site1_id = std::to_string(i+1) + "_" + site1.siteID;
          std::string site2_id = std::to_string(j+1) + "_" + site2.siteID;

          // Retrieve the coordinates from siteCoordsMap
	  auto it_coords1 = siteCoordsMap.find(site1_id);
          auto it_coords2 = siteCoordsMap.find(site2_id);
          if (it_coords1 == siteCoordsMap.end() || it_coords2 == siteCoordsMap.end()) continue;

	  // Capture the start time
          //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

          coords1 = it_coords1->second;
          coords2 = it_coords2->second;

          //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
          //std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
          //std::cout << "\n     ***** It took " << time_span.count() << " seconds. *****" << std::endl;

          //std::cout << "     ***** coords 1: " << coords1.x << "     " << coords1.y << "     " << coords1.z << "     *****\n";
          //std::cout << "     ***** coords 2: " << coords2.x << "     " << coords2.y << "     " << coords2.z << "     *****\n\n";
                 
	  // Copy the original coordinates to local variables
          site_i_local[0] = coords1.x; site_i_local[1] = coords1.y; site_i_local[2] = coords1.z;
          site_j_local[0] = coords2.x; site_j_local[1] = coords2.y; site_j_local[2] = coords2.z;
 
//          std::cout << "     ***** site_i_local: " << site_i_local[0] << "     " << site_i_local[1] << "     " << site_i_local[2] << "     *****\n";
//          std::cout << "     ***** site_j_local: " << site_j_local[0] << "     " << site_j_local[1] << "     " << site_j_local[2] << "     *****\n\n";

          // Rotate the local coordinates to the global frame using the quaternions
          PairLockKey::rotateVectorByQuaternion(site_i_local, quaternion_p1, site_i_global);
          PairLockKey::rotateVectorByQuaternion(site_j_local, quaternion_p2, site_j_global);

//          std::cout << "     ***** site_i_global: " << site_i_global[0] << "     " << site_i_global[1] << "     " << site_i_global[2] << "     *****\n";
//          std::cout << "     ***** site_j_global: " << site_j_global[0] << "     " << site_j_global[1] << "     " << site_j_global[2] << "     *****\n\n";
          
	  // GET OLD COORDS FROM ROT/BROWN CODE
	  old_pos_i = fixRotBrownian->getOldCoords(i);
	  old_pos_j = fixRotBrownian->getOldCoords(j);

//          std::cout << "     ***** old_coords1: " << old_pos_i[0] << "     " << old_pos_i[1] << "     " << old_pos_i[2] << "     *****\n";
//          std::cout << "     ***** old_coords2: " << old_pos_j[0] << "     " << old_pos_j[1] << "     " << old_pos_j[2] << "     *****\n\n";
//
//          std::cout << "     ***** pos_i : " << pos_i[0] << "     " << pos_i[1] << "     " << pos_i[2] << "     *****\n";
//          std::cout << "     ***** pos_j : " << pos_j[0] << "     " << pos_j[1] << "     " << pos_j[2] << "     *****\n\n";

          // Calculate the absolute position of each site (in the global frame)
          site_i_pos[0] = pos_i[0] - old_pos_i[0] + site_i_global[0]; site_i_pos[1] = pos_i[1] - old_pos_i[1] + site_i_global[1]; site_i_pos[2] = pos_i[2] - old_pos_i[2] + site_i_global[2];
          site_j_pos[0] = pos_j[0] - old_pos_j[0] + site_j_global[0]; site_j_pos[1] = pos_j[1] - old_pos_j[1] + site_j_global[1]; site_j_pos[2] = pos_j[2] - old_pos_j[2] + site_j_global[2];

//          std::cout << "     ***** site_i_pos: " << site_i_pos[0] << "     " << site_i_pos[1] << "     " << site_i_pos[2] << "     *****\n";
//          std::cout << "     ***** site_j_pos: " << site_j_pos[0] << "     " << site_j_pos[1] << "     " << site_j_pos[2] << "     *****\n\n";

	  siteCoordsMap[site1_id] = {site_i_pos[0], site_i_pos[1], site_i_pos[2]};
          siteCoordsMap[site2_id] = {site_j_pos[0], site_j_pos[1], site_j_pos[2]};

          if (writeSiteCoords) {
//            utils::logmesg(lmp, "\n     ***** Stored sites' coordinates into siteCoordsData! *****\n\n");
	    siteCoordsData << site1_id << ", " << site_i_pos[0] << ", " << site_i_pos[1] << ", " << site_i_pos[2] << "\n";
            siteCoordsData << site2_id << ", " << site_j_pos[0] << ", " << site_j_pos[1] << ", " << site_j_pos[2] << "\n";
          }
      
          // Compute the distance between site_i and site_j
          delx = site_i_pos[0] - site_j_pos[0];
          dely = site_i_pos[1] - site_j_pos[1];
          delz = site_i_pos[2] - site_j_pos[2];
          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq < (siteCutoff * siteCutoff)) {
            r = sqrt(rsq);	    
	    dr = r - r0;

            gaussValue = -A * exp(-(dr * dr) / (2.0 * sigma * sigma));
            fpair = (dr / (sigma * sigma)) * gaussValue; 

	    fpair *= factor_lj / r;

	    f[i][0] += delx * fpair;
            f[i][1] += dely * fpair;
            f[i][2] += delz * fpair;
            if (newton_pair || j < nlocal) {
              f[j][0] -= delx * fpair;
              f[j][1] -= dely * fpair;
              f[j][2] -= delz * fpair;
            }

	    if (eflag) evdwl = factor_lj * (gaussValue - gaussValueAtCutoff); 
            if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
          }
        }
      }
    }
  }
  // Update the sites_coords_table.xyz file with the latest coordinates
  updateSiteCoordsFile();
  // Write the collected site coordinates to the file
  if (writeSiteCoords && rsq < (siteCutoff * siteCutoff)) {

    // Write the header for each simulation step
    siteCoordsFile << "SiteID, X, Y, Z\n";

    siteCoordsFile << siteCoordsData.str();
    siteCoordsData.str("");  // Clear the stringstream for the next iteration
    siteCoordsData.clear();  // Clear any error flags
  }
  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   Rotate vector (sites) by quaternions 
------------------------------------------------------------------------- */

void PairLockKey::rotateVectorByQuaternion(const double *vec, const double *quat, double *result) {
  // Quaternion multiplication: result = quat * vec * conj(quat)
  
  double tmp[4];    // Temporary quaternion to hold intermediate results
  
  // First, form the quaternion representation of vec
  double vec_quat[4] = {0.0, vec[0], vec[1], vec[2]};
  
  // Step 1: tmp = quat * vec_quat
  tmp[0] =  quat[0]*vec_quat[0] - quat[1]*vec_quat[1] - quat[2]*vec_quat[2] - quat[3]*vec_quat[3];
  tmp[1] =  quat[0]*vec_quat[1] + quat[1]*vec_quat[0] - quat[2]*vec_quat[3] + quat[3]*vec_quat[2];
  tmp[2] =  quat[0]*vec_quat[2] + quat[1]*vec_quat[3] + quat[2]*vec_quat[0] - quat[3]*vec_quat[1];
  tmp[3] =  quat[0]*vec_quat[3] - quat[1]*vec_quat[2] + quat[2]*vec_quat[1] + quat[3]*vec_quat[0];
  
  // Step 2: result = tmp * conj(quat), 
  // NOTE!! we don't consider the real/scalar part of the quaternions 
  result[0] = quat[0]*tmp[1] - quat[1]*tmp[0] + quat[2]*tmp[3] - quat[3]*tmp[2];
  result[1] = quat[0]*tmp[2] - quat[1]*tmp[3] - quat[2]*tmp[0] + quat[3]*tmp[1];
  result[2] = quat[0]*tmp[3] + quat[1]*tmp[2] - quat[2]*tmp[1] - quat[3]*tmp[0];

}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLockKey::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLockKey::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Pair style lock/key must have exactly one argument");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLockKey::coeff(int narg, char **arg)
{
  // Check that the correct number of arguments has been provided
  if (narg < 3 || narg > 4) error->all(FLERR, "Incorrect args for pair coefficients");

  if (!allocated) allocate();

  // Get the path to the CSV file from the arguments
  std::string csvPath = arg[2];

  // Load the data from the CSV file into your lookup tables
  loadLookupTables(csvPath);

  // Initialize the flag for writing the sites coordinates into a file
  writeSiteCoords = false;  

  // Set the flag based on the second input argument
  if (narg == 4 || narg == 5) 
    writeSiteCoords = (atoi(arg[narg -1]) == 1);

  // Checking if 'sites_traj.xyz' is available for writing the trajectory into
  if (writeSiteCoords && !isSiteCoordsFileOpen) {
    siteCoordsFile.open("sites_traj.xyz", std::ios::app);
    if (!siteCoordsFile.is_open()) {
      // If opening sites_traj.xyz fails, copy the content of sites_coords_table.xyz to sites_traj.xyz
      std::ifstream src("sites_coords_table.xyz", std::ios::binary);
      std::ofstream dst("sites_traj.xyz", std::ios::binary);
      dst << src.rdbuf();
      // Now try opening sites_traj.xyz again
      siteCoordsFile.open("sites_traj.xyz", std::ios::app);
      if (!siteCoordsFile.is_open()) {
        // If it still fails, then you might want to log a warning message
        utils::logmesg(lmp, "\n     ***** Warning: Could not open sites_traj.xyz for writing. *****\n");
      }
    }
    isSiteCoordsFileOpen = true;
  }

  std::string writeSiteCoordsStr = writeSiteCoords ? "true" : "false";
  utils::logmesg(lmp, "\n     *****  writeSiteCoords is set to " + writeSiteCoordsStr + ", a trajectory of sites' coordinates are written to sites_traj.xyz! *****\n\n");

  // reset per-type pair cutoffs that have been explicitly set previously
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++){
      for (int j = i; j <= atom->ntypes; j++){
        cut[i][j] = cut_global;
        setflag[i][j] = 1;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   load data from CSV file into lookup tables
---------------------------------------------------------------------- */

void PairLockKey::loadLookupTables(const std::string &csvPath)
{

  // Open the CSV file
  std::ifstream file(csvPath);
  if (!file.is_open())
    error->all(FLERR, "\n     ***** Could not open '" + csvPath + "' CSV file ***** \n\n"); 
  else
    utils::logmesg(lmp, "\n     ***** " + csvPath + " was accessed! ***** \n\n");
  

  std::string line;
  int line_counter = 0;
  TableState currentState = NONE;

  while (std::getline(file, line)) {

    std::istringstream ss(line);
    std::vector<std::string> cells;
    std::string cell;
    line_counter += 1;

    while (std::getline(ss, cell, ',')) {
      cell.erase(remove_if(cell.begin(), cell.end(), isspace), cell.end());
      cells.push_back(cell);
    }

    // Determine the current state (which table we are reading)
    if (line == "# Protein table") {
      currentState = PROTEIN;
    }
    else if (line == "# Site table") {
      currentState = SITE;
    }
    else if (line == "# Interacting sites table") {
      currentState = INTERACTING_SITES;
    }
    else if (line.find_first_not_of(" \t\n\r") != std::string::npos && line[0] != '#') {
      // Process the data row
      while (std::getline(ss, cell, ',')) {
          cell.erase(remove_if(cell.begin(), cell.end(), isspace), cell.end());
          cells.push_back(cell);
      }

      // Handle data based on the current state
      switch (currentState) {
          case PROTEIN: {
              if (!(cells.size() ==  5)) {
		std::string errorMsg = "Error: on line " + std::to_string(line_counter) + " expected 5 columns, but got " + std::to_string(cells.size()) + ", on line: " + line + "\n\n";

                error->all(FLERR, errorMsg.c_str());
              }
              Protein proteinEntry;
              proteinEntry.lammpsType = cells[0];
              proteinEntry.fragName   = cells[1];
              proteinEntry.uniProtID = cells[2];

              try {
              proteinEntry.radius = std::stod(cells[3]);
              }
              catch (const std::invalid_argument& ia) {
                std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                          + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[3] + "' to integer on line: " + line + "\n\n";
                error->all(FLERR, errorMsg.c_str());
              }
              try {
              proteinEntry.mass = std::stod(cells[4]);
              }
              catch (const std::invalid_argument& ia) {
                std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                          + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[4] + "' to double on line: " + line + "\n\n";
                error->all(FLERR, errorMsg.c_str());
              }
              //Insert the protein data into the proteinTable using LAMMPS type as the key
              proteinTable[proteinEntry.lammpsType] = proteinEntry;
              break;

          }

          case SITE: {

              if (!(cells.size() ==  2)) {
                std::string errorMsg = "Error: on line " + std::to_string(line_counter) + " expected 2 columns, but got " + std::to_string(cells.size()) + ", on line: " + line + "\n\n";
                error->all(FLERR, errorMsg.c_str());
              }
              Site  siteEntry;
              siteEntry.protID = cells[0];
              siteEntry.siteID = cells[1];

              // If the protein ID does not exist in the siteTable, add it with an empty vector
              if (siteTable.find(siteEntry.protID) == siteTable.end()) {
                  siteTable[siteEntry.protID] = std::vector<Site>();
              }

              // Append the site to the vector of sites associated with the protein ID
              siteTable[siteEntry.protID].push_back(siteEntry);

              break;
          }

          case INTERACTING_SITES: {

              if (!(cells.size() ==  5)) {
                std::string errorMsg = "Error: on line " + std::to_string(line_counter) + " expected 5 columns, but got " + std::to_string(cells.size()) + ", on line: " + line + "\n\n";
                error->all(FLERR, errorMsg.c_str());
              }
              InteractingSites interactingSitesEntry;
              interactingSitesEntry.intID = cells[0];
              interactingSitesEntry.lockKeyPrm.resize(4);

              for (int i=0;i<4;++i){
                try{
                  interactingSitesEntry.lockKeyPrm[i] = std::stod(cells[i+1]);
                }
                catch (const std::invalid_argument& ia) {
                  std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                    + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[i+1] + "' to double on line: " + line + "\n\n";
                  error->all(FLERR, errorMsg.c_str());
                }
              }

              // Append the site to the vector of sites associated with the protein ID
              interactingSitesTable[interactingSitesEntry.intID] = interactingSitesEntry;

              break;
          }
          default: {
            // Handle unrecognized states or lines if necessary
            std::string errorMsg = "Warning: on line "+ std::to_string(line_counter) + " encountered line outside of recognized tables: " + line + '\n';
            error->all(FLERR, errorMsg.c_str());
            break;
          }
        }
    }
  }
  // Close the CSV file
  file.close();
}

/* -------------------------------------------------------------------------
   write the updated site coordinates back into file sites_coords_table.xyz 
--------------------------------------------------------------------------- */

void PairLockKey::updateSiteCoordsFile() {

  // Open the file in read-write mode
  std::ifstream infile("sites_coords_table.xyz");
  if (!infile.is_open()) {
    error->all(FLERR, "Error opening sites_coords_table.xyz for updating.");
    return;
  }

  // Read the file into memory
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
      lines.push_back(line);
  }
  infile.close();

  // Modify the lines in memory
  for (std::string& currentLine : lines) {
    std::istringstream iss(currentLine);
    std::string siteID;
    std::getline(iss, siteID, ',');

    if (siteCoordsMap.find(siteID) != siteCoordsMap.end()) {
        Coords coords = siteCoordsMap[siteID];
        currentLine = siteID + ", " + std::to_string(coords.x) + ", " + std::to_string(coords.y) + ", " + std::to_string(coords.z);
    }
  }

  // Write the modified content back to the file
  std::ofstream outfile("sites_coords_table.xyz");
  if (!outfile.is_open())
     error->all(FLERR, "Error opening sites_coords_table.xyz for writing.");

  for (const std::string& currentLine : lines) {
      outfile << currentLine << "\n";
  }
  outfile.close();
}

/* ----------------------------------------------------------------------
   load data from Site coordinates file and populate the unordered_map
---------------------------------------------------------------------- */

void PairLockKey::readSiteCoords() {
  std::ifstream file("sites_coords_table.xyz");
  if (!file.is_open()) {
    error->all(FLERR, "Error opening sites_coords_table.xyz");
    return;
  }

  std::string line;
  int line_counter = 0;
  TableState currentState = NONE;

  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::vector<std::string> cells;
    std::string cell;
    line_counter += 1;

    while (std::getline(ss, cell, ',')) {
      cell.erase(remove_if(cell.begin(), cell.end(), isspace), cell.end());
      cells.push_back(cell);
    }

    // Determine the current state (which table we are reading)
    if (line == "SiteID, X, Y, Z") {
      currentState = SITE_COORDS;
    }
    else if (line.find_first_not_of(" \t\n\r") != std::string::npos && line[0] != '#') {
      // Handle data based on the current state
      switch (currentState) {
        case SITE_COORDS: {
          if (!(cells.size() == 4)) {
            std::string errorMsg = "Error: on line " + std::to_string(line_counter) + " expected 4 columns, but got " + std::to_string(cells.size()) + ", on line: " + line;
            error->all(FLERR, errorMsg.c_str());
            continue;
          }
          Coords coords;
          try {
            coords.x = std::stod(cells[1]);
            coords.y = std::stod(cells[2]);
            coords.z = std::stod(cells[3]);
          }
          catch (const std::invalid_argument& ia) {
            std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                      + "On line " + std::to_string(line_counter) + " could not convert value to double on line: " + line;
            error->all(FLERR, errorMsg.c_str());
            continue;
          }
          siteCoordsMap[cells[0]] = coords; 
          break;
        }
        default: {
          // Handle unrecognized states or lines if necessary
          std::string errorMsg = "Warning: on line "+ std::to_string(line_counter) + " encountered line outside of recognized tables: " + line;
          error->all(FLERR, errorMsg.c_str());
          break;
        }
      }
    }
  }
  // Close the file
  file.close();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLockKey::init_one(int i, int j)
{

  // Check that all necessary parameters have been set
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  // Access the rotational Brownian fix (FixProRotation class) to get access to the quaternions array
  int ifix = modify->find_fix("1");
  if (ifix < 0) error->all(FLERR, "Fix rot/brownian is capable of returning quaternions for rotational motion and is not found!");
  fixRotBrownian = (FixRotBrownian *) modify->fix[ifix];

  // Return the maximum cutoff distance for this pair of atom types
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairLockKey::single(int i, int j, int type_i, int type_j, double rsq,
                           double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r = sqrt(rsq);
  double totalForce = 0.0;
  double totalPotential = 0.0;

  double A, r0, sigma, gaussForce;

  // Loop over sites on atom type 'type_i'
  for (const auto& site1 : siteTable[std::to_string(type_i)]) {
    // Loop over sites on atom type 'type_j'
    for (const auto& site2 : siteTable[std::to_string(type_j)]) {
      std::string interaction_id = (site1.siteID < site2.siteID)
                                   ? (site1.siteID + "-" + site2.siteID)
                                   : (site2.siteID + "-" + site1.siteID);

      if (interactingSitesTable.count(interaction_id) > 0) {
        InteractingSites interaction_params = interactingSitesTable[interaction_id];

        r0 = interaction_params.lockKeyPrm[0];
        A = interaction_params.lockKeyPrm[1];
        sigma = interaction_params.lockKeyPrm[2];

        double dr = r - r0;  // Declare and initialize dr

        double gaussValue = -A * exp(-(dr * dr) / (2.0 * sigma * sigma));
        gaussForce = -A * (dr / (sigma * sigma)) * exp(-(dr * dr) / (2.0 * sigma * sigma));

        totalForce += gaussForce;
        totalPotential -= gaussValue;
      }
    }
  }
  fforce = factor_lj * totalForce / r;
  return factor_lj * totalPotential;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLockKey::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
}

void PairLockKey::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}


/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLockKey::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}
