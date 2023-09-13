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

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "modify.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>



using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

PairLockKey::PairLockKey(LAMMPS *lmp) : Pair(lmp), cut(nullptr), offset(nullptr)
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
    memory->destroy(offset);
    memory->destroy(torque);    
  }
}

/* ---------------------------------------------------------------------- */

void PairLockKey::compute(int eflag, int vflag)
{
  // Declare variables
  int i, j, ii, jj, inum, jnum, type_i, type_j;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double A, r0, sigma, gaussValueAtCutoff, gaussValue, rsq, r, dr, factor_lj;
  double torque_i[3], torque_j[3];
  int *ilist, *jlist, *numneigh, **firstneigh;
  double site_i_local[3], site_j_local[3], site_i_global[3], site_j_global[3], site_i_pos[3], site_j_pos[3];
  double *quaternion_p1, *quaternion_p2;


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

  // zero out the torque vectors since we're adding them here and LAMMPS is not aware of it
  for (int i = 0; i < nlocal; ++i) {
    torque[i][0] = 0.0;
    torque[i][1] = 0.0;
    torque[i][2] = 0.0;
  }

  // Get neighbor list information
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Outer loop over neighbors of my atoms
  for (int ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    type_i = type[i];
    double* pos_i = x[i];

    std::string particle1_id = std::to_string(type_i);

    // Check if particle1_id exists in the siteTable
    if (siteTable.count(particle1_id) == 0) continue;    

    // Get the sites and coordinates of the first particle (in the local frame)
    auto& particle1_sites = siteTable[particle1_id];

    //get the quaternions of the particle ii
    quaternion_p1 = fixRotBrownian->getQuaternion(ii);  // Get quaternion for particle i

    // Loop over all neighbors of atom i within the cutoff distance
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (int jj = 0; jj < jnum; ++jj) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      // Get the type and position of atom j
      type_j = type[j];
      double* pos_j = x[j];

      std::string particle2_id = std::to_string(type_j);

      // Check if particle2_id exists in the siteTable
      if (siteTable.count(particle2_id) == 0) continue;    

      // Get the sites and coordinates of the second particle (in the local frame)
      auto& particle2_sites = siteTable[particle2_id];

      //get the quaternions of the particle ii
      quaternion_p2 = fixRotBrownian->getQuaternion(jj);  // Get quaternion for particle i
      
      // Iterate over all pairs of sites
      for (const auto& site1 : particle1_sites) {
        for (const auto& site2 : particle2_sites) {
          std::string interaction_id = (site1.siteID < site2.siteID) 
                                        ? (site1.siteID + "-" + site2.siteID) 
                                        : (site2.siteID + "-" + site1.siteID);      
      
          // Look up the interaction parameters in the interactingSitesTable
	  if (interactingSitesTable.count(interaction_id) == 0) continue; 
          InteractingSites interaction_params = interactingSitesTable[interaction_id];

	  A = interaction_params.lockKeyPrm[1];
          r0 = interaction_params.lockKeyPrm[2];
          sigma = interaction_params.lockKeyPrm[3];
                 
          // Calculate the absolute position of each site (in the global frame)
	  site_i_local[0] = site1.coords[0], site_i_local[1] = site1.coords[1]; site_i_local[2] = site1.coords[2];
          site_j_local[0] = site2.coords[0]; site_j_local[1] = site2.coords[1]; site_j_local[2] = site2.coords[2];
          
          // Rotate the local coordinates to the global frame using the quaternions
          PairLockKey::rotateVectorByQuaternion(site_i_local, quaternion_p1, site_i_global);
          PairLockKey::rotateVectorByQuaternion(site_j_local, quaternion_p2, site_j_global);
          
          // Calculate the absolute position of each site (in the global frame)
          site_i_pos[0] = pos_i[0] + site_i_global[0]; site_i_pos[1] = pos_i[1] + site_i_global[1]; site_i_pos[2] = pos_i[2] + site_i_global[2];
          site_j_pos[0] = pos_j[0] + site_j_global[0]; site_j_pos[1] = pos_j[1] + site_j_global[1]; site_j_pos[2] = pos_j[2] + site_j_global[2];
      
          // Compute the distance between site_i and site_j
          delx = site_i_pos[0] - site_j_pos[0];
          dely = site_i_pos[1] - site_j_pos[1];
          delz = site_i_pos[2] - site_j_pos[2];
          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq < cutsq[type_i][type_j]) {
            r = sqrt(rsq);	    
	    dr = r - r0;
	    gaussValueAtCutoff = -A * exp(-(cut[i][j] - r0) * (cut[i][j] - r0) / (2.0 * sigma * sigma));
            gaussValue         = -A * exp(-(dr * dr) / (2.0 * sigma * sigma)) - gaussValueAtCutoff;

            fpair = A * (dr / (sigma * sigma)) * exp(-(dr * dr) / (2.0 * sigma * sigma));
	    fpair *= factor_lj / r;

	    f[i][0] += delx * fpair;
            f[i][1] += dely * fpair;
            f[i][2] += delz * fpair;
            if (newton_pair || j < nlocal) {
              f[j][0] -= delx * fpair;
              f[j][1] -= dely * fpair;
              f[j][2] -= delz * fpair;
            }
            // Calculate torque
            torque_i[0] = dely * f[i][2] - delz * f[i][1];
            torque_i[1] = delz * f[i][0] - delx * f[i][2];
            torque_i[2] = delx * f[i][1] - dely * f[i][0];

            torque_j[0] = dely * f[j][2] - delz * f[j][1];
            torque_j[1] = delz * f[j][0] - delx * f[j][2];
            torque_j[2] = delx * f[j][1] - dely * f[j][0];

            // Update the torque array
            torque[i][0] += torque_i[0];
            torque[i][1] += torque_i[1];
            torque[i][2] += torque_i[2];

            if (newton_pair || j < nlocal) {
              torque[j][0] -= torque_j[0];
              torque[j][1] -= torque_j[1];
              torque[j][2] -= torque_j[2];
            }
	    
	    if (eflag) evdwl = factor_lj * gaussValue; // - offset[type_i][type_j];
            if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
          }
        }
      }
    }
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
  
  // Step 2: result = tmp * conj(quat) 
  result[0] = quat[0]*tmp[0] + quat[1]*tmp[1] + quat[2]*tmp[2] + quat[3]*tmp[3];
  result[1] = quat[0]*tmp[1] - quat[1]*tmp[0] + quat[2]*tmp[3] - quat[3]*tmp[2];
  result[2] = quat[0]*tmp[2] - quat[1]*tmp[3] - quat[2]*tmp[0] + quat[3]*tmp[1];
  result[3] = quat[0]*tmp[3] + quat[1]*tmp[2] - quat[2]*tmp[1] - quat[3]*tmp[0];

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
  memory->create(offset, np1, np1, "pair:offset");
  memory->create(torque, np1, np1, "pair:torque");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLockKey::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Pair style lock/key must have exactly one argument");

  // reset per-type pair cutoffs that have been explicitly set previously

    if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++) {
      for (int j = i; j <= atom->ntypes; j++) {
        if (setflag[i][j]) {
          offset[i][j] = 0.0;  // Initialize to zero or some default value if necessary
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLockKey::coeff(int narg, char **arg)
{
  // Check that the correct number of arguments has been provided
  if (narg != 1) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  // Get the path to the CSV file from the arguments
  std::string csvPath = arg[0];

  // Load the data from the CSV file into your lookup tables
  loadLookupTables(csvPath);

  // Loop over all atom types and set the cutoffs based on the lookup table
  for (int i = 1; i <= atom->ntypes; i++) {
    std::string p1_id = std::to_string(i);
  
    // Get the sites and coordinates of the first particle
    auto& p1_sites = siteTable[p1_id];
  
    // Check if particle1_id exists in the siteTable
    if (siteTable.count(p1_id) == 0) continue;
  
    for (int j = 1; j <= atom->ntypes; j++) {
      std::string p2_id = std::to_string(j);
  
      // Get the sites and coordinates of the second particle
      auto& p2_sites = siteTable[p2_id];
  
      // Check if particle2_id exists in the siteTable
      if (siteTable.count(p2_id) == 0) continue;
  
      double max_cut_ij = 0.0; // Initialize to some minimum value
  
      // Iterate over all pairs of sites
      for (const auto& site1 : p1_sites) {
        for (const auto& site2 : p2_sites) {
          std::string interaction_id = (site1.siteID < site2.siteID)
                                        ? (site1.siteID + "-" + site2.siteID)
                                        : (site2.siteID + "-" + site1.siteID);
  
          if (interactingSitesTable.count(interaction_id) > 0) {
            InteractingSites interaction_params = interactingSitesTable[interaction_id];
            double cut_ij = interaction_params.lockKeyPrm[0]; // Declare cut_ij here
  
            if (cut_ij > max_cut_ij) {
              max_cut_ij = cut_ij;
            }
          }
        }
      }
      // Set the cutoffs once after looping through all site-site pairs
      cut[i][j] = max_cut_ij;
      cutsq[i][j] = max_cut_ij * max_cut_ij;
      setflag[i][j] = 1;
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
    error->all(FLERR, "Could not open CSV file");

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
//    else if (line == "# Interacting particles table") {
//      currentState = INTERACTING_PARTICLES;
//    }
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

              if (!(cells.size() ==  5)) {
                std::string errorMsg = "Error: on line " + std::to_string(line_counter) + " expected 5 columns, but got " + std::to_string(cells.size()) + ", on line: " + line + "\n\n";
                error->all(FLERR, errorMsg.c_str());
              }
              Site  siteEntry;
              siteEntry.protID = cells[0];
              siteEntry.siteID = cells[1];
              siteEntry.coords.resize(3);
              for (int i=0;i<3;++i){
                  try {
                    siteEntry.coords[i] = std::stod(cells[i+2]);
                  }
                  catch (const std::invalid_argument& ia) {
                    std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                      + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[i+2] + "' to double on line: " + line + "\n\n";
                    error->all(FLERR, errorMsg.c_str());
                  }
              }
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
/*          case INTERACTING_PARTICLES: {
              if (!(cells.size() ==  13)) {
                std::string errorMsg = "Error: on line " + std::to_string(line_counter) + " expected 13 columns, but got " + cells.size() + ", on line: " + line + "\n\n";
                error->all(FLERR, errorMsg.c_str());
              }
              InteractingParticles interactingParticlesEntry;
              interactingParticlesEntry.intPID = cells[0];
              interactingParticlesEntry.vdW.resize(4);
              interactingParticlesEntry.Pauli.resize(4);
              interactingParticlesEntry.electrostatic.resize(4);
              for (int i=0;i<4;++i){
                try{
                  interactingParticlesEntry.vdW[i] = std::stod(cells[i+1]);
                }
                catch (const std::invalid_argument& ia) {
                  std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                    + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[i+1] + "' to double on line: " + line + "\n\n";
                  error->all(FLERR, errorMsg.c_str());
                }
              }
              for (int i=0;i<4;++i){
                try{
                  interactingParticlesEntry.Pauli[i] = std::stod(cells[i+5]);
                }
                catch (const std::invalid_argument& ia) {
                  std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                    + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[i+5] + "' to double on line: " + line + "\n\n";
                  error->all(FLERR, errorMsg.c_str());
                }
              }
              for (int i=0;i<4;++i){
                try{
                  interactingParticlesEntry.electrostatic[i] = std::stod(cells[i+9]);
                }
                catch (const std::invalid_argument& ia) {
                  std::string errorMsg = std::string("Invalid argument: ") + ia.what() + '\n'
                    + "On line " + std::to_string(line_counter) + " could not convert value '" + cells[i+9] + "' to double on line: " + line + "\n\n";
                  error->all(FLERR, errorMsg.c_str());
                }
              }

              // Append the site to the vector of sites associated with the protein ID
              interactingParticlesTable[interactingParticlesEntry.intPID] = interactingParticlesEntry;

              break;
          }*/
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

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLockKey::init_one(int i, int j)
{
  // Check that all necessary parameters have been set
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  // Initialize variables
  double A, r0, sigma;

  // Retrieve the specific parameters for this pair of atom types
  std::string interaction_id = std::to_string(i) + "-" + std::to_string(j);
  if (interactingSitesTable.count(interaction_id) > 0) {
    InteractingSites interaction_params = interactingSitesTable[interaction_id];
    A = interaction_params.lockKeyPrm[1];
    r0 = interaction_params.lockKeyPrm[2];
    sigma = interaction_params.lockKeyPrm[3];
  } else {
    // Handle the case where the interaction parameters are not found
    error->all(FLERR, "Interaction parameters not found for the given pair of atom types");
  }

  // Calculate the offset
  double dr = cut[i][j] - r0;
  offset[i][j] = A * exp(-(dr * dr) / (2.0 * sigma * sigma));

  // Symmetrize
  offset[j][i] = offset[i][j];

  // Access the rotational Brownian fix (FixProRotation class) to get access to the quaternions array
  int ifix = modify->find_fix("name_of_your_fix_rot_brownian");
  if (ifix < 0) error->all(FLERR, "Fix rot_brownian not found");
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

                A = interaction_params.lockKeyPrm[1];
                r0 = interaction_params.lockKeyPrm[2];
                sigma = interaction_params.lockKeyPrm[3];

                double dr = r - r0;  // Declare and initialize dr

                offset[i][j] = -A * exp(-(cut[i][j] - r0) * (cut[i][j] - r0) / (2.0 * sigma * sigma));
                gaussForce = A * (dr / (sigma * sigma)) * exp(-(cut[i][j] - r0) * (cut[i][j] - r0) / (2.0 * sigma * sigma));

                totalForce += gaussForce;
                totalPotential -= offset[i][j];
            }
        }
    }

    fforce = factor_lj * totalForce / r;
    return factor_lj * totalPotential;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLockKey::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&offset[i][j], sizeof(double), 1, fp);
      }
    }
  }
}


/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLockKey::write_restart_settings(FILE *fp)
{
  fwrite(&offset_flag, sizeof(int), 1, fp);
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
	  utils::sfread(FLERR, &offset[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&offset[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}


/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLockKey::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
}
