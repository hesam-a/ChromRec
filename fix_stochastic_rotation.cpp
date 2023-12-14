/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------------
   Contributing author: Hesam Arabzadeh, University of Missouri, hacr6@missouri.edu
---------------------------------------------------------------------------------- */

#include "fix_stochastic_rotation.h"

#include "atom.h"
#include "error.h"
#include "modify.h"
#include "domain.h"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */
FixStochasticRotation::FixStochasticRotation(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)  {
  if (narg < 5 || narg > 6) utils::missing_cmd_args(FLERR, "fix stochastic/rotation", error);

  // Parse min and max rotation angles
  angle_std_dev = utils::numeric(FLERR, arg[3], false, lmp);
  coeff = utils::numeric(FLERR, arg[4], false, lmp);

  if (angle_std_dev < 0)
    error->all(FLERR, "Illegal fix stochastic/rotation angle values: angle std dev {}", angle_std_dev);

  std::random_device rd;
  gen = std::mt19937(rd());

}


/* ---------------------------------------------------------------------- */
int FixStochasticRotation::setmask() {
  int mask = 0;
  mask |= FixConst::INITIAL_INTEGRATE;
  return mask;
}


/* ---------------------------------------------------------------------- */
// Function to calculate the dominant interaction vector
std::pair<double, Eigen::Vector3d> FixStochasticRotation::calculateDominantInteractionVector(int particle_index, double **f, int num_bonded, int *bonded_atoms) {

  int j, site_idx;
  double max_force_mag = 0.0;
  double force_mag = 0.0;
  Eigen::Vector3d dominant_force_dir;

  for (j = 0; j < num_bonded; ++j) {
    site_idx = bonded_atoms[j];
//    std::cout << "    site_idx: " << site_idx << '\n'; 
    Eigen::Vector3d force(f[site_idx - 1][0], f[site_idx - 1][1], f[site_idx - 1][2]);
    force_mag = force.norm();

    if (force_mag > max_force_mag) {
      max_force_mag = force_mag;
      dominant_force_dir = force.normalized();
//      std::cout << "force direction: " << dominant_force_dir.transpose() << '\n'; 
    }
  }

  return std::make_pair(max_force_mag, dominant_force_dir);
}


/* ---------------------------------------------------------------------- */
// Function to scale rotation based on force magnitude
double FixStochasticRotation::scale_rotation(double force_mag, double coeff) {
  return 1.0 / (1.0 + (coeff * force_mag));
}


/* ---------------------------------------------------------------------- */
void FixStochasticRotation::initial_integrate(int vflag) {
  Atom *atom = lmp->atom;
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i, j, site_idx;
  int num_bonded, bonded_atom_idx;
  int *bonded_atoms;
  double angle, force_mag, scale_factor;

  std::pair<double, Eigen::Vector3d> force_info; 
  Eigen::Vector3d site_pos, rotated_pos, dominant_force_dir, rotation_axis;
  Eigen::Vector3d particle_pos;
  Eigen::Matrix3d rotation_matrix;

  std::normal_distribution<> gauss_dist(0.0, angle_std_dev);

  for (i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit)) continue;

//    std::cout << "     ***** particle " << i+1 <<  "******\n";

    num_bonded = atom->num_bond[i];
    bonded_atoms = atom->bond_atom[i];

    force_info = calculateDominantInteractionVector(i, f, num_bonded, bonded_atoms);
    force_mag = force_info.first;
    dominant_force_dir = force_info.second;

//    std::cout << "force magnitude: " << force_mag << '\n'; 

    if (force_mag > 0) {
      // Normalize the force direction to use as rotation axis
      rotation_axis = dominant_force_dir;
//      std::cout << "force direction: " << rotation_axis.transpose() << '\n'; 

      scale_factor = scale_rotation(force_mag, coeff);
      angle = gauss_dist(gen); 
//      std::cout << "angle before: " << angle << '\n'; 
      angle *= scale_factor;
//      std::cout << "angle after: " << angle << '\n';

      rotation_matrix = Eigen::AngleAxisd(angle, rotation_axis).toRotationMatrix();
//      std::cout << "rotation matrix:\n" << rotation_matrix.transpose() << '\n'; 

      // Apply rotation to each bonded site
      for (j = 0; j < num_bonded; ++j) {
        site_idx = bonded_atoms[j];
        site_pos = Eigen::Vector3d(x[site_idx - 1][0], x[site_idx - 1][1], x[site_idx - 1][2]);
        particle_pos = Eigen::Vector3d(x[i][0], x[i][1], x[i][2]);
        
        rotated_pos = rotation_matrix * (site_pos - particle_pos) + particle_pos;	
        
        x[site_idx - 1][0] = rotated_pos[0];
        x[site_idx - 1][1] = rotated_pos[1];
        x[site_idx - 1][2] = rotated_pos[2];
      }
    }
  }
} 
