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
#include "universe.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "math_const.h"
#include "math_extra.h"

#include <random>
#include <iostream>
#include <Eigen/Dense>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;


/* ---------------------------------------------------------------------- */
FixStochasticRotation::FixStochasticRotation(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)  {
  if (narg < 5 || narg > 6) utils::missing_cmd_args(FLERR, "fix stochastic/rotation", error);

  // Parse min and max rotation angles
  min_angle = utils::numeric(FLERR, arg[3], false, lmp);
  max_angle = utils::numeric(FLERR, arg[4], false, lmp);

  if (min_angle < 0 || max_angle <= min_angle)
    error->all(FLERR, "Illegal fix stochastic/rotation angle values: min {} max {}", min_angle, max_angle);


  std::random_device rd;
  gen = std::mt19937(rd());
  //std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

}


/* ---------------------------------------------------------------------- */
int FixStochasticRotation::setmask() {
  int mask = 0;
  mask |= FixConst::INITIAL_INTEGRATE;
  return mask;
}

void FixStochasticRotation::BuildRxMatrixEigen(Eigen::Matrix3d &R, double angle) {
    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
}

void FixStochasticRotation::BuildRyMatrixEigen(Eigen::Matrix3d &R, double angle) {
    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
}

void FixStochasticRotation::BuildRzMatrixEigen(Eigen::Matrix3d &R, double angle) {
    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

/* ---------------------------------------------------------------------- */
void FixStochasticRotation::initial_integrate(int vflag) {
  Atom *atom = lmp->atom;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i, j, k;
  int axis, num_bonded, bonded_atom_idx;
  int *bonded_atoms;
  double angle;
  Eigen::Matrix3d R;
  Eigen::Vector3d displacement;
  Eigen::Vector3d rotated_displacement;

  std::uniform_real_distribution<> dis(min_angle, max_angle);


  for (i = 0; i < nlocal; ++i) {
    if (mask[i] & groupbit){
      //std::cout << "   after mask  Particle " << i+1 << '\n';

      // Get the list of bonded atoms for particle i
      num_bonded = atom->num_bond[i];
      bonded_atoms = atom->bond_atom[i];

      // Generate random angle
      angle = dis(gen);

      // Randomly choose an axis (0 = X, 1 = Y, 2 = Z)
      axis = rand() % 3;

      // Build rotation matrix for the chosen axis
      if (axis == 0) BuildRxMatrixEigen(R, angle);
      else if (axis == 1) BuildRyMatrixEigen(R, angle);
      else BuildRzMatrixEigen(R, angle);

      // Inside the loop over bonded atoms
      for (j = 0; j < num_bonded; ++j) {
        bonded_atom_idx = bonded_atoms[j]-1;

        //std::cout << "         bonded_atom_idx: " << bonded_atom_idx << '\n';

        // Calculate the vector from the central particle to the bonded site
        for (k = 0; k < 3; ++k)
          displacement[k] = x[bonded_atom_idx][k] - x[i][k];

        //std::cout << "   displacement:         " << displacement.transpose() << std::endl;

        // Apply the rotation matrix to this displacement vector
        rotated_displacement = R * displacement;

        //std::cout << "   rotated_displacement: " << rotated_displacement.transpose()<< std::endl;

        //std::cout << " site position before rotation:   " << x[bonded_atom_idx][0] << "    " << x[bonded_atom_idx][1] << "    " << x[bonded_atom_idx][2] << '\n';

        // Update the position of the bonded site
        for (k = 0; k < 3; ++k)
          x[bonded_atom_idx][k] = x[i][k] + rotated_displacement[k];

        //std::cout << " site position after rotation:   " << x[bonded_atom_idx][0] << "    " << x[bonded_atom_idx][1] << "    " << x[bonded_atom_idx][2] << '\n';
      }
    }
  }
}
