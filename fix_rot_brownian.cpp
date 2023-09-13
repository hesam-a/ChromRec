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

/* --------------------------------------------------------------------------------

   Originally modified from CG-DNA/fix_nve_dotc_langevin.cpp.

   Contributing author: Hesam Arabzadeh, University of Missouri, hacr6@missouri.edu

-------------------------------------------------------------------------------- */

#include "fix_rot_brownian.h"
#include "pair_vdw_attract.h"
#include "pair_pauli_repuls.h"
#include "pair_lock_key.h"


#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "math_extra.h"
#include "random_mars.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRotBrownian::FixRotBrownian (LAMMPS *lmp, int narg, char **arg) :
  FixBrownianBase(lmp, narg, arg)
{
  if (dipole_flag || gamma_t_eigen_flag || gamma_r_eigen_flag || rot_temp_flag || planar_rot_flag) {
    error->all(FLERR, "Illegal fix rot/brownian command.");
  }

  if (!gamma_t_flag || !gamma_r_flag) { error->all(FLERR, "Illegal fix rot/brownian command."); }
}

/* ---------------------------------------------------------------------- */

void FixRotBrownian::init()
{
  memory->create(q, atom->nlocal, 4, "fixRotBrownian:q");
  FixBrownianBase::init();  // Initialize variables from the base class

  g3 = g1 / gamma_r;  // Scaling factor for rotational noise
  g1 /= gamma_t;      // Update scaling factor for translational noise
  g2 *= sqrt(temp / gamma_t);  // Update scaling factor for translational noise
}

/* ---------------------------------------------------------------------- */

FixRotBrownian::~FixRotBrownian()
{
  memory->destroy(q);	
}

/* ---------------------------------------------------------------------- */

void FixRotBrownian::initial_integrate(int /*vflag */)
{
  if (!noise_flag) {
    initial_integrate_templated<0, 0>();
  } else if (gaussian_noise_flag) {
    initial_integrate_templated<0, 1>();
  } else {
    initial_integrate_templated<1, 0>();
  }
}


template <int Tp_UNIFORM, int Tp_GAUSS>
void FixRotBrownian::initial_integrate_templated()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double dx, dy, dz;
  double wx, wy, wz;
  double dt_half = 0.5 * dt;
  double q_old[4];
  double w[4];
  double q_norm;


  // Initialize total_torque to zero
  std::vector<std::array<double, 3>> total_torque(nlocal, {0.0, 0.0, 0.0});

  PairVDWAttract  *pair1 = (PairVDWAttract  *) force->pair_match("vdw/attract", 0);
  PairPauliRepuls *pair2 = (PairPauliRepuls *) force->pair_match("pauli/repuls", 0);
  PairLockKey     *pair3 = (PairLockKey     *) force->pair_match("lock/key    ", 0);

  if (!pair1 || !pair2 || !pair3) {
    error->all(FLERR, "One of the pair styles is not found.");
  }

  double  **torque1 = pair1->getTorque();
  double  **torque2 = pair2->getTorque();
  double  **torque3 = pair3->getTorque();

  // Sum up the torques
  for (int i = 0; i < nlocal; ++i) {

    total_torque[i][0] = torque1[i][0] + torque2[i][0] + torque3[i][0]; 
    total_torque[i][1] = torque1[i][1] + torque2[i][1] + torque3[i][1]; 
    total_torque[i][2] = torque1[i][2] + torque2[i][2] + torque3[i][2]; 
  }

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;


  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (Tp_UNIFORM) {
        dx = dt * (g1 * f[i][0] + g2 * (rng->uniform() - 0.5));
        dy = dt * (g1 * f[i][1] + g2 * (rng->uniform() - 0.5));
        dz = dt * (g1 * f[i][2] + g2 * (rng->uniform() - 0.5));
        wx = (rng->uniform() - 0.5);
        wy = (rng->uniform() - 0.5);
        wz = (rng->uniform() - 0.5);
      } 
      else if (Tp_GAUSS) {
        dx = dt * (g1 * f[i][0] + g2 * rng->gaussian());
        dy = dt * (g1 * f[i][1] + g2 * rng->gaussian());
        dz = dt * (g1 * f[i][2] + g2 * rng->gaussian());
        wx = rng->gaussian();
        wy = rng->gaussian();
        wz = rng->gaussian();
      }
      else {
	dx = dt * g1 * f[i][0];
        dy = dt * g1 * f[i][1];
        dz = dt * g1 * f[i][2];
        wx = wy = wz = 0;
      }

      x[i][0] += dx;
      v[i][0] = dx / dt;

      x[i][1] += dy;
      v[i][1] = dy / dt;

      x[i][2] += dz;
      v[i][2] = dz / dt;

      wx += g3 * total_torque[i][0];
      wy += g3 * total_torque[i][1];
      wz += g3 * total_torque[i][2];

      // Update quaternions based on angular velocities
      w[0] = 0.0; w[1] = wx * dt_half; w[2] = wy * dt_half; w[3] = wz * dt_half;
      q_old[0] = q[i][0]; q_old[1] = q[i][1]; q_old[2] = q[i][2]; q_old[3] = q[i][3];
      
      // Quaternion multiplication: q_new = w * q_old
      q[i][0] =  q_old[0]*w[0] - q_old[1]*w[1] - q_old[2]*w[2] - q_old[3]*w[3];
      q[i][1] =  q_old[0]*w[1] + q_old[1]*w[0] - q_old[2]*w[3] + q_old[3]*w[2];
      q[i][2] =  q_old[0]*w[2] + q_old[1]*w[3] + q_old[2]*w[0] - q_old[3]*w[1];
      q[i][3] =  q_old[0]*w[3] - q_old[1]*w[2] + q_old[2]*w[1] + q_old[3]*w[0];

      // Normalize the quaternion
      q_norm = sqrt(q[i][0]*q[i][0] + q[i][1]*q[i][1] + q[i][2]*q[i][2] + q[i][3]*q[i][3]);
      if (q_norm != 0) {
        q[i][0] /= q_norm;
        q[i][1] /= q_norm;
        q[i][2] /= q_norm;
        q[i][3] /= q_norm;
      }
      else {
	error->warning(FLERR, "Quaternion normalization failed. Setting to identity quaternion.");
        q[i][0] = 1.0;
        q[i][1] = 0.0;
        q[i][2] = 0.0;
        q[i][3] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

double* FixRotBrownian::getQuaternion(int i) {
  if (i >= 0 && i < atom->nlocal) {  // Check if the index is within bounds
    return q[i];
  } else {
    error->all(FLERR, "Index out of bounds in getQuaternion");
    return nullptr;  
  }
}
