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

/* ----------------------------------------------------------------------
   Originally modified from BROWNIAN/fix_brownian.cpp.

   Contributing author: Hesam Arabzadeh (University of Missouri)
------------------------------------------------------------------------- */

#include "fix_brownian_force_rotation.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "random_mars.h"

#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include "/usr/include/eigen3/Eigen/Dense"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBrownianForceRotation::FixBrownianForceRotation(LAMMPS *lmp, int narg, char **arg) : FixBrownianBase(lmp, narg, arg)
{
  if (dipole_flag || gamma_t_eigen_flag || gamma_r_eigen_flag || gamma_r_flag || rot_temp_flag ||
      planar_rot_flag) {
    error->all(FLERR, "Illegal fix brownian/force/rotation command.");
  }
  if (!gamma_t_flag) { error->all(FLERR, "Illegal fix brownian/force/rotation command."); }
}

/* ---------------------------------------------------------------------- */
void FixBrownianForceRotation::init()
{
  Eigen::setNbThreads(1);
  FixBrownianBase::init();
  g1 /= gamma_t;
  g2 *= sqrt(temp / gamma_t);
}

/* ---------------------------------------------------------------------- */
// Function to scale rotation based on force magnitude
double FixBrownianForceRotation::scale_rotation(double force_mag, double scaling_factor) {
  return 1.0 / (1.0 + (scaling_factor * force_mag));
}


/* ---------------------------------------------------------------------- */
void FixBrownianForceRotation::initial_integrate(int /*vflag */)
{
  if (!noise_flag) {
    initial_integrate_templated<0, 0>();
  } else if (gaussian_noise_flag) {
    initial_integrate_templated<0, 1>();
  } else {
    initial_integrate_templated<1, 0>();
  }
}

/* ---------------------------------------------------------------------- */
template <int Tp_UNIFORM, int Tp_GAUSS> void FixBrownianForceRotation::initial_integrate_templated()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

//  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double dx, dy, dz;
  int i, j, k, num_bonded;
  int local_site_idx_F,local_site_idx_R;
  int *bonded_atoms;
  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;  
  double angle, scale_rot, force_magnitude;  

  Eigen::Vector3d rotation_axis, particle_pos, site_pos, rotated_pos, net_force;
  Eigen::Matrix3d rotation_matrix;
  std::normal_distribution<> angle_dist(0.0, angle_std_dev);
  std::normal_distribution<> axis_dist(0.0, 1.0);  

  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

      num_bonded = atom->num_bond[i];
      bonded_atoms = atom->bond_atom[i];

//      std::cout << "\nParticle Global ID " << atom->tag[i] << "  " << x[i][0] << "  " << x[i][1] << "  " << x[i][2] << "\n";
//      std::cout << "     Befor force addition " << atom->tag[i] << "  " << f[i][0] << "  " << f[i][1] << "  " << f[i][2] << '\n';

      for (j = 0; j < num_bonded; ++j) {
        local_site_idx_F = atom->map(bonded_atoms[j]);

        if (local_site_idx_F == -1) continue; // Site not found or is a ghost atom

        f[i][0] += f[local_site_idx_F][0];
        f[i][1] += f[local_site_idx_F][1];
        f[i][2] += f[local_site_idx_F][2];
      }

//      std::cout << "     After force addition " << atom->tag[i] << "  " << f[i][0] << "  " << f[i][1] << "  " << f[i][2] << '\n';
      if (Tp_UNIFORM) { 
        dx = dt * (g1 * f[i][0] + g2 * (rng->uniform() - 0.5));
        dy = dt * (g1 * f[i][1] + g2 * (rng->uniform() - 0.5));
        dz = dt * (g1 * f[i][2] + g2 * (rng->uniform() - 0.5));
      } else if (Tp_GAUSS) {
        dx = dt * (g1 * f[i][0] + g2 * rng->gaussian());
        dy = dt * (g1 * f[i][1] + g2 * rng->gaussian());
        dz = dt * (g1 * f[i][2] + g2 * rng->gaussian());
      } else {
        dx = dt * g1 * f[i][0];
        dy = dt * g1 * f[i][1];
        dz = dt * g1 * f[i][2];
      }
    
//    Eigen::Vector3d particle_pos_before(x[i][0], x[i][1], x[i][2]);

    x[i][0] += dx;
    v[i][0] = dx / dt;

    x[i][1] += dy;
    v[i][1] = dy / dt;

    x[i][2] += dz;
    v[i][2] = dz / dt;

    // Calculate the magnitude of the force on the parent particle
    net_force = Eigen::Vector3d(f[i][0], f[i][1], f[i][2]);
    force_magnitude = net_force.norm();

    // Calculate scale factor for rotation based on force magnitude
    scale_rot = scale_rotation(force_magnitude, scaling_factor);    

    // Set up rotation
    rotation_axis = Eigen::Vector3d(axis_dist(gen), axis_dist(gen), axis_dist(gen));
    rotation_axis.normalize();
    angle = angle_dist(gen) * scale_rot;
    rotation_matrix = Eigen::AngleAxisd(angle, rotation_axis).toRotationMatrix();
  
    // Apply updates to bonded sites
    particle_pos = Eigen::Vector3d(x[i][0], x[i][1], x[i][2]);

    for (k = 0; k < num_bonded; ++k) {
      local_site_idx_R = atom->map(bonded_atoms[k]);
      if (local_site_idx_R == -1) continue;

 //     Eigen::Vector3d original_site_pos(x[local_site_idx_R][0], x[local_site_idx_R][1], x[local_site_idx_R][2]);
 //     double distance_before = (original_site_pos - particle_pos).norm();

//      std::cout << "\nBefore site ID " << atom->map(bonded_atoms[j]) << "  " << x[local_site_idx_R][0] << "  " << x[local_site_idx_R][1] << "  " << x[local_site_idx_R][2] << "\n";

      // Update site position to follow parent particle
      x[local_site_idx_R][0] += dx;  
      x[local_site_idx_R][1] += dy;
      x[local_site_idx_R][2] += dz;


 //     double site_displace = Eigen::Vector3d(site_pos - original_site_pos).norm(); 
 //     double part_displace = Eigen::Vector3d(particle_pos - particle_pos_before).norm(); 

      // Apply stochastic rotation
      site_pos = Eigen::Vector3d(x[local_site_idx_R][0], x[local_site_idx_R][1], x[local_site_idx_R][2]);
      rotated_pos = rotation_matrix * (site_pos - particle_pos) + particle_pos;
      x[local_site_idx_R][0] = rotated_pos[0];
      x[local_site_idx_R][1] = rotated_pos[1];
      x[local_site_idx_R][2] = rotated_pos[2];
      
//      Eigen::Vector3d site_pos2(x[local_site_idx_R][0], x[local_site_idx_R][1], x[local_site_idx_R][2]);

//      double distance_after = (rotated_pos - particle_pos).norm();
//      double site_displace_2 = Eigen::Vector3d(site_pos2 - site_pos).norm(); 

      // Print or log the distances for verification
//      std::cout << "Particle: " << atom->tag[i] << ", Site: " << atom->tag[local_site_idx_R]
//                << ", Distance before: " << distance_before
//                << ", Distance after: " << distance_after
//	       	<< ", particle displacement: " << part_displace 
//		<< ", site displacement: " << site_displace
//		<< ", site displacement after rotation: " << site_displace_2 << std::endl;

//      std::cout << "\nAfter Particle ID " << atom->tag[j] << "  " << x[local_site_idx_R][0] << "  " << x[local_site_idx_R][1] << "  " << x[local_site_idx_R][2] << "\n";
//      std::cout << "     Position of particle " << atom->tag[i] << ":   " << x[i][0] << "   " << x[i][1] << "   " << x[i][2] << '\n'

    }
  }
}
