/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(brownian/force/rotation,FixBrownianForceRotation);
// clang-format on
#else

#ifndef LMP_FIX_BROWNIAN_FORCE_ROTATION_H
#define LMP_FIX_BROWNIAN_FORCE_ROTATION_H

#include "fix_brownian_base.h"
#include <random>

namespace LAMMPS_NS {


class FixBrownianForceRotation : public FixBrownianBase {
 public:
  FixBrownianForceRotation(class LAMMPS *, int, char **);

//  int setmask() override;  
  void init() override;
  void initial_integrate(int) override;
  double scale_rotation(double, double);  

 private:
  template <int Tp_UNIFORM, int Tp_GAUSS> void initial_integrate_templated();

 protected:
 double angle_std_dev, scaling_factor;
 std::mt19937 gen;  // Random number generator

};
}    // namespace LAMMPS_NS

#endif
#endif
