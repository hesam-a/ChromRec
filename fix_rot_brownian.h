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
FixStyle(rot/brownian,FixRotBrownian);
// clang-format on
#else

#ifndef LMP_FIX_ROT_BROWNIAN_H
#define LMP_FIX_ROT_BROWNIAN_H

#include "fix_brownian_base.h"

namespace LAMMPS_NS {

class FixRotBrownian : public FixBrownianBase {
 public:
  FixRotBrownian(class LAMMPS *, int, char **);
  ~FixRotBrownian();

  double* getQuaternion(int i);
  void init() override;
  void initial_integrate(int) override;

 private:
  template <int Tp_UNIFORM, int Tp_GAUSS>
  void initial_integrate_templated();

  // Additional member variables specific to FixRotBrownian
  double g3;   // Scaling factor for rotational noise
  double **q;  // Quaternion array
};

}    // namespace LAMMPS_NS
#endif
#endif
