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
FixStyle(stochastic/rotation,FixStochasticRotation);
// clang-format on
#endif

#ifndef LMP_FIX_STOCHASTIC_ROTATION_H
#define LMP_FIX_STOCHASTIC_ROTATION_H

#include <random>

#include "fix.h"
#include "/usr/include/eigen3/Eigen/Dense"

namespace LAMMPS_NS {

class FixStochasticRotation : public Fix {
 public:
  FixStochasticRotation(class LAMMPS *, int, char **);
  virtual ~FixStochasticRotation() {}
  int setmask() override;
//  void BuildRxMatrixEigen(Eigen::Matrix3d &, double);
//  void BuildRyMatrixEigen(Eigen::Matrix3d &, double);
//  void BuildRzMatrixEigen(Eigen::Matrix3d &, double);
  std::pair<double,Eigen::Vector3d> calculateDominantInteractionVector(int , double **, int , int *); 
  double scale_rotation(double, double );

  void initial_integrate(int) override;

 protected:
  double angle_std_dev, coeff;
  std::mt19937 gen;  // Random number generator

};

}

#endif // LMP_FIX_STOCHASTIC_ROTATION_H
