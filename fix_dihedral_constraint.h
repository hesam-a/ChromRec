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
FixStyle(dihedral/constraint,FixDihedralConstraint);
// clang-format on
#else



#ifndef LMP_FIX_DIHEDRAL_CONSTRAINT_H
#define LMP_FIX_DIHEDRAL_CONSTRAINT_H

#include "fix.h"
#include "neighbor.h"
#include <vector>
#include <utility>

namespace LAMMPS_NS {


struct DihedralParams {
    double k;            // force constant
    double sign;         // sign
    int  multiplicity;   // multiplicity of the dihedral
    double cos_shift;    // cosine shift
    double sin_shift;    // sine shift 
};

struct DihedralConfig {
    int type1;
    int type2;
    int type3;
    int type4;
    int cut;
    DihedralParams params;
};


class FixDihedralConstraint : public Fix {
 public:
  FixDihedralConstraint(class LAMMPS *, int, char **);
  ~FixDihedralConstraint();
  int setmask() override ;
  void calculateAndApplyDihedralForce(int, int, int, int, DihedralParams);
  void initial_integrate(int) override;
  int findBondedAtomByType(int, int);
  bool isEligibleForDihedral(int, int); 
  void init() override;
  void init_list(int, class NeighList *) override;

 private:
  class NeighList *list;

 protected:
  double k, cos_shift, sin_shift;
  int sign, multiplicity;
  double cutoff;

  std::vector<DihedralConfig> dihedralConfigs;
  //std::vector<std::pair<int, int>> eligiblePairs;
  //std::vector<std::pair<std::pair<int, int>, DihedralParams>> dihedralInfo;
  //void addDihedral(int, int, int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
