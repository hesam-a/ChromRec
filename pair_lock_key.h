/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lock/key,PairLockKey)
// clang-format on

#else

#ifndef LMP_PAIR_LOCK_KEY_H
#define LMP_PAIR_LOCK_KEY_H

#include <map>
#include <vector>
#include <unordered_map>
#include "pair.h"

#include "fix_rot_brownian.h"

namespace LAMMPS_NS {


enum TableState {
    NONE,
    PROTEIN,
    SITE,
    INTERACTING_SITES,
    SITE_COORDS
//    INTERACTING_PARTICLES
};

struct Protein {
    std::string lammpsType;
    std::string fragName;
    std::string uniProtID;
    double radius;
    double mass;
};

struct Site {
    std::string protID;
    std::string siteID;
    std::vector<double> coords;
};

struct InteractingSites {
    std::string  intID;
    std::vector<double> lockKeyPrm;
};

// Define a struct to hold the coordinates
struct Coords {
    double x;
    double y;
    double z;
};

class PairLockKey : public Pair {
  public:
  PairLockKey(class LAMMPS *);
  ~PairLockKey();

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
//  virtual void init_style() override;
  void loadLookupTables(const std::string &);

  void write_restart(FILE *) override;
  void write_restart_settings(FILE *);
  void read_restart(FILE *) override;
  void rotateVectorByQuaternion(const double *vec, const double *quat, double *result);
  void readSiteCoords();
  void updateSiteCoordsFile();

  double single(int, int, int, int, double, double, double, double &) override;

//  double** getTorque() {
//    return torque;
//  }

  private:

  std::map<std::string, Protein> proteinTable;
  std::map<std::string, std::vector<Site>> siteTable;
  std::map<std::string, InteractingSites> interactingSitesTable;
  std::unordered_map<std::string, Coords> siteCoordsMap; 

  std::ofstream siteCoordsFile;  // File stream for writing site coordinates
  bool isSiteCoordsFileOpen = false;  // Flag to check if the file is open
  bool writeSiteCoords;

//  std::map<std::string, InteractingParticles> interactingParticlesTable;

  FixRotBrownian *fixRotBrownian;  // Pointer to FixRotBrownian object


 protected:

  double cut_global;
  double **cut; // **torque;

  virtual void allocate();
};

}  // namespace LAMMPS_NS

#endif
#endif
