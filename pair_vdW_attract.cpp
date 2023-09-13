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
 
#include "pair_vdw_attract.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairVDWAttract::PairVDWAttract(LAMMPS *lmp) : Pair(lmp), cut(nullptr), offset(nullptr)
{
    single_enable = 1;
    respa_enable = 0; 
    writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairVDWAttract::~PairVDWAttract()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(offset);
    memory->destroy(R);
    memory->destroy(amp);
    memory->destroy(std_dev);
    memory->destroy(torque);
  }
}

/* ---------------------------------------------------------------------- */

void PairVDWAttract::compute(int eflag, int vflag) {
  int ii, i, j, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, delta_r,std_dev2, vdW, factor_lj;
  double torque_i[3], torque_j[3];
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

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

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      factor_lj = special_lj[sbmask(j)];


      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      r = sqrt(rsq);

      if (rsq < cutsq[itype][jtype]) {
        delta_r  = r - R[itype][jtype];
	std_dev2 = std_dev[itype][jtype] * std_dev[itype][jtype];

        vdW = -amp[itype][jtype] * exp(-(delta_r * delta_r)/ 2 * std_dev2 );
	fpair = (-delta_r/std_dev2) * vdW;

        // Apply the special scaling factor to the forces
        fpair *= factor_lj;

        f[i][0] += fpair * delx / r;
        f[i][1] += fpair * dely / r;
        f[i][2] += fpair * delz / r;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fpair * delx / r;
          f[j][1] -= fpair * dely / r;
          f[j][2] -= fpair * delz / r;
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

	if (eflag) evdwl = factor_lj * vdW - offset[itype][jtype];
	if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairVDWAttract::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(offset, np1, np1, "pair:offset");
  memory->create(R, np1, np1, "pair:R");
  memory->create(amp, np1, np1, "pair:amp");
  memory->create(std_dev, np1, np1, "pair:std_dev");
  memory->create(torque, np1, np1, "pair:torque");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairVDWAttract::settings(int narg, char **arg)
{
  if (narg != 9) error->all(FLERR, "Pair style vdw/attract requires exactly one argument: global cutoff");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset per-type pair cutoffs that have been explicitly set previously

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairVDWAttract::coeff(int narg, char **arg)
{
  if (narg != 9) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double R_i = utils::numeric(FLERR, arg[2], false, lmp);
  double amp_i = utils::numeric(FLERR, arg[3], false, lmp);
  double std_dev_i = utils::numeric(FLERR, arg[4], false, lmp);
  double R_j = utils::numeric(FLERR, arg[5], false, lmp);
  double amp_j = utils::numeric(FLERR, arg[6], false, lmp);
  double std_dev_j = utils::numeric(FLERR, arg[7], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[8], false, lmp);

 // Error checks for invalid parameter values
  if (std_dev_i <= 0 || std_dev_j <= 0) error->all(FLERR, "Standard deviation for vdw/attract must be positive");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      R[i][j] = 0.5 * (R_i + R_j);  // Using arithmetic mean for mixing
      amp[i][j] = sqrt(amp_i * amp_j);    // Using geometric mean for mixing
      std_dev[i][j] = 0.5 * (std_dev_i + std_dev_j);  // Using arithmetic mean for mixing
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairVDWAttract::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  // Compute the offset
  offset[i][j] = -amp[i][j] * exp(-std_dev[i][j] * cut[i][j]);

  // Symmetrize the potential parameter arrays
  cut[j][i] = cut[i][j];
  amp[j][i]   = amp[i][j];
  std_dev[j][i]   = std_dev[i][j];
  offset[j][i] = offset[i][j];


  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairVDWAttract::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                             double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r, delta_r, vdW;

  r = sqrt(rsq);
  delta_r = r - R[itype][jtype]; 

  vdW = -amp[itype][jtype] * exp(-(delta_r * delta_r)/ 2 * std_dev[itype][jtype] * std_dev[itype][jtype] );

  fforce = factor_lj * (-delta_r/(std_dev[itype][jtype] * std_dev[itype][jtype])) * vdW/r;
  return factor_lj * (vdW - offset[itype][jtype]);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairVDWAttract::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&std_dev[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&amp[i][j], sizeof(double), 1, fp);
        fwrite(&R[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairVDWAttract::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairVDWAttract::read_restart(FILE *fp)
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
          utils::sfread(FLERR, &std_dev[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &amp[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &R[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&std_dev[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&amp[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&R[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
----------------------------------------------------------------------- */

void PairVDWAttract::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairVDWAttract::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g\n", i, j, std_dev[i][j], cut[i][i], amp[i][i], R[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairVDWAttract::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g\n", i, j, std_dev[i][j], cut[i][j], amp[i][i], R[i][i]);
}


/* ---------------------------------------------------------------------- */

void *PairVDWAttract::extract(const char *str, int &dim)
{
  dim =2;
  if (strcmp(str, "std_dev") == 0) return (void *) std_dev;
  if (strcmp(str, "amp") == 0)  return (void *) amp;
  if (strcmp(str, "R" )  == 0)  return (void *) R;
  return nullptr;
}

