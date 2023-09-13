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
 
#include "pair_gauss_sum.h"

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

PairGaussSum::PairGaussSum(LAMMPS *lmp) : Pair(lmp), cut(nullptr), offset(nullptr)
{
    single_enable = 1;
    respa_enable = 0; 
    writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairGaussSum::~PairGaussSum()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairGaussSum::compute(int eflag, int vflag) {
  int ii, i, j, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, factor_lj, charge_product, gauss1, gauss2, gauss_sum ;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

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
        charge_product = atom->q[i] * atom->q[j];  // product of charges
        gauss1 = charge_product * exp(-pow((r - D/2), 2) / (2 * pow(std_dev, 2)));
        gauss2 = charge_product * exp(-pow((r + D/2), 2) / (2 * pow(std_dev, 2)));
        gauss_sum = gauss1 + gauss2;

        // Apply the special scaling factor to the forces
        fpair = factor_lj * gauss_sum;

        f[i][0] += fpair * delx / r;
        f[i][1] += fpair * dely / r;
        f[i][2] += fpair * delz / r;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fpair * delx / r;
          f[j][1] -= fpair * dely / r;
          f[j][2] -= fpair * delz / r;
        }

	if (eflag) evdwl = factor_lj * gauss_sum - offset[itype][jtype];
	if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz),
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGaussSum::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(offset, np1, np1, "pair:offset");
}


/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGaussSum::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Pair style gauss/sum must have exactly one argument");
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

void PairGaussSum::coeff(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double D_one = utils::numeric(FLERR, arg[2], false, lmp);
  double cut_one = cut_global;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      D[i][j] = D_one;
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

double PairGaussSum::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  // Compute the offset
  if (offset_flag) {
    double r_norm = cut[i][j];
    double charge_product = atom->q[i] * atom->q[j];  // product of charges
    gauss1 =  charge_product * exp(-pow((r_norm - D/2), 2) / (2 * pow(std_dev, 2)));
    gauss2 =  charge_product * exp(-pow((r_norm + D/2), 2) / (2 * pow(std_dev, 2)));
    offset[i][j]  = gauss1 + gauss2;
  } else
    offset[i][j] = 0.0;

  // Symmetrize the potential parameter arrays
  cut[j][i] = cut[i][j];
  offset[j][i] = offset[i][j];


  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairGaussSum::init_style()
{

  // Request a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->id = 0;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ---------------------------------------------------------------------- */

double PairGaussSum::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                             double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r,charge_product,gauss1,gauss2;

  r = sqrt(rsq);
  //charge_product = atom->q[i] * atom->q[j];  // product of charges
  gauss1 =  exp(-pow((r_norm - D/2), 2) / (2 * pow(std_dev, 2)));
  gauss2 =  exp(-pow((r_norm + D/2), 2) / (2 * pow(std_dev, 2)));

  fforce = factor_lj * (gauss1 + gauss2);
  return factor_lj * (gauss1 + gauss2 - offset[itype][jtype]);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGaussSum::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGaussSum::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGaussSum::read_restart(FILE *fp)
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
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

//void PairGaussSum::read_restart_settings(FILE *fp)
//{
//  if (comm->me == 0) {
//    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
//    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
//    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
//  }
//  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
//  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
//  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
//}
//
///* ----------------------------------------------------------------------
//   proc 0 writes to data file
//------------------------------------------------------------------------- */
//
//void PairGaussSum::write_data(FILE *fp)
//{
//  for (int i = 1; i <= atom->ntypes; i++)
//    fprintf(fp, "%d %g %g %g\n", i, D[i][i], std_dev[i][i], cut[i][i]);
//}
//
///* ----------------------------------------------------------------------
//   proc 0 writes all pairs to data file
//------------------------------------------------------------------------- */
//
//void PairGaussSum::write_data_all(FILE *fp)
//{
//  for (int i = 1; i <= atom->ntypes; i++)
//    for (int j = i; j <= atom->ntypes; j++)
//      fprintf(fp, "%d %d %g %g %g\n", i, j, D[i][j], std_dev[i][j], cut[i][j]);
//}
//

/* ---------------------------------------------------------------------- */

void *PairGaussSum::extract(const char *str, int &dim)
{
  if (strcmp(str, "D") == 0) {
    dim = 0;
    return (void *) &D;
  }
  if (strcmp(str, "std_dev") == 0) {
    dim = 0;
    return (void *) &std_dev;
  }
  if (strcmp(str, "cut") == 0) {
    dim = 2;
    return (void *) cut;
  }
  return nullptr;
}

