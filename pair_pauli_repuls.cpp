#include "pair_pauli_repuls.h"

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

PairPauliRepuls::PairPauliRepuls(LAMMPS *lmp) : Pair(lmp), cut(nullptr), offset(nullptr)
{
    single_enable = 1;
    respa_enable = 0; 
    writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairPauliRepuls::~PairPauliRepuls()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(offset);
    memory->destroy(overlap);
    memory->destroy(R);
    memory->destroy(amp);
    memory->destroy(decay_const);
  }
}

/* ---------------------------------------------------------------------- */

void PairPauliRepuls::compute(int eflag, int vflag) {
  int ii, i, j, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r, delta_r, over, pauli, factor_lj ;
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
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
//    xtmp = x[i][0];
//    ytmp = x[i][1];
//    ztmp = x[i][2];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj] & NEIGHMASK;
      factor_lj = special_lj[sbmask(j)];

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
	over = overlap[itype][jtype] * R[itype][jtype];
        delta_r = r - over;

        pauli = amp[itype][jtype] * exp(-decay_const[itype][jtype] * delta_r);
        fpair = decay_const[itype][jtype] * pauli;

        // Apply the special scaling factor to the forces
        fpair *= factor_lj / r;

        f[i][0] += fpair * delx;
        f[i][1] += fpair * dely;
        f[i][2] += fpair * delz;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fpair * delx;
          f[j][1] -= fpair * dely;
          f[j][2] -= fpair * delz;
        }
	
	if (eflag) evdwl = factor_lj * pauli - offset[itype][jtype];
	if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPauliRepuls::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;

  memory->create(setflag, np1, np1, "pair:setflag");
  for (int i = 1; i < np1; i++)
    for (int j = i; j < np1; j++) setflag[i][j] = 0;

  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(cut, np1, np1, "pair:cut");
  memory->create(offset, np1, np1, "pair:offset");
  memory->create(overlap, np1, np1, "pair:overlap");
  memory->create(R, np1, np1, "pair:R");
  memory->create(amp, np1, np1, "pair:amp");
  memory->create(decay_const, np1, np1, "pair:decay_const");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPauliRepuls::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Pair style pauli/repuls requires exactly one argument: global cutoff");
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

void PairPauliRepuls::coeff(int narg, char **arg)
{
  if (narg != 10) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  // Check if the 'none' keyword is used
  if (strcmp(arg[2], "none") == 0) {
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo, i); j <= jhi; j++) {
        setflag[i][j] = 0; // Mark this pair as not set
      }
    }
    return; // Exit the function
  }  

  double overlap = utils::numeric(FLERR, arg[2], false, lmp);
  double R_i     = utils::numeric(FLERR, arg[3], false, lmp);
  double amp_i   = utils::numeric(FLERR, arg[4], false, lmp);
  double decay_const_i = utils::numeric(FLERR, arg[5], false, lmp);
  double R_j     = utils::numeric(FLERR, arg[6], false, lmp);
  double amp_j   = utils::numeric(FLERR, arg[7], false, lmp);
  double decay_const_j = utils::numeric(FLERR, arg[8], false, lmp);
  double cut_one = utils::numeric(FLERR, arg[9], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      R[i][j] = (R_i + R_j);        // Using arithmetic mean for mixing
      amp[i][j] = sqrt(amp_i * amp_j);    // Using geometric mean for mixing
      decay_const[i][j] = 0.5 * (decay_const_i + decay_const_j);  // Using arithmetic mean for mixing
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

double PairPauliRepuls::init_one(int i, int j) {
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  if (offset_flag) {
    offset[i][j] = amp[i][j] * exp(-decay_const[i][j] * (cut[i][j] - (overlap[i][j] * R[i][j])));
  } else {
    offset[i][j] = 0.0;
  }

  // Symmetrize the potential parameter arrays
  cut[j][i] = cut[i][j];
  amp[j][i] = amp[i][j];
  decay_const[j][i] = decay_const[i][j];
  offset[j][i] = offset[i][j];
  overlap[j][i] = overlap[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairPauliRepuls::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                             double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r, delta_r, pauli, fpair, over;

  r = sqrt(rsq);
  over = overlap[itype][jtype] * R[itype][jtype];
  delta_r = r - over;

  pauli = amp[itype][jtype] * exp(-decay_const[itype][jtype] * delta_r);
  fpair = decay_const[itype][jtype] * pauli;

  fforce = factor_lj * fpair/r;
  return factor_lj * (pauli - offset[itype][jtype]);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPauliRepuls::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&decay_const[i][j], sizeof(double), 1, fp);
        fwrite(&overlap[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        fwrite(&amp[i][j], sizeof(double), 1, fp);
        fwrite(&R[i][j],   sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPauliRepuls::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPauliRepuls::read_restart(FILE *fp)
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
          utils::sfread(FLERR, &decay_const[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &overlap[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &amp[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &R[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&decay_const[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&overlap[i][j], 1, MPI_DOUBLE, 0, world);
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

void PairPauliRepuls::read_restart_settings(FILE *fp)
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

void PairPauliRepuls::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g\n", i, j, decay_const[i][j], overlap[i][j], cut[i][i], amp[i][j],  R[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairPauliRepuls::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g %g\n", i, j, decay_const[i][j], overlap[i][j], cut[i][j], amp[i][j], R[i][i]);
}


/* ---------------------------------------------------------------------- */

void *PairPauliRepuls::extract(const char *str, int &dim)
{
  dim =2;
  if (strcmp(str, "decay_const") == 0) return (void *) decay_const;
  if (strcmp(str, "overlap") == 0) return (void *) overlap;
  if (strcmp(str, "amp" )  == 0)  return (void *) amp;
  if (strcmp(str, "R" )  == 0)  return (void *) R;
  return nullptr;
}
