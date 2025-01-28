// clang-format off
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

#include "pair_pauli_repuls_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
template<class DeviceType>
PairPauliRepulsKokkos<DeviceType>::PairPauliRepulsKokkos(LAMMPS *lmp) : PairPauliRepuls(lmp) {

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  //datamask_read = X_MASK | F_MASK | TYPE_MASK;
  //datamask_modify = F_MASK;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
PairPauliRepulsKokkos<DeviceType>::~PairPauliRepulsKokkos() {
  if (copymode) return; // skip destructor if in copy mode

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}


/* ---------------------------------------------------------------------- */
template<class DeviceType>
void PairPauliRepulsKokkos<DeviceType>::settings(int narg, char **arg) {
  if (narg > 2) error->all(FLERR,"Illegal pair_style command");

  PairPauliRepuls::settings(1,arg);
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
void PairPauliRepulsKokkos<DeviceType>::init_style()

{
  // call the base class, which does neighbor->add_request(this,...)
  PairPauliRepuls::init_style();

  // KOKKOS logic
//  neighflag = lmp->kokkos->neighflag;
//  auto request = neighbor->find_request(this);
//  if (request) {
//    request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType>);
//    request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
//  }
  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
  
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
double PairPauliRepulsKokkos<DeviceType>::init_one(int i, int j) {
  double cutone = PairPauliRepuls::init_one(i, j);

  // Populate the Kokkos DualView for pair parameters
  k_params.h_view(i, j).amp = amp[i][j];
  k_params.h_view(i, j).decay_const = decay_const[i][j];
  k_params.h_view(i, j).overlap = overlap[i][j];
  k_params.h_view(i, j).R = R[i][j];
  k_params.h_view(i, j).offset = offset[i][j];
  k_params.h_view(i, j).cutsq = cutone * cutone;

  // Symmetrize for KOKKOS views
  k_params.h_view(j, i) = k_params.h_view(i, j);

  // Handle the minimal params array for small optimizations
  if (i < MAX_TYPES_STACKPARAMS + 1 && j < MAX_TYPES_STACKPARAMS + 1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i, j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone * cutone;
  }

  // Update Kokkos cutsq view
  k_cutsq.h_view(i, j) = k_cutsq.h_view(j, i) = cutone * cutone;

  // Mark host views as modified and sync them with the device
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  //printf("Init pair_coeff: i=%d, j=%d, amp=%f, decay_const=%f, overlap=%f, R=%f, offset=%f\n",
  //     i, j, amp[i][j], decay_const[i][j], overlap[i][j], R[i][j], offset[i][j]);

  return cutone;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
void PairPauliRepulsKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // Reallocate per-atom arrays if necessary
  
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  // Sync data to device
  atomKK->sync(execution_space, datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();


  //k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space, F_MASK);

  // Assign KOKKOS views
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  copymode = 1;


  // Use pair_compute for neighbor iteration
  EV_FLOAT ev = pair_compute<PairPauliRepulsKokkos<DeviceType>,void>(this,(NeighListKokkos<DeviceType>*)list);

  // Accumulate global energy and virial
  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  // Sync per-atom arrays back to host if needed
  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairPauliRepulsKokkos<DeviceType>::
compute_fpair(const F_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {
  const F_FLOAT r = sqrt(rsq);
  const F_FLOAT overlap = (STACKPARAMS ? m_params[itype][jtype].overlap : params(itype, jtype).overlap);
  const F_FLOAT R = (STACKPARAMS ? m_params[itype][jtype].R : params(itype, jtype).R);
  const F_FLOAT amp = (STACKPARAMS ? m_params[itype][jtype].amp : params(itype, jtype).amp);
  const F_FLOAT decay_const = (STACKPARAMS ? m_params[itype][jtype].decay_const : params(itype, jtype).decay_const);

  const F_FLOAT delta_r = r - overlap * R;
  const F_FLOAT pauli = amp * exp(-decay_const * delta_r);
  const F_FLOAT fpair = decay_const * pauli / r;

//  printf("compute_fpair: rsq=%f, itype=%d, jtype=%d\n", rsq, itype, jtype);
//  printf("Params: amp=%f, decay_const=%f, overlap=%f, R=%f\n",
//       STACKPARAMS ? m_params[itype][jtype].amp : params(itype, jtype).amp,
//       STACKPARAMS ? m_params[itype][jtype].decay_const : params(itype, jtype).decay_const,
//       STACKPARAMS ? m_params[itype][jtype].overlap : params(itype, jtype).overlap,
//       STACKPARAMS ? m_params[itype][jtype].R : params(itype, jtype).R);


  return fpair;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairPauliRepulsKokkos<DeviceType>::
compute_evdwl(const F_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {
  const F_FLOAT r = sqrt(rsq);

  // Access parameters via STACKPARAMS or params
  const F_FLOAT overlap = (STACKPARAMS ? m_params[itype][jtype].overlap : params(itype, jtype).overlap);
  const F_FLOAT R = (STACKPARAMS ? m_params[itype][jtype].R : params(itype, jtype).R);
  const F_FLOAT amp = (STACKPARAMS ? m_params[itype][jtype].amp : params(itype, jtype).amp);
  const F_FLOAT decay_const = (STACKPARAMS ? m_params[itype][jtype].decay_const : params(itype, jtype).decay_const);

  const F_FLOAT delta_r = r - overlap * R;

//  printf("compute_evdwl: rsq=%f, itype=%d, jtype=%d\n", rsq, itype, jtype);
//  printf("Params: amp=%f, decay_const=%f, overlap=%f, R=%f\n",
//         STACKPARAMS ? m_params[itype][jtype].amp : params(itype, jtype).amp,
//         STACKPARAMS ? m_params[itype][jtype].decay_const : params(itype, jtype).decay_const,
//         STACKPARAMS ? m_params[itype][jtype].overlap : params(itype, jtype).overlap,
//         STACKPARAMS ? m_params[itype][jtype].R : params(itype, jtype).R);

  // Compute and return the energy
  return amp * exp(-decay_const * delta_r);
}


/* ---------------------------------------------------------------------- */
template<class DeviceType>
void PairPauliRepulsKokkos<DeviceType>::allocate() {
  PairPauliRepuls::allocate();  // Call base class allocation

  int n = atom->ntypes;

  memory->destroy(cutsq);       // Destroy original `cutsq`
  cutsq = nullptr;              // so base destructor won’t double-free
  memoryKK->create_kokkos(k_cutsq, cutsq, n+1, n+1, "pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  // Create and initialize parameters array
  k_params = Kokkos::DualView<params_pauli**, Kokkos::LayoutRight, DeviceType>("PairPauliRepuls::params", n+1, n+1);
  params = k_params.template view<DeviceType>();
  //allocated = 0;
}

/* ---------------------------------------------------------------------- */
namespace LAMMPS_NS {
template class PairPauliRepulsKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairPauliRepulsKokkos<LMPHostType>;
#endif
}
