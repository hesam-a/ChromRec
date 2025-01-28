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
PairStyle(lock/key/kk, PairLockKeyKokkos<LMPDeviceType>);
PairStyle(lock/key/kk/device, PairLockKeyKokkos<LMPDeviceType>);
PairStyle(lock/key/kk/host, PairLockKeyKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LOCK_KEY_KOKKOS_H
#define LMP_PAIR_LOCK_KEY_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_lock_key.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairLockKeyKokkos : public PairLockKey {
 public:
  enum { EnabledNeighFlags = FULL | HALFTHREAD | HALF };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairLockKeyKokkos(class LAMMPS *);
  ~PairLockKeyKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  struct params_lock {
    KOKKOS_INLINE_FUNCTION
    params_lock() { cutsq = 0; amp = 0; std_dev = 0; offset=0; };
    KOKKOS_INLINE_FUNCTION
    params_lock(int /*i*/) { cutsq = 0; amp = 0; std_dev = 0; offset=0; };
    F_FLOAT cutsq, amp, std_dev, offset;
  };

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  Kokkos::DualView<params_lock**, Kokkos::LayoutRight, DeviceType> k_params;
  typename Kokkos::DualView<params_lock**, Kokkos::LayoutRight, DeviceType>::t_dev_const_um params;
  params_lock m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // Hardwired to space for 12 atom types
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;


  int neighflag;
  int nlocal, nall, eflag, vflag;

  void allocate() override;
  friend struct PairComputeFunctor<PairLockKeyKokkos, FULL, true, 0>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, FULL, true, 1>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, HALF, true>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, HALFTHREAD, true>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, FULL, false, 0>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, FULL, false, 1>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, HALF, false>;
  friend struct PairComputeFunctor<PairLockKeyKokkos, HALFTHREAD, false>;
  friend EV_FLOAT pair_compute_neighlist<PairLockKeyKokkos, FULL, 0>(PairLockKeyKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLockKeyKokkos, FULL, 1>(PairLockKeyKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLockKeyKokkos, HALF>(PairLockKeyKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLockKeyKokkos, HALFTHREAD>(PairLockKeyKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLockKeyKokkos>(PairLockKeyKokkos*, NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairLockKeyKokkos>(PairLockKeyKokkos*);
};

}

#endif
#endif
