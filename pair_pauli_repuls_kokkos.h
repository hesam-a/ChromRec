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
PairStyle(pauli/repuls/kk, PairPauliRepulsKokkos<LMPDeviceType>);
PairStyle(pauli/repuls/kk/device, PairPauliRepulsKokkos<LMPDeviceType>);
PairStyle(pauli/repuls/kk/host, PairPauliRepulsKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_PAULI_REPULS_KOKKOS_H
#define LMP_PAIR_PAULI_REPULS_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_pauli_repuls.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairPauliRepulsKokkos : public PairPauliRepuls {
 public:
  enum { EnabledNeighFlags = FULL | HALFTHREAD | HALF };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairPauliRepulsKokkos(class LAMMPS *);
  ~PairPauliRepulsKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  struct params_pauli {
    KOKKOS_INLINE_FUNCTION
    params_pauli() { cutsq = 0; amp = 0; decay_const = 0; overlap = 0; R = 0;offset=0; };
    KOKKOS_INLINE_FUNCTION
    params_pauli(int /*i*/) { cutsq = 0; amp = 0; decay_const = 0; overlap = 0; R = 0;offset=0; };
    F_FLOAT cutsq, amp, decay_const, overlap, R, offset;
  };

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  Kokkos::DualView<params_pauli**, Kokkos::LayoutRight, DeviceType> k_params;
  typename Kokkos::DualView<params_pauli**, Kokkos::LayoutRight, DeviceType>::t_dev_const_um params;
  params_pauli m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // Hardwired to space for 12 atom types
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
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, FULL, true, 0>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, FULL, true, 1>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, HALF, true>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, HALFTHREAD, true>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, FULL, false, 0>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, FULL, false, 1>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, HALF, false>;
  friend struct PairComputeFunctor<PairPauliRepulsKokkos, HALFTHREAD, false>;
  friend EV_FLOAT pair_compute_neighlist<PairPauliRepulsKokkos, FULL, 0>(PairPauliRepulsKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairPauliRepulsKokkos, FULL, 1>(PairPauliRepulsKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairPauliRepulsKokkos, HALF>(PairPauliRepulsKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairPauliRepulsKokkos, HALFTHREAD>(PairPauliRepulsKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairPauliRepulsKokkos>(PairPauliRepulsKokkos*, NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairPauliRepulsKokkos>(PairPauliRepulsKokkos*);
};

}

#endif
#endif
