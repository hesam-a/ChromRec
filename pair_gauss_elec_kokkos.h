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
PairStyle(gauss/elec/kk, PairGaussElecKokkos<LMPDeviceType>);
PairStyle(gauss/elec/kk/device, PairGaussElecKokkos<LMPDeviceType>);
PairStyle(gauss/elec/kk/host, PairGaussElecKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_GAUSS_ELEC_KOKKOS_H
#define LMP_PAIR_GAUSS_ELEC_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_gauss_elec.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairGaussElecKokkos : public PairGaussElec {
 public:
  enum { EnabledNeighFlags = FULL | HALFTHREAD | HALF };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairGaussElecKokkos(class LAMMPS *);
  ~PairGaussElecKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  struct params_gauss {
    KOKKOS_INLINE_FUNCTION
    params_gauss() { cutsq = 0; amp = 0; std_dev = 0; offset=0; };
    KOKKOS_INLINE_FUNCTION
    params_gauss(int /*i*/) { cutsq = 0; amp = 0; std_dev = 0; offset=0; };
    F_FLOAT cutsq, amp, std_dev, offset;
  };

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  Kokkos::DualView<params_gauss**, Kokkos::LayoutRight, DeviceType> k_params;
  typename Kokkos::DualView<params_gauss**, Kokkos::LayoutRight, DeviceType>::t_dev_const_um params;
  params_gauss m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // Hardwired to space for 12 atom types
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
  friend struct PairComputeFunctor<PairGaussElecKokkos, FULL, true, 0>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, FULL, true, 1>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, HALF, true>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, HALFTHREAD, true>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, FULL, false, 0>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, FULL, false, 1>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, HALF, false>;
  friend struct PairComputeFunctor<PairGaussElecKokkos, HALFTHREAD, false>;
  friend EV_FLOAT pair_compute_neighlist<PairGaussElecKokkos, FULL, 0>(PairGaussElecKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairGaussElecKokkos, FULL, 1>(PairGaussElecKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairGaussElecKokkos, HALF>(PairGaussElecKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairGaussElecKokkos, HALFTHREAD>(PairGaussElecKokkos*, NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairGaussElecKokkos>(PairGaussElecKokkos*, NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairGaussElecKokkos>(PairGaussElecKokkos*);
};

}

#endif
#endif
