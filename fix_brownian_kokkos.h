#ifdef FIX_CLASS
// clang-format off
FixStyle(brownian/kk,FixBrownianKokkos<LMPDeviceType>);
FixStyle(brownian/kk/device,FixBrownianKokkos<LMPDeviceType>);
FixStyle(brownian/kk/host,FixBrownianKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_BROWNIAN_KOKKOS_H
#define LMP_FIX_BROWNIAN_KOKKOS_H

#include "fix_brownian.h"          // Your CPU base class
#include "kokkos_type.h"           // For LMPDeviceType, etc.
#include "Kokkos_Random.hpp"

namespace LAMMPS_NS {

template <class DeviceType>
class FixBrownianKokkos : public FixBrownian {
 public:
  // Constructor/Destructor
  FixBrownianKokkos(class LAMMPS *lmp, int narg, char **arg);
  virtual ~FixBrownianKokkos() {}

  // Required LAMMPS fix methods
  void init() override;
  void initial_integrate(int vflag) override;

 private:
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
  //using rnd_pool_type = Kokkos::Random_XorShift64_Pool<DeviceType>;
  //rnd_pool_type rng_pool;

 protected:
  // You can store device views or random pools here

  // Host copies of needed parameters
  double dt_local;
  double g1_local, g2_local;

  // Because Brownian might have different flags
  // from your base class, keep track of them:
  bool is2d;             // dimension check
  // etc.

  // Optional: if you want a Box-Muller Gaussian
  KOKKOS_INLINE_FUNCTION
  double generate_gaussian(Kokkos::Random_XorShift64<DeviceType> &rng) const {
    // Basic Box-Muller
    double r1 = rng.drand();
    double r2 = rng.drand();
    return sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
