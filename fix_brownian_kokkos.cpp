#include "fix_brownian_kokkos.h"

// LAMMPS headers:
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "update.h"
#include "force.h"
#include "modify.h"
#include "random_park.h"  // or whichever random generator you prefer
#include "memory_kokkos.h"

// Standard libraries:
#include <cmath>
#include <cstdlib>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
   Constructor
------------------------------------------------------------------------- */

template <class DeviceType>
FixBrownianKokkos<DeviceType>::FixBrownianKokkos(LAMMPS *lmp, int narg, char **arg):
       FixBrownian(lmp, narg, arg), //rng_pool(seed + comm->me)
       rand_pool(seed + comm->me)
{

  //rng_pool = Kokkos::Random_XorShift64_Pool<DeviceType>(seed + comm->me);
  kokkosable = 1;          // Indicate KOKKOS awareness
  fuse_integrate_flag = 1; // Indicate we can do the integration on device
  sort_device = 1;         // If we want Kokkos-based sorting
  atomKK = (AtomKokkos *) atom;			   

  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

}

/* ----------------------------------------------------------------------
   init()
   This is called once LAMMPS has set up domain, etc. The CPU "FixBrownian::init()"
   calls "FixBrownianBase::init()", which among other things sets g1 and g2.
   We replicate that logic plus set up local data.
------------------------------------------------------------------------- */

template <class DeviceType>
void FixBrownianKokkos<DeviceType>::init()
{
  // First call the CPU parent's init() to handle CPU-side checks
  FixBrownian::init();

  // Access dimension from domain
  is2d = (domain->dimension == 2);

  // CPU code does: g1 /= gamma_t; g2 *= sqrt(temp/gamma_t);
  // That is effectively done in FixBrownian::init(). So we store them in local copies:
  dt_local = dt;                 // from Fix class or "update->dt"
  g1_local = g1;                 // these are now the final "g1" after parent's init
  g2_local = g2;

  // Sync or do any other Kokkos-level initialization if needed
  // e.g. we might want to check if "atom" is actually AtomKokkos:
  if (dynamic_cast<AtomKokkos*>(atom) == nullptr) {
    error->all(FLERR,"FixBrownianKokkos requires AtomKokkos");
  }
}

/* ----------------------------------------------------------------------
   initial_integrate()
   This is the main step: we add displacements based on (f + random) * dt
   then set velocities = displacement/dt, exactly as your CPU version does.
------------------------------------------------------------------------- */

template <class DeviceType>
void FixBrownianKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{

  atomKK->sync(execution_space, X_MASK | F_MASK | V_MASK | MASK_MASK);
  atomKK->modified(execution_space, X_MASK | V_MASK);

  auto xview = atomKK->k_x.view<DeviceType>();
  auto fview = atomKK->k_f.view<DeviceType>();
  auto vview = atomKK->k_v.view<DeviceType>();
  auto maskv = atomKK->k_mask.view<DeviceType>();

  //const int nlocal = (igroup == atomKK->firstgroup) ? atomKK->nfirst : atomKK->nlocal;
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;
  const double dt_loc = dt_local;
  const double g1_loc = g1_local;
  const double g2_loc = g2_local;
  const bool   is2D   = is2d;

  const tagint groupbit_local = groupbit;
  const bool noise = noise_flag;           // from base fix
  const bool gauss_noise = gaussian_noise_flag;
  const auto   groupb = groupbit_local;

  // Perform the integration on device
  // We'll do a parallel_for with a lambda capturing everything we need.
  //auto local_pool = rng_pool;
  auto local_pool = rand_pool;

  Kokkos::parallel_for("FixBrownianKokkos::initial_integrate",
                       Kokkos::RangePolicy<DeviceType>(0,nlocal),
    KOKKOS_LAMBDA (const int i) {

      if (maskv(i) & groupb) {
        //auto rand_gen = local_pool.get_state();
        rand_type rand_gen = local_pool.get_state();

        double dx=0.0, dy=0.0, dz=0.0;

        if (!noise) {
          // No random displacement
          dx = dt_loc * g1_loc * fview(i,0);
          dy = dt_loc * g1_loc * fview(i,1);
          if (!is2D) dz = dt_loc * g1_loc * fview(i,2);

        } else {
          // noise_flag == true
          // Check uniform vs Gaussian
          double rx, ry, rz;
          if (gauss_noise) {
            // Generate 3 Gaussian random numbers (Box–Muller):
            // Or you can do your own function "generate_gaussian()"
            auto gauss = [&](Kokkos::Random_XorShift64<DeviceType> &rg) {
              double r1 = rg.drand();
              double r2 = rg.drand();
              return sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);
            };
            rx = gauss(rand_gen);
            ry = gauss(rand_gen);
            rz = is2D ? 0.0 : gauss(rand_gen);

          } else {
            // Uniform
            rx = rand_gen.drand() - 0.5;
            ry = rand_gen.drand() - 0.5;
            rz = is2D ? 0.0 : (rand_gen.drand() - 0.5);
          }

          dx = dt_loc * (g1_loc * fview(i,0) + g2_loc * rx);
          dy = dt_loc * (g1_loc * fview(i,1) + g2_loc * ry);
          if (!is2D)
            dz = dt_loc * (g1_loc * fview(i,2) + g2_loc * rz);

        }

        // Update position
        xview(i,0) += dx;
        xview(i,1) += dy;
        if (!is2D) xview(i,2) += dz;

        // Update velocity
        vview(i,0) = dx / dt_loc;
        vview(i,1) = dy / dt_loc;
        if (!is2D) vview(i,2) = dz / dt_loc;

        local_pool.free_state(rand_gen);
      }
    }
  );

  // Mark the device arrays as modified
  atomKK->modified(execution_space, X_MASK | V_MASK);
}


//
// And at the bottom, instantiate the template for GPU (and optionally for CPU Kokkos):
//
namespace LAMMPS_NS {
template class FixBrownianKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixBrownianKokkos<LMPHostType>;
#endif
} // namespace LAMMPS_NS
