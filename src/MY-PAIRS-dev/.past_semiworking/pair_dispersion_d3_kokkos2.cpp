#include "pair_dispersion_d3_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include <cmath>
#include "error.h"
#include "comm.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "update.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "pair_kokkos.h"
#include "kokkos_base.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include "kokkos_few.h"


using namespace LAMMPS_NS;
static constexpr int NUM_ELEMENTS=94;
static constexpr int N_PARS_COLS=5;  // number columns in C6 table
static constexpr int N_PARS_ROWS=32395; // number of rows C6 table
static constexpr double K1 = 16.0;
static constexpr double K3 = -4.0;
static constexpr double autoang =  0.52917725 ;
static constexpr double autoev  = 27.21140795 ;

#include "d3_parameters.h"

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp) : PairDispersionD3(lmp), PairKokkos(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::~PairDispersionD3Kokkos()
{
  if(!copymode)
  {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
  }

}

template<class DeviceType>
