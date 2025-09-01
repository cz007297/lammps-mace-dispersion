#include "pair_dispersion_d3_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neighbor_kokkos.h"
#include "update.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

using namespace LAMMPS_NS;

// global ad hoc parameters

static constexpr double K1 = 16.0;
static constexpr double K3 = -4.0;

static constexpr int NUM_ELEMENTS=94;
static constexpr int N_PARS_COLS=5;  // number columns in C6 table
static constexpr int N_PARS_ROWS=32395; // number of rows C6 table 

static constexpr double autoang =  0.52917725 ; // au-Bohr to angstroms
static constexpr double autoev  = 27.21140795 ; // au-Hartree to eV 


/*  reasonable choices for k3 are between 3 and 5 :
    this gives smoth curves with maxima around the integer values
    k3=3 give for CN=0 a slightly smaller value than computed
    for the free atom. This also yields to larger CN for atoms
    in larger molecules but with the same chemical environment
    which is physically not right.
    values >5 might lead to bumps in the potential.
*/

static constexpr int NUM_ELEMENTS = 94;      // maximum element number
static constexpr int N_PARS_COLS = 5;        // number of columns in C6 table
static constexpr int N_PARS_ROWS = 32385;    // number of rows in C6 table

static constexpr double autoang = 0.52917725;    // atomic units (Bohr) to Angstrom
static constexpr double autoev = 27.21140795;    // atomic units (Hartree) to eV

#include <d3_parameters.h>

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp) :
    PairKokkos<DeviceType>(lmp),
    k_r2r4("r2r4", 0),
    k_rcov("rcov", 0),
    k_mxci("mxci", 0),
    k_r0ab("r0ab", 0, 0), 
    k_c6ab("c6ab", 0, 0),
    k_cn("cn", 0),
    k_dc6("dc6", 0)
{

  atomKK = (AtomKokkos *) atom ; 
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify F_MASK | ENERGY_MASK | VIRIAL_MASK ; 

  this->nmax           = 0;
  this->k_comm_forward = 2;
  this->k_comm_reverse = 2;
  this->restartinfo    = 0;
  this->manybody_flag  = 1;
  this->one_coeff      = 1;
  this->single_enable  = 0;
  
  s6 = s8 = s18 = rs6 = rs8 = rs18 = a1 = a2 = alpha = alpha6 = alpha8 = 0.0; 
 
  kokkosable = 1;
}

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::~PairDispersionD3Kokkos()
{
  if (copymode) return;
 
  if (allocated) 
  {
  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);  
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::allocate() 
{
  this->allocated=1 ;
  int n = this->atom->ntypes; 

  memoryKK->create_kokkos(k_set_flag, "pair:setflag", n+1, n+1);
  Kokkos::deep_copy(k_set_flag, 0); 

  memoryKK->create_kokkos(k_cutsq, "pair:cutsq", n+1, n+1);
  memoryKK->create_kokkos(k_mxci, "pair:mxci", n+1);
  memoryKK->create_kokkos(k_r2r4, "pair:r2r4", n+1);
  memoryKK->create_kokkos(k_rcov, "pair:rcov", n+1); 

  memoryKK->create_kokkos(k_r0ab, "pair:r0ab", n+1, n+1);
  memoryKK->create_kokkos(k_c6ab, "pair:c6ab", n+1, n+1, 5, 5, 3,);
}


}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::settings(int narg, charg **arg)
{
  if (narg != 4) this->error->ALL(FLERR, "Pair_style dispersion/d3 needs 4 arguments");

  this->damping_type = arg[0];
  std::string functional_name = arg[1];
  std::transform(this->damping_type.begin(),
                 this->damping_type.end(),
                 this->damping_type.begin(),
                 ::tolower);
  this->rthr = utils::numeric(FLERR, arg[2], false, this->lmp);
  this->cn_thr = utils::numeric(FLERR, arg[3], false, this->lmp);

  this->rthr *= this->rthr;
  this->cn_thr *= this->cn_thr; 
  
  this->set_funcpar(functional_name);
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::find_atomic_number(std:: string &key)
{
  std::transform(this->key.begin(),
                 this->key.end(),
                 this->key.begin(),
                 ::tolower);
  if (this->key.length() == 1) key+= " ";
  this->key.resize(2); 
  
  std::vector<std::string> element_table = {
      "h ", "he", "li", "be", "b ", "c ", "n ", "o ", "f ", "ne", "na", "mg", "al", "si",
      "p ", "s ", "cl", "ar", "k ", "ca", "sc", "ti", "v ", "cr", "mn", "fe", "co", "ni",
      "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y ", "zr", "nb", "mo",
      "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i ", "xe", "cs", "ba",
      "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb",
      "lu", "hf", "ta", "w ", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po",
      "at", "rn", "fr", "ra", "ac", "th", "pa", "u ", "np", "pu"};
  for (size_t i = 0; i < element_table.size(); i++) 
  {
    if (element_table[i] == this->key)
    {
    int atomic_number i + 1;
    return this->atomic_number;
    }
  }
  return -1;
}

template<class DeviceType>
std::vector<int> PairDispersionD3Kokkos<DeviceType>::is_int_in_array(int array[], int size, int value)
{
  std::vector<int> indices;
  for (int i = 0; i < size; ++i)
  {
    if(array[i]==value indices.push_back(i+1);
  }
  return this->indices;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::read_r0ab(int *atomic_numbers, int ntypes)
{
  for (int i = 1; i <= ntypes; i++)
  {
    for (int j = i; j <= ntypes; j++)
    {
      this->k_r0ab[i][j] = r0ab_table[atomic_numbers[i-1] - 1][atomic_numbers[j-1] -1];
    }
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::set_limit_in_pars_array(int &idx_atom_1, int &idx_atom_2, 
                                                                 int &idx_i, int &idx_j)
{
  idx_i = 0;
  idx_j = 0;
  int shift = 100; 

  while (idx_atom_1 > shift) 
  {
    idx_atom_1 -= shift;
    idx_i++;
  }
  
  while (idx_atom_2 > shift) 
  {
    idx_atom_2 -= shift;
    idx_j++;
  }

}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::read_c6ab(int *atomic_numbers, int ntypes)
{
  for (int i = 0; i <= ntypes; i++) mxci[i] = 0;
  int grid_i = 0, grid_j = 0; 
  
  for (int i = 0; i < N_PARS_ROWS; i++)
  {
    const double ref_c6 = c6ab_table[i][0];
   
    int atom_number_1 = std::round(c6ab_table[i][1]);
    int atom_number_2 = std::round(c6ab_table[i][2]);

    set_limit_in_pars_array(atom_number_1, atom_number_2, grid_i, grid_j);
    
    std::vector<int> idx_atoms_1 = is_int_in_array(atomic_numbers, ntypes, atom_number_1);
    if (idx_atoms_1.empty()) continue; 

    std::vector<int> idx_atoms_2 = is_int_in_array(atomic_numbers, ntypes, atom_number_2); 
    if (idx_atoms_2.empty()) continue; 

    const double ref_cn1 = c6ab_table[i][3];
    const double ref_cn2 = c6ab_table[i][4];

    for (int idx_atom_1 : idx_atoms_1)
    {
      for (int idx_atom_2 : idx_atoms_2)
      {
        mxci[idx_atom_1] = std::max(mxci[idx_atom_1], grid_i);
        mxci[idx_atom_2] = std::max(mxci[idx_atom_2], grid_j); 
      
        c6ab[idx_atom_1][idx_atom_2][grid_i][grid_j][0] = ref_c6;
        c6ab[idx_atom_1][idx_atom_2][grid_i][grid_j][1] = ref_cn1;
        c6ab[idx_atom_1][idx_atom_2][grid_i][grid_j][2] = ref_cn2;
        c6ab[idx_atom_2][idx_atom_1][grid_j][grid_i][0] = ref_c6;
        c6ab[idx_atom_2][idx_atom_1][grid_j][grid_i][1] = ref_cn1;
        c6ab[idx_atom_2][idx_atom_1][grid_j][grid_i][2] = ref_cn2; 
      }
    }
  }
}


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  int ntypes = this->atom->ntypes;
  if (narg != ntypes + 2) error->all(FLERR, "Pair_coeff * * needs: element1 element2 ...");
  
  if (!allocated) allocate();
  
  std::string element; 
  int *atomic_numbers = (int *) malloc(sizeof(int) * ntypes);
  for (int i = 0; i < ntypes; i++)
  {
    element = arg[i+2];
    atomic_numbers[i] = find_atomic_numbers(element);
  }  
  int count = 0; 
  for (int i = 1; i <= ntypes; i++)
  {
    for (int j = 1; j <= ntypes; j++)
    {
      setflag[i][j] = 1; 
      count++; 
    }
  }
  
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i < ntypes; i++)
  {
    this->r2r4[i] = r2r4_ref[atomic_numbers[i-1]];
    this->rcov[i] = rcov_ref[atomic_numbers[i-1]];
  }
  
  read_r0ab(atomic_numbers, ntypes);
  read_c6ab(atomic_numbers, ntypes);

  free(atomic_numbers)
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_number()
{
// TODO - convert 
}

template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::get_dC6(int iat, int jat, double cni, double cnj) 
{
// TODO - convert 
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag, int vflag)
{
//TODO - convert 
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::set_funcpar(std:string &functional_name)
{
// TODO UGH import logic as is - doesn't need to be gpu-coded 
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_one(int i, int j)
{
// TODO - confused 
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_style()
{
// TODO import logic as is - doesn't need to be gpu-coded 
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_forward_comm(int n, int*list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
// TODO we'll see 
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf) 
{
//TODO we'll see 
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
// TODO we'll see
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
// TODO we'll see
} 

































