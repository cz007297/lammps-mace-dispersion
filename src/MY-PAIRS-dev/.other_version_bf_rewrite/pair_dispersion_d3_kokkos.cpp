#include "pair_dispersion_d3_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "update.h"
#include "neigh_list_kokkos.h"
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

#include <d3_parameters.h>


namespace LAMMPS_NS {

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp) :
  PairDispersionD3(lmp),
  nmax(0),
  s6(0.0), s8(0.0), s18(0.0), rs6(0.0), rs18(0.0),
  a1(0.0), a2(0.0), alpha(0.0), alpha6(0.0), alpha8(0.0),
  k_rthr(0.0), k_cn_thr(0.0)
{
  // Kokkos default: no allocation until allocate() is called
  // Optionally set defaults for damping_type etc.
  damping_type = "zero_damping";
}

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::~PairDispersionD3Kokkos()
{
  // Kokkos Views free automatically when out of scope
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_allocate()
{
  // Make sure atom types known
  int ntypes = atom->ntypes;

  // 1D views (length = ntypes+1 for type indices starting at 1)
  k_r2r4   = view_1d("r2r4", ntypes+1);
  k_rcov   = view_1d("rcov", ntypes+1);
  k_mxci   = view_1d("mxci", ntypes+1);
  k_cn     = view_1d("cn",   ntypes+1);
  k_dc6    = view_1d("dc6",  ntypes+1);
  // 2D views
  k_r0ab   = view_2d("r0ab", ntypes+1, ntypes+1);
  k_cutsq  = view_2d("cutsq", ntypes+1, ntypes+1);

  // 5D view — for full c6 table
  k_c6ab   = view_5d("c6ab",
                     ntypes+1, ntypes+1,
                     /*dim3*/ 1, /*dim4*/ 1, /*dim5*/ 1, /*dim6*/ 1);
  // Adjust dims to match your table

  // Now fill host mirrors from ref tables
  auto h_r2r4 = Kokkos::create_mirror_view(k_r2r4);
  auto h_rcov = Kokkos::create_mirror_view(k_rcov);
  auto h_r0ab = Kokkos::create_mirror_view(k_r0ab);
  auto h_c6ab = Kokkos::create_mirror_view(k_c6ab);

  for (int i = 1; i <= ntypes; i++) {
    h_r2r4(i) = r2r4_ref[i];
    h_rcov(i) = rcov_ref[i];
    for (int j = 1; j <= ntypes; j++) {
      h_r0ab(i,j) = r0ab_table[i][j];
      // Fill c6ab appropriately
      h_c6ab(i,j,0,0,0,0) = c6ab_table[i][j];
    }
  }

  // Copy to device
  Kokkos::deep_copy(k_r2r4, h_r2r4);
  Kokkos::deep_copy(k_rcov, h_rcov);
  Kokkos::deep_copy(k_r0ab, h_r0ab);
  Kokkos::deep_copy(k_c6ab, h_c6ab);
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal pair_style dispersion/d3/kk");

  // Example: first arg is damping type
  damping_type = std::string(arg[0]);

  // Optional parameters
  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"s6") == 0) {
      s6 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"s8") == 0) {
      s8 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else {
      error->all(FLERR,"Unknown keyword in dispersion/d3/kk settings");
    }
  }

  // Possibly call set_funcpar(damping_type) to setup a1, a2, alpha...
  set_funcpar(damping_type);
}

} // namespace LAMMPS_NS
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_read_r0ab(const int *atomic_numbers, int ntypes)
{
  // Host mirror of device view
  auto h_r0ab = Kokkos::create_mirror_view(k_r0ab);

  for (int i = 1; i <= ntypes; i++) {
    for (int j = 1; j <= ntypes; j++) {
      h_r0ab(i,j) =
        r0ab_table[ atomic_numbers[i - 1] - 1 ]
                  [ atomic_numbers[j - 1] - 1 ];
    }
  }

  // Copy populated host data to device once
  Kokkos::deep_copy(k_r0ab, h_r0ab);
}



template<class DeviceType>
bool PairDispersionD3Kokkos<DeviceType>::k_is_int_in_array(int value, const host_view_1d &view) {
  for (size_t i = 0; i < view.extent(0); i++) {
    if (view(i) == value) return true;
  }
  return false;
}


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_set_limit_in_pars_array(view_1d &parsview, double limit) {
  for (size_t i = 0; i < parsview.extent(0); i++) {
    if (parsview(i) > limit) parsview(i) = limit;
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_read_c6ab(const int *atomic_numbers, int ntypes)
{
  // host mirrors for GPU data
  auto h_mxci = Kokkos::create_mirror_view(k_mxci);
  auto h_c6ab = Kokkos::create_mirror_view(k_c6ab);

  // zero-init mxci on host
  for (int i = 0; i <= ntypes; i++) h_mxci(i) = 0;

  int grid_i = 0, grid_j = 0;

  for (int i = 0; i < N_PARS_ROWS; i++) {
    const double ref_c6  = c6ab_table[i][0];
    const int    atom_number_1 = static_cast<int>(std::round(c6ab_table[i][1]));
    const int    atom_number_2 = static_cast<int>(std::round(c6ab_table[i][2]));

    set_limit_in_pars_array(atom_number_1, atom_number_2, grid_i, grid_j);

    std::vector<int> idx_atoms_1 = is_int_in_array(atomic_numbers, ntypes, atom_number_1);
    if (idx_atoms_1.empty()) continue;

    std::vector<int> idx_atoms_2 = is_int_in_array(atomic_numbers, ntypes, atom_number_2);
    if (idx_atoms_2.empty()) continue;

    const double ref_cn1 = c6ab_table[i][3];
    const double ref_cn2 = c6ab_table[i][4];

    // Fill symmetric entries
    for (int idx_atom_1 : idx_atoms_1) {
      for (int idx_atom_2 : idx_atoms_2) {
        // track maximum cutoff index per atom
        h_mxci(idx_atom_1) = std::max(h_mxci(idx_atom_1), grid_i);
        h_mxci(idx_atom_2) = std::max(h_mxci(idx_atom_2), grid_j);

        // direct order
        h_c6ab(idx_atom_1, idx_atom_2, grid_i, grid_j, 0) = ref_c6;
        h_c6ab(idx_atom_1, idx_atom_2, grid_i, grid_j, 1) = ref_cn1;
        h_c6ab(idx_atom_1, idx_atom_2, grid_i, grid_j, 2) = ref_cn2;
        // swapped order
        h_c6ab(idx_atom_2, idx_atom_1, grid_j, grid_i, 0) = ref_c6;
        h_c6ab(idx_atom_2, idx_atom_1, grid_j, grid_i, 1) = ref_cn2;
        h_c6ab(idx_atom_2, idx_atom_1, grid_j, grid_i, 2) = ref_cn1;
      }
    }
  }

  // Push fully initialized host data to device
  Kokkos::deep_copy(k_mxci, h_mxci);
  Kokkos::deep_copy(k_c6ab, h_c6ab);
}


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_calc_coordination_number()
{
  // Prepare device views and constants
  auto d_x    = atomKK->k_x.template view<DeviceType>();
  auto d_type = atomKK->k_type.template view<DeviceType>();

  const int l_nlocal      = atom->nlocal;
  const int l_newton_pair = force->newton_pair;

  if (atom->nmax > nmax) {
    nmax  = atom->nmax;
    k_cn  = view_1d("cn",   nmax);
    k_dc6 = view_1d("dc6",  nmax);
    memoryKK->grow_kokkos(k_dc6, nmax, "pair:dc6");
  }

  if (l_newton_pair) {
    Kokkos::deep_copy(k_cn .template view<DeviceType>(), 0.0);
    Kokkos::deep_copy(k_dc6.template view<DeviceType>(), 0.0);
  } else {
    Kokkos::deep_copy(Kokkos::subview(k_cn .template view<DeviceType>(),
                                      std::make_pair(0, l_nlocal)), 0.0);
    Kokkos::deep_copy(Kokkos::subview(k_dc6.template view<DeviceType>(),
                                      std::make_pair(0, l_nlocal)), 0.0);
  }

  // Instantiate the functor with all needed data
  PairDispersionD3Kokkos_CalcCN<DeviceType> functor{
    d_x,
    d_type,
    view_1i("ilist", nmax),
    view_1i("numneigh", nmax),
    view_2i("firstneigh", nmax),
    k_cn.template view<DeviceType>(),
    k_rcov.template view<DeviceType>(),
    l_nlocal,
    l_newton_pair,
    cn_thr,
    K1,
    autoang
  };

  // Launch the kernel
  Kokkos::parallel_for(
    "CalcCN",
    Kokkos::RangePolicy<DeviceType>(0, list->inum),
    functor
  );

  k_cn.template modify<DeviceType>();

  communicationStage = 1;
  if (l_newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);
}



template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_coeff(int narg, char **arg)
{
  const int ntypes = atom->ntypes;
  if (narg != ntypes + 2)
    error->all(FLERR, "Pair_coeff * * needs: element1 element2 ...");

  if (!allocated) allocate();

  // Build atomic number list for each type
  std::vector<int> atomic_numbers(ntypes);
  for (int i = 0; i < ntypes; i++) {
    std::string element = arg[i + 2];
    atomic_numbers[i] = find_atomic_number(element);
  }

  // Mark all pairs as set
  for (int i = 1; i <= ntypes; i++)
    for (int j = 1; j <= ntypes; j++)
      setflag[i][j] = 1;

  // Fill simple scalar tables on host mirrors
  auto h_r2r4 = Kokkos::create_mirror_view(k_r2r4);
  auto h_rcov = Kokkos::create_mirror_view(k_rcov);

  for (int i = 1; i <= ntypes; i++) {
    h_r2r4(i) = r2r4_ref[ atomic_numbers[i - 1] ];
    h_rcov(i) = rcov_ref[ atomic_numbers[i - 1] ];
  }

  // special j for use in compute
  //Kokkos::View<double*, Kokkos::HostSpace> h_special_lj("h_special_lj", 4);
  //for (int k = 0; k < 4; ++k) { h_special_lj(k) = force->special_lj[k] }' ;

  // Push to device once
  Kokkos::deep_copy(k_r2r4, h_r2r4);
  Kokkos::deep_copy(k_rcov, h_rcov);

  // Fill and transfer the multi‑dimensional parameter tables
  read_r0ab(atomic_numbers.data(), ntypes); // now just a 2D table lookup
  read_c6ab(atomic_numbers.data(), ntypes); // symmetric, grid‑aware fill
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_dC6(
    int iat, int jat, double cni, double cnj,
    view_5d d_c6ab, view_1d k_mxci,
    double l_autoev, double l_autoang, double l_K3,
    view_1d c6_res)
{
  DevOpC6Functor<DeviceType> functor{
    iat, jat, cni, cnj,
    d_c6ab, k_mxci,
    l_autoev, l_autoang, l_K3,
    c6_res
  };

  // Launch — single iteration here; could be batched for many pairs
  Kokkos::parallel_for("DerivativeC6Functor", Kokkos::RangePolicy<DeviceType>(0, 1), functor);
}

// Helper is as we sketched earlier:
// template<class DeviceType> KOKKOS_INLINE_FUNCTION
// void d_get_dC6(int iat, int jat, double cni, double cnj,
//                 const Kokkos::View<double*****,DeviceType>& c6ab_d,
//                 const Kokkos::View<int*,DeviceType>& mxci_d,
//                 double autoev_l, double autoang_l, double K3_l,
//                 double* res);

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_computeKK(int eflag, int vflag)
{
  // --- host-side setup and flags
  std::unordered_map<std::string,int> dampingMap = {{"original",1},{"zerom",2},{"bj",3},{"bjm",4}};
  const int dampingCode = dampingMap[damping_type];

  ev_init(eflag, vflag);

  // Stage 1: CN on device (your Kokkos calc_coordination_number)
  calc_coordination_number_kokkos<DeviceType>();

  // Constants captured locally
  const int    l_nlocal       = atom->nlocal;
  const int    l_newton_pair  = force->newton_pair;
  const double l_rs6          = rs6;
  const double l_rs8          = rs8;
  const double l_s6           = s6;
  const double l_s8           = s8;
  const double l_alpha        = alpha;
  const double l_a1           = a1;
  const double l_a2           = a2;
  const double l_autoang      = autoang;
  const double l_autoev       = autoev;
  const double l_K3           = K3;
  const double l_cn_thr       = cn_thr;
  
  // Device views
  auto d_x      = atomKK->k_x.template view<DeviceType>();     // double** [nall,3]
  auto d_f      = atomKK->k_f.template view<DeviceType>();     // double** [nall,3]
  auto d_type   = atomKK->k_type.template view<DeviceType>();  // int*    [nall]
  auto d_cn     = k_cn.template view<DeviceType>();         // double* [nall]
  auto d_dc6    = k_dc6.template view<DeviceType>();        // double* [nall]
  auto d_rcov   = k_rcov.template view<DeviceType>();
  auto d_cutsq  = k_cutsq;      // Kokkos::View<double**,DeviceType>
  auto d_r0ab   = k_r0ab;       // Kokkos::View<double**,DeviceType>
  auto d_r2r4   = k_r2r4;       // Kokkos::View<double*,DeviceType>

  auto d_c6ab   = k_c6ab;       // Kokkos::View<double*****,DeviceType>
  auto d_mxci   = k_mxci;       // Kokkos::View<int*,DeviceType>

  // Neighbor list (device)
  int     d_inum;
  view_1i d_ilist, d_numneigh;
  view_2i d_firstneigh;
 
  d_inum            = list->inum;
  d_ilist           = list->ilist;
  d_numneigh        = list->numneigh; 
  d_firstneigh      = list->firstneigh;  
  auto d_special_lj = view_1dconst("SpecialLJ", l_nlocal) ; // Kokkos::View<double*,DeviceType>
  

  // Zero dc6 on device (we’ll accumulate into it)
  Kokkos::deep_copy(d_dc6, 0.0);

  // Optional energy reduction
  double evdwl_sum = 0.0;

  // --- First kernel: dispersion forces and dc6 accumulation
  Kokkos::parallel_reduce(
    "pair_d3_stage1",
    Kokkos::RangePolicy<DeviceType>(0, list->inum),
    KOKKOS_LAMBDA(const int ii, double& l_evdwl) {

      const int l_i     = d_ilist(ii);
      const double l_xi = d_x(l_i,0), l_yi = d_x(l_i,1), l_zi = d_x(l_i,2);
      const double l_cni = d_cn(l_i);

      const int l_jnum   = d_numneigh(l_i);
      const int* l_jlist = d_firstneigh(l_i);

      for (int jj = 0; jj < l_jnum; jj++) {
        const int l_j = l_jlist[jj];
        const double l_factor_lj = d_special_lj(this->sbmask(l_j)); // device-safe sbmask()
        const int l_j = (l_j & NEIGHMASK);
        const double l_cnj = d_cn(l_j);

        const double l_dx = l_xi - d_x(l_j,0);
        const double l_dy = l_yi - d_x(l_j,1);
        const double l_dz = l_zi - d_x(l_j,2);
        const double l_rsq = l_dx*l_dx + l_dy*l_dy + l_dz*l_dz;

        if (l_rsq >= d_cutsq(d_type(l_i),d_type(l_j))) continue;

        const double l_r     = sqrt(l_rsq);
        const double l_r2inv = 1.0/l_rsq;
        const double l_r4inv = l_r2inv*l_r2inv;
        const double l_r6inv = l_r4inv*l_r2inv;
        const double l_r8inv = l_r4inv*l_r4inv;
        const double l_r10inv= l_r8inv*l_r2inv;

        // C6 and derivatives wrt CN
        view_1d  l_c6res;
        d_dC6 = k_dC6<DeviceType>(d_type(l_i), d_type(l_j), l_cni, l_cnj,
                              d_c6ab, d_mxci, l_autoev, l_autoang, l_K3, l_c6res);
        const double l_C6 = l_c6res[0];
        const double l_C8 = 3.0 * l_C6 * d_r2r4(d_type(l_i)) * d_r2r4(d_type(j)) * l_autoang * l_autoang;

        const double l_alpha6 = l_alpha;
        const double l_alpha8 = l_alpha + 2.0;

        double l_t6=0.0, l_t8=0.0, l_damp6=0.0, l_damp8=0.0, l_e6=0.0, l_e8=0.0, l_fpair=0.0;

        switch (dampingCode) {
          case 1: { // original
            const double l_r0 = l_r / d_r0ab(l_itype,l_jtype);
            l_t6    = pow(l_rs6 / l_r0, l_alpha6);
            l_damp6 = 1.0 / (1.0 + 6.0 * l_t6);
            l_t8    = pow(l_rs8 / l_r0, l_alpha8);
            l_damp8 = 1.0 / (1.0 + 6.0 * l_t8);
            l_e6 = l_C6 * l_damp6 * l_r6inv;
            l_e8 = l_C8 * l_damp8 * l_r8inv;
            const double l_tmp6 = 6.0 * l_s6 * l_C6 * l_r8inv  * l_damp6;
            const double l_tmp8 = 8.0 * l_s8 * l_C8 * l_r10inv * l_damp8;
            const double l_f1 = -(l_tmp6 + l_tmp8);
            const double l_f2 = l_tmp6 * l_alpha6 * l_t6 * l_damp6
                                       + 0.75 * l_tmp8 * l_alpha8 * l_t8 * l_damp8;
            l_fpair = (l_f1 + l_f2) * l_factor_lj;
          } break;

          case 2: { // zerom
            const double l_r0 = d_r0ab(l_itype,l_jtype);
            l_t6    = pow((l_r / (l_rs6 * l_r0)) + l_rs8 * l_r0, -l_alpha6);
            l_damp6 = 1.0 / (1.0 + 6.0 * l_t6);
            l_t8    = pow((l_r / l_r0) + l_rs8 * l_r0, -l_alpha8);
            l_damp8 = 1.0 / (1.0 + 6.0 * l_t8);
            l_e6 = l_C6 * l_damp6 * l_r6inv;
            l_e8 = l_C8 * l_damp8 * l_r8inv;
            const double l_tmp6 = 6.0 * l_s6 * l_C6 * l_r8inv  * l_damp6;
            const double l_tmp8 = 8.0 * l_s8 * l_C8 * l_r10inv * l_damp8;
            const double l_f1 = -(l_tmp6 + l_tmp8);
            const double l_fp26 = l_tmp6 * l_alpha6 * l_t6 * l_damp6 * l_r / (l_r + l_rs6 * l_rs8 * l_r0 * l_r0);
            const double l_fp28 = l_tmp8 * l_alpha8 * l_t8 * l_damp8 * l_r / (l_r + l_rs8 * l_r0 * l_r0);
            const double l_f2 = l_fp26 + 0.75 * l_fp28;
            l_fpair = (l_f1 + l_f2) * l_factor_lj;
          } break;

          case 3: // bj (Becke–Johnson)
          case 4: { // bjm
            const double l_r0 = sqrt(l_C8 / l_C6);
            const double l_r4 = l_rsq * l_rsq;
            const double l_r6 = l_r4 * l_rsq;
            const double l_r8 = l_r4 * l_r4;
            l_t6 = l_r6 + pow((l_a1 * l_r0 + l_a2), 6);
            l_t8 = l_r8 + pow((l_a1 * l_r0 + l_a2), 8);
            l_e6 = l_C6 / l_t6;
            l_e8 = l_C8 / l_t8;
            const double l_tmp6 = 6.0 * l_s6 * l_C6 * l_r4 / (l_t6 * l_t6);
            const double l_tmp8 = 8.0 * l_s8 * l_C8 * l_r6 / (l_t8 * l_t8);
            l_fpair = -(l_tmp6 + l_tmp8) * l_factor_lj;
          } break;
        }

        // Energy (negative sign as in your original)
        if (eflag) {
          const double l_evdwl = -(l_s6 * l_e6 + l_s8 * l_e8) * l_factor_lj;
          evdwl += l_evdwl;
        }

        // dc6 accumulation
        const double l_rest = (l_s6 * l_e6 + l_s8 * l_e8) / l_C6;
        Kokkos::atomic_add(&d_dc6(l_i), l_rest * l_c6res[1]);
        if (l_newton_pair || l_j < l_nlocal) {
          Kokkos::atomic_add(&d_dc6(l_j), l_rest * l_c6res[2]);
        }

        // Forces
        const double l_fx = l_dx * l_fpair;
        const double l_fy = l_dy * l_fpair;
        const double l_fz = l_dz * l_fpair;

        Kokkos::atomic_add(&d_f(l_i,0),  l_fx);
        Kokkos::atomic_add(&d_f(l_i,1),  l_fy);
        Kokkos::atomic_add(&d_f(l_i,2),  l_fz);
        if (l_newton_pair || l_j < l_nlocal) {
          Kokkos::atomic_add(&d_f(l_j,0), -l_fx);
          Kokkos::atomic_add(&d_f(l_j,1), -l_fy);
          Kokkos::atomic_add(&d_f(l_j,2), -l_fz);
        }

        if (evflag) {
          // equivalent to ev_tally(l_i,l_j, l_nlocal, l_newton_pair, evdwl, 0.0, fpair, delx, dely, delz)
          const double evdw_here = evdwl; // already computed
          const double l_vxx = l_delx * l_fpair;
          const double l_vyy = l_dely * l_fpair;
          const double l_vzz = l_delz * l_fpair;
          const double l_vxy = l_delx * l_fpair; // or appropriate shear terms
          const double l_vxz = l_delx * l_fpair;
          const double l_vyz = l_dely * l_fpair;
        
          EV_ATOM(l_i).evdwl += evdw_here;
          EV_ATOM(l_i).v[0]  += l_vxx;
          EV_ATOM(l_i).v[1]  += l_vyy;
          EV_ATOM(l_i).v[2]  += l_vzz;
          EV_ATOM(l_i).v[3]  += l_vxy;
          EV_ATOM(l_i).v[4]  += l_vxz;
          EV_ATOM(l_i).v[5]  += l_vyz;
        
          if (l_newton_pair || l_j < l_nlocal) {
            EV_ATOM(l_j).evdwl += evdw_here;
            EV_ATOM(l_j).v[0]  += l_vxx;
            EV_ATOM(l_j).v[1]  += l_vyy; 
            EV_ATOM(l_j).v[2]  += l_vzz; 
            EV_ATOM(l_j).v[3]  += l_vxy;
            EV_ATOM(l_j).v[4]  += l_vxz;
            EV_ATOM(l_j).v[5]  += l_vxy;
          }
        }

// ev_tally omitted here; use Kokkos-compatible tally helpers if needed
      }
    },
    evdwl_sum)

  // Accumulate energy on host if requested
  if (eflag) {
    l_evdwl += evdwl_sum;
  }

  // --- Comm Stage 2: synchronize dc6 owner/ghosts
  dc6_dual.modify<DeviceType>();           // device updated
  dc6_dual.sync<HostSpace>();              // host sees latest for MPI
  communicationStage = 2;
  if (l_newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);
  dc6_dual.modify<HostSpace>();            // host just wrote ghosts
  dc6_dual.sync<DeviceType>();             // push back to device

  // --- Second kernel: CN-derivative forces (uses dc6)
  Kokkos::parallel_for(
    "pair_d3_stage2",
    Kokkos::RangePolicy<DeviceType>(0, list->inum),
    KOKKOS_LAMBDA(const int ii) {

      const int l_i     = d_ilist(ii);
      const int l_itype = d_type(l_i);
      const double l_xi = d_x(l_i,0), l_yi = d_x(l_i,1), l_zi = d_x(l_i,2);

      const int l_jnum   = d_numneigh(l_i);
      const int* l_jlist = d_firstneigh(l_i);

      for (int jj = 0; jj < l_jnum; jj++) {
        int l_j = l_jlist[jj];
        const double l_factor_lj = d_special_lj(this->sbmask(l_j));
        const int l_j = (l_j & NEIGHMASK);
        const int l_jtype = d_type(l_j);

        const double l_dx = l_xi - d_x(l_j,0);
        const double l_dy = l_yi - d_x(l_j,1);
        const double l_dz = l_zi - d_x(l_j,2);
        const double l_rsq = l_dx*l_dx + l_dy*l_dy + l_dz*l_dz;

        if (l_rsq >= d_cutsq(l_itype,l_jtype)) continue;

        double l_dcn = 0.0;
        if (l_rsq < l_cn_thr) {
          const double l_r     = sqrt(l_rsq);
          const double l_rcov  = (d_rcov(d_type(j)) + d_rcov(d_type(j))) * l_autoang; // d_rcov: View<double*,DeviceType>
          const double l_expt  = exp(-K1 * (l_rcov / l_r - 1.0));
          l_dcn = -K1 * l_rcov * expt / (l_rsq * (l_expt + 1.0) * (l_expt + 1.0));
        }

        const double l_r     = sqrt(l_rsq);
        const double l_fpair = l_factor_lj * l_dcn * (d_dc6(l_i) + d_dc6(l_j)) / l_r;

        const double l_fx = l_dx * l_fpair;
        const double l_fy = l_dy * l_fpair;
        const double l_fz = l_dz * l_fpair;

        Kokkos::atomic_add(&d_f(l_i,0),  l_fx);
        Kokkos::atomic_add(&d_f(l_i,1),  l_fy);
        Kokkos::atomic_add(&d_f(l_i,2),  l_fz);
        if (l_newton_pair || l_j < l_nlocal) {
          Kokkos::atomic_add(&d_f(l_j,0), -l_fx);
          Kokkos::atomic_add(&d_f(l_j,1), -l_fy);
          Kokkos::atomic_add(&d_f(l_j,2), -l_fz);
        }

        // ev_tally for virial from this stage can be added similarly if needed
        if (evflag) {
          // equivalent to ev_tally(i,j, l_nlocal, l_newton_pair, evdwl, 0.0, fpair, delx, dely, delz)
          const double evdw_here = evdwl; // already computed
          const double l_vxx = l_dx * l_fpair;
          const double l_vyy = l_dy * l_fpair;
          const double l_vzz = l_dz * l_fpair;
          const double l_vxy = l_dx * l_fpair; // or appropriate shear terms
          const double l_vxz = l_dx * l_fpair;
          const double l_vyz = l_dy * l_fpair;
        
          EV_ATOM(l_i).v[0]  += l_vxx;
          EV_ATOM(l_i).v[1]  += l_vyy;
          EV_ATOM(l_i).v[2]  += l_vzz;
          EV_ATOM(l_i).v[3]  += l_vxy;
          EV_ATOM(l_i).v[4]  += l_vxz;
          EV_ATOM(l_i).v[5]  += l_vyz;
        
          if (l_newton_pair || l_j < l_nlocal) {
            EV_ATOM(l_j).v[0]  += l_vxx;
            EV_ATOM(l_j).v[1]  += l_vyy; 
            EV_ATOM(l_j).v[2]  += l_vzz; 
            EV_ATOM(l_j).v[3]  += l_vxy;
            EV_ATOM(l_j).v[4]  += l_vxz;
            EV_ATOM(l_j).v[5]  += l_vxy;
          }
        }
      }
    });

  if (vflag_fdotr) virial_fdotr_compute(); // host-side utility; ok to call after kernels
}


template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::k_init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, "All pair coeffs are not set");

  // Mirror, modify, copy back to device
  auto h_r0ab = Kokkos::create_mirror_view(k_r0ab);
  Kokkos::deep_copy(h_r0ab, k_r0ab);

  h_r0ab(j,i) = h_r0ab(i,j);

  Kokkos::deep_copy(k_r0ab, h_r0ab);

  return std::sqrt(rthr);
}
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style D3 requires atom IDs");

  // If you later re‑enable this, uncomment to enforce Newton pair requirement
  // if (force->l_newton_pair == 0)
  //   error->all(FLERR, "Pair style D3 requires newton pair on");

  // Request a half neighbor list
  neighbor->add_request(this);
}


template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::k_pack_forward_comm(int n, int *list,
                                                          double *buf,
                                                          int /*pbc_flag*/,
                                                          int* /*pbc*/)
{
  // Bring device data to host for packing
  auto h_cn  = Kokkos::create_mirror_view(k_cn);
  auto h_dc6 = Kokkos::create_mirror_view(k_dc6);
  Kokkos::deep_copy(h_cn,  k_cn);
  Kokkos::deep_copy(h_dc6, k_dc6);

  int m = 0;
  if (communicationStage == 1) {
    for (int ii = 0; ii < n; ii++) {
      int j = list[ii];
      buf[m++] = h_cn(j);
    }
  }
  if (communicationStage == 2) {
    for (int ii = 0; ii < n; ii++) {
      int j = list[ii];
      buf[m++] = h_dc6(j);
    }
  }
  return m;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_unpack_forward_comm(int n, int first,
                                                             double *buf)
{
  auto h_cn  = Kokkos::create_mirror_view(k_cn);
  auto h_dc6 = Kokkos::create_mirror_view(k_dc6);
  Kokkos::deep_copy(h_cn,  k_cn);
  Kokkos::deep_copy(h_dc6, k_dc6);

  int m = 0;
  int last = first + n;
  if (communicationStage == 1) {
    for (int i = first; i < last; i++) h_cn(i) = buf[m++];
    Kokkos::deep_copy(k_cn, h_cn);
  }
  if (communicationStage == 2) {
    for (int i = first; i < last; i++) h_dc6(i) = buf[m++];
    Kokkos::deep_copy(k_dc6, h_dc6);
  }
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::k_pack_reverse_comm(int n, int first,
                                                          double *buf)
{
  auto h_cn  = Kokkos::create_mirror_view(k_cn);
  auto h_dc6 = Kokkos::create_mirror_view(k_dc6);
  Kokkos::deep_copy(h_cn,  k_cn);
  Kokkos::deep_copy(h_dc6, k_dc6);

  int m = 0;
  int last = first + n;
  if (communicationStage == 1) {
    for (int i = first; i < last; i++) buf[m++] = h_cn(i);
  }
  if (communicationStage == 2) {
    for (int i = first; i < last; i++) buf[m++] = h_dc6(i);
  }
  return m;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_unpack_reverse_comm(int n, int *list,
                                                             double *buf)
{
  auto h_cn  = Kokkos::create_mirror_view(k_cn);
  auto h_dc6 = Kokkos::create_mirror_view(k_dc6);
  Kokkos::deep_copy(h_cn,  k_cn);
  Kokkos::deep_copy(h_dc6, k_dc6);

  int m = 0;
  if (communicationStage == 1) {
    for (int ii = 0; ii < n; ii++) {
      int j = list[ii];
      h_cn(j) += buf[m++];
    }
    Kokkos::deep_copy(k_cn, h_cn);
  }
  if (communicationStage == 2) {
    for (int ii = 0; ii < n; ii++) {
      int j = list[ii];
      h_dc6(j) += buf[m++];
    }
    Kokkos::deep_copy(k_dc6, h_dc6);
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::k_set_funcpar()
{
  // First, run the original CPU logic
  PairDispersionD3::set_funcpar();

  // Mirror and populate from base's host array (assuming it's named funcpar)
  auto h_funcpar = Kokkos::create_mirror_view(k_funcpar);
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = 1; j <= atom->ntypes; j++) {
      for (int k = 0; k < NPAR; k++) { // NPAR = number of parameters per pair
        h_funcpar(i,j,k) = funcpar[i][j][k];
      }
    }
  }

  // Push to device
  Kokkos::deep_copy(k_funcpar, h_funcpar);
}


