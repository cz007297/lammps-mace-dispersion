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
void PairDispersionD3Kokkos<DeviceType>::allocate()
{
  // Make sure atom types known
  int ntypes = atom->ntypes;

  // 1D views (length = ntypes+1 for type indices starting at 1)
  k_r2r4 = view_1d("r2r4", ntypes+1);
  k_rcov = view_1d("rcov", ntypes+1);
  k_mxci = view_1d("mxci", ntypes+1);
  k_cn   = view_1d("cn",   ntypes+1);
  k_dc6  = view_1d("dc6",  ntypes+1);

  // 2D views
  k_r0ab = view_2d("r0ab", ntypes+1, ntypes+1);

  // 5D view — for full c6 table
  k_c6ab = view_6d("c6ab",
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
void PairDispersionD3Kokkos<DeviceType>::settings(int narg, char **arg)
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
void PairDispersionD3Kokkos<DeviceType>::read_r0ab(const int *atomic_numbers, int ntypes)
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



template <typename T>
using host_view_1d = Kokkos::View<T*,Kokkos::HostSpace>;
bool is_int_in_array(int value, const host_view_1d<int> &view) {
  for (size_t i = 0; i < view.extent(0); i++) {
    if (view(i) == value) return true;
  }
  return false;
}


template <typename ViewType>
void set_limit_in_pars_array(ViewType &view, double limit) {
  for (size_t i = 0; i < view.extent(0); i++) {
    if (view(i) > limit) view(i) = limit;
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::read_c6ab(const int *atomic_numbers, int ntypes)
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
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_number()
{
  auto x = atomKK -> k_x.view(DeviceType);
  auto type = atomKK -> k_type.view(DeviceType);
  
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost; 
  const int newton_pair = force->newton_pair; 

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    k_cn = tdual_float_1d("cn", nmax);
    k_dc6 = tdual_float_1d("dc6", nmax);
     
    memoryKK->grow_kokkos(dc6, nmax, "pair:dc6");
  }

  if (newton_pair)
  {
    Kokkos::deep_copy(k_cn.view<DeviceType>(), 0.0);
    Kokkos::deep_copy(k_dc6.view<DeviceType>(), 0.0);
  }
  else
  {
    Kokkos::deep_copy(Kokkos::subview(cn_dual.view<DeviceType>(),
                                      std::make_pair(0,nlocal)), 0.0);
    Kokkos::deep_copy(Kokkos::subview(dc6_dual.view<DeviceType>(),
                                      std::make_pair(0,nlocal)), 0.0);
  }
  
  auto d_ilist = neighKK->d_ilist;
  auto d_numneigh = neighKK->d_numneigh; 
  auto d_firstneigh = neighKK->d_firstneigh;
 
  auto cn_view  = cn_dual.view<DeviceType>();
  auto d_rcov   = rcov_view;

 
  const double l_cn_thr = cn_thr;
  const double l_K1 = K1; 
  const double l_autoang = autoang; 
  
  Kokkos::parallel_for("CalcCN",
                        Kokkos::RangePolicy<DeviceType>(0, list->inum),
                        KOKKOS_LAMBDA (const int ii)
  {
    const int i = d_list(ii);
    const int itype = type(i);
    const int jnum  = d_numneigh(i);
    const int* jlist = d_firstneigh(i);
    
    for ( int jj = 0 ; jj < jnum ; jj++ )
    {
      int j = jlist[jj] & NEIGHMASK;
      const int jtype = type(j);

      double delx = x(i,0) - x(j,0);
      double dely = x(i,1) - x(j,1);
      double delz = x(i,2) - x(j,2);
      double rsq  = delx*delx + dely*dely + delz*delz;
      
      if (rsq > l_cn_thr) continue;
      
      double rr      = sqrt(rsq);
      double rcov_ij = (d_rcov(itype) + d_rcov(jtype)) * l_autoang;
      double cn_ij   = 1.0 / (1.0 + exp(-l_K1 * ((rcov_ij / rr) - 1.0)));

      Kokkos::atomic_add(&cn_view(i), cn_ij);
      if (newton_pair || j < nlocal) {
        Kokkos::atomic_add(&cn_view(j), cn_ij);
      }

    }
  });

  cn_dual.modify<DeviceType>();
  
  communicationStage = 1;
  if (newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);

}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
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

  // Push to device once
  Kokkos::deep_copy(k_r2r4, h_r2r4);
  Kokkos::deep_copy(k_rcov, h_rcov);

  // Fill and transfer the multi‑dimensional parameter tables
  read_r0ab(atomic_numbers.data(), ntypes); // now just a 2D table lookup
  read_c6ab(atomic_numbers.data(), ntypes); // symmetric, grid‑aware fill
}




template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void get_dC6( int iat, int jat, double cni, double cnj, 
                const view5d d_c6ab, const view_1d k_mxci,
                double l_autoev, double l_autoang, double l_K3, 
                double *res)
{
  double c6mem = -1.0e20;
  double r_save = 1.0e20;
  double num = 0.0, den = 0.0;
  double d_num_i = 0.0, d_num_j = 0.0;
  double d_den_i = 0.0, d_den_j = 0.0;

  for (int ci = 0; ci <= k_mxci(iat); ci++) 
  {
    for (int cj = 0; cj <= k_mxci(jat); cj++)
    {
      c6_ref = d_c6ab(iat,jat,ci,cj,0) * l_autoev * pow(l_autoang, 6);;
      if (c6_ref > 0.0)
      {
        cni_ref = d_c6ab(iat,jat,ci,cj,1);
        cnj_ref = d_c6ab(iat,jat,ci,cj,2);
  
        double r = (cni - cni_ref)*(cni - cni_ref) 
                 + (cnj - cnj_ref)*(cnj - cnj_ref); 
       
        if (r < r_save) 
        {
          r_save = r ;
          c6mem = c6_ref ;
        } 
        
        double expterm = exp(l_K3 * r); 
        num += c6_ref * expterm;
        den += expterm;
        
        expterm *= 2.0 * l_K3 ; 
       
        double term = expterm * (cni - cni_ref);
        d_num_i += c6_ref * term ;
        d_den_i += term;

        term = expterm * (cnj - cnj_ref);
        d_num_j += c6_ref * term ; 
        d_den_j += term ;      

      }
    }
  }
  
  if (den > 1.0E-99)
  {
    c6_res[0] = num / den ; 
    c6_res[1] = ((d_num_i * den) - (d_den_i * num)) / (den * den);
    c6_res[2] = ((d_num_j * den) - (d_den_j * num)) / (den * den);
  }  
  else
  {
    c6_res[0] = c6mem ;
    c6_res[1] = 0.0 ; 
    c6_res[2] = 0.0 ; 
  }
}

template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::compute_dC6(int iat, int jat, double cni, double cnj) 
{
  auto d_c6ab = this->c6ab_view; 
  auto d_mxci  = this->mxci_view; 
  auto d_cn = this->cn_dual.template view<DeviceType>();

  const double l_autoev = autoev ; 
  const double l_autoang = autoang ; 
  const double l_K3 = K3 ; 

  auto d_ilist      = neighKK->d_ilist;
  auto d_numneigh   = neighKK->d_numneigh;
  auto d_firstneigh = neighKK->d_firstneigh;
  auto d_type       = atomKK->k_type.view<DeviceType>();

  Kokkos::parallel_for("C6Compute", Kokkos::RangePolicy<DeviceType>(0, list->inum),
                       KOKKOS_LAMBDA(const int ii)
  {
    const int i     = d_ilist(ii);
    const int itype = d_type(i);
    const double cni = d_cn(i);

    const int jnum  = d_numneigh(i);
    const int* jlist = d_firstneigh(i);
    
    for (int jj = 0; jj < jnum; jj++)
    {
      int j = jlist[jj] & NEIGHMASK;
      const int jtype = d_type(j);
      const double cnj = d_cn(j);
 
      double results[3];
      // results[0] = C6, results[1] = dC6/dCNi, results[2] = dC6/dCNj
      d_get_dC6(itype, jtype, cni, cnj,
                  d_c6ab, d_mxci,
                  l_autoev, l_autoang, l_K3,
                  results);

    }



  }  

  

}

// Helper is as we sketched earlier:
// template<class DeviceType> KOKKOS_INLINE_FUNCTION
// void d_get_dC6(int iat, int jat, double cni, double cnj,
//                 const Kokkos::View<double*****,DeviceType>& c6ab_d,
//                 const Kokkos::View<int*,DeviceType>& mxci_d,
//                 double autoev_l, double autoang_l, double K3_l,
//                 double* res);

template<class DeviceType>
void PairDispersionD3Kokkos::compute_kokkos(int eflag, int vflag)
{
  // --- host-side setup and flags
  std::unordered_map<std::string,int> dampingMap = {{"original",1},{"zerom",2},{"bj",3},{"bjm",4}};
  const int dampingCode = dampingMap[damping_type];

  ev_init(eflag, vflag);

  // Stage 1: CN on device (your Kokkos calc_coordination_number)
  calc_coordination_number_kokkos<DeviceType>();

  // Device views
  auto d_x      = atomKK->k_x.template view<DeviceType>();     // double** [nall,3]
  auto d_f      = atomKK->k_f.template view<DeviceType>();     // double** [nall,3]
  auto d_typd   = atomKK->k_type.template view<DeviceType>();  // int*    [nall]
  auto d_cn     = cn_dual.template view<DeviceType>();         // double* [nall]
  auto d_dc6    = dc6_dual.template view<DeviceType>();        // double* [nall]

  auto d_cutsq  = cutsq_view;      // Kokkos::View<double**,DeviceType>
  auto d_r0ab   = r0ab_view;       // Kokkos::View<double**,DeviceType>
  auto d_r2r4   = r2r4_view;       // Kokkos::View<double*,DeviceType>
  auto d_special_lj = special_lj_view; // Kokkos::View<double*,DeviceType>

  auto d_c6ab   = c6ab_view;       // Kokkos::View<double*****,DeviceType>
  auto d_mxci   = mxci_view;       // Kokkos::View<int*,DeviceType>

  // Neighbor list (device)
  auto d_ilist      = neighKK->d_ilist;
  auto d_numneigh   = neighKK->d_numneigh;
  auto d_firstneigh = neighKK->d_firstneigh;

  // Constants captured locally
  const int    nlocal       = atom->nlocal;
  const int    newton_pair  = force->newton_pair;
  const double l_rs6        = rs6;
  const double l_rs8        = rs8;
  const double l_s6         = s6;
  const double l_s8         = s8;
  const double l_alpha      = alpha;
  const double l_a1         = a1;
  const double l_a2         = a2;
  const double l_autoang    = autoang;
  const double l_autoev     = autoev;
  const double l_K3         = K3;
  const double l_cn_thr     = cn_thr;

  // Zero dc6 on device (we’ll accumulate into it)
  Kokkos::deep_copy(d_dc6, 0.0);

  // Optional energy reduction
  double evdwl_sum = 0.0;

  // --- First kernel: dispersion forces and dc6 accumulation
  Kokkos::parallel_reduce(
    "pair_d3_stage1",
    Kokkos::RangePolicy<DeviceType>(0, list->inum),
    KOKKOS_LAMBDA(const int ii, double& l_evdwl) {

      const int i     = d_ilist(ii);
      const int itype = d_type(i);
      const double xi = d_x(i,0), yi = d_x(i,1), zi = d_x(i,2);
      const double cni = d_cn(i);

      const int jnum   = d_numneigh(i);
      const int* jlist = d_firstneigh(i);

      for (int jj = 0; jj < jnum; jj++) {
        int jcoded = jlist[jj];
        const double factor_lj = d_special_lj(sbmask(jcoded)); // device-safe sbmask()
        const int j = (jcoded & NEIGHMASK);
        const int jtype = d_type(j);

        const double dx = xi - d_x(j,0);
        const double dy = yi - d_x(j,1);
        const double dz = zi - d_x(j,2);
        const double rsq = dx*dx + dy*dy + dz*dz;

        if (rsq >= d_cutsq(itype,jtype)) continue;

        const double r     = sqrt(rsq);
        const double r2inv = 1.0/rsq;
        const double r4inv = r2inv*r2inv;
        const double r6inv = r4inv*r2inv;
        const double r8inv = r4inv*r4inv;
        const double r10inv= r8inv*r2inv;

        // C6 and derivatives wrt CN
        double c6res[3];
        d_get_dC6<DeviceType>(itype, jtype, cni, d_cn(j),
                              d_c6ab, d_mxci, l_autoev, l_autoang, l_K3, c6res);
        const double C6 = c6res[0];
        const double C8 = 3.0 * C6 * d_r2r4(itype) * d_r2r4(jtype) * l_autoang * l_autoang;

        const double alpha6 = l_alpha;
        const double alpha8 = l_alpha + 2.0;

        double t6=0.0, t8=0.0, damp6=0.0, damp8=0.0, e6=0.0, e8=0.0, fpair=0.0;

        switch (dampingCode) {
          case 1: { // original
            const double r0 = r / d_r0ab(itype,jtype);
            t6    = pow(l_rs6 / r0, alpha6);
            damp6 = 1.0 / (1.0 + 6.0 * t6);
            t8    = pow(l_rs8 / r0, alpha8);
            damp8 = 1.0 / (1.0 + 6.0 * t8);
            e6 = C6 * damp6 * r6inv;
            e8 = C8 * damp8 * r8inv;
            const double tmp6 = 6.0 * l_s6 * C6 * r8inv  * damp6;
            const double tmp8 = 8.0 * l_s8 * C8 * r10inv * damp8;
            const double f1 = -(tmp6 + tmp8);
            const double f2 = tmp6 * alpha6 * t6 * damp6
                            + 0.75 * tmp8 * alpha8 * t8 * damp8;
            fpair = (f1 + f2) * factor_lj;
          } break;

          case 2: { // zerom
            const double r0 = d_r0ab(itype,jtype);
            t6    = pow((r / (l_rs6 * r0)) + l_rs8 * r0, -alpha6);
            damp6 = 1.0 / (1.0 + 6.0 * t6);
            t8    = pow((r / r0) + l_rs8 * r0, -alpha8);
            damp8 = 1.0 / (1.0 + 6.0 * t8);
            e6 = C6 * damp6 * r6inv;
            e8 = C8 * damp8 * r8inv;
            const double tmp6 = 6.0 * l_s6 * C6 * r8inv  * damp6;
            const double tmp8 = 8.0 * l_s8 * C8 * r10inv * damp8;
            const double f1 = -(tmp6 + tmp8);
            const double fp26 = tmp6 * alpha6 * t6 * damp6 * r / (r + l_rs6 * l_rs8 * r0 * r0);
            const double fp28 = tmp8 * alpha8 * t8 * damp8 * r / (r + l_rs8 * r0 * r0);
            const double f2 = fp26 + 0.75 * fp28;
            fpair = (f1 + f2) * factor_lj;
          } break;

          case 3: // bj (Becke–Johnson)
          case 4: { // bjm
            const double r0 = sqrt(C8 / C6);
            const double r4 = rsq * rsq;
            const double r6 = r4 * rsq;
            const double r8 = r4 * r4;
            t6 = r6 + pow((l_a1 * r0 + l_a2), 6);
            t8 = r8 + pow((l_a1 * r0 + l_a2), 8);
            e6 = C6 / t6;
            e8 = C8 / t8;
            const double tmp6 = 6.0 * l_s6 * C6 * r4 / (t6 * t6);
            const double tmp8 = 8.0 * l_s8 * C8 * r6 / (t8 * t8);
            fpair = -(tmp6 + tmp8) * factor_lj;
          } break;
        }

        // Energy (negative sign as in your original)
        if (eflag) {
          const double evdwl = -(l_s6 * e6 + l_s8 * e8) * factor_lj;
          l_evdwl += evdwl;
        }

        // dc6 accumulation
        const double rest = (l_s6 * e6 + l_s8 * e8) / C6;
        Kokkos::atomic_add(&d_dc6(i), rest * c6res[1]);
        if (newton_pair || j < nlocal) {
          Kokkos::atomic_add(&d_dc6(j), rest * c6res[2]);
        }

        // Forces
        const double fx = dx * fpair;
        const double fy = dy * fpair;
        const double fz = dz * fpair;

        Kokkos::atomic_add(&d_f(i,0),  fx);
        Kokkos::atomic_add(&d_f(i,1),  fy);
        Kokkos::atomic_add(&d_f(i,2),  fz);
        if (newton_pair || j < nlocal) {
          Kokkos::atomic_add(&d_f(j,0), -fx);
          Kokkos::atomic_add(&d_f(j,1), -fy);
          Kokkos::atomic_add(&d_f(j,2), -fz);
        }

        if (evflag) {
          // equivalent to ev_tally(i,j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz)
          const double evdw_here = evdwl; // already computed
          const double vxx = delx * fpair;
          const double vyy = dely * fpair;
          const double vzz = delz * fpair;
          const double vxy = delx * fpair; // or appropriate shear terms
          const double vxz = delx * fpair;
          const double vyz = dely * fpair;
        
          EV_ATOM(i).evdwl += evdw_here;
          EV_ATOM(i).v[0]  += vxx;
          EV_ATOM(i).v[1]  += vyy;
          EV_ATOM(i).v[2]  += vzz;
          EV_ATOM(i).v[3]  += vxy;
          EV_ATOM(i).v[4]  += vxz;
          EV_ATOM(i).v[5]  += vyz;
        
          if (newton_pair || j < nlocal) {
            EV_ATOM(j).evdwl += evdw_here;
            EV_ATOM(j).v[0]  += vxx;
            EV_ATOM(j).v[1]  += vyy; 
            EV_ATOM(j).v[2]  += vzz; 
            EV_ATOM(j).v[3]  += vxy;
            EV_ATOM(j).v[4]  += vxz;
            EV_ATOM(j).v[5]  += vxy;
          }
        }

// ev_tally omitted here; use Kokkos-compatible tally helpers if needed
      }
    },
    evdwl_sum);

  // Accumulate energy on host if requested
  if (eflag) {
    eng_vdwl += evdwl_sum;
  }

  // --- Comm Stage 2: synchronize dc6 owner/ghosts
  dc6_dual.modify<DeviceType>();           // device updated
  dc6_dual.sync<HostSpace>();              // host sees latest for MPI
  communicationStage = 2;
  if (newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);
  dc6_dual.modify<HostSpace>();            // host just wrote ghosts
  dc6_dual.sync<DeviceType>();             // push back to device

  // --- Second kernel: CN-derivative forces (uses dc6)
  Kokkos::parallel_for(
    "pair_d3_stage2",
    Kokkos::RangePolicy<DeviceType>(0, list->inum),
    KOKKOS_LAMBDA(const int ii) {

      const int i     = d_ilist(ii);
      const int itype = d_type(i);
      const double xi = d_x(i,0), yi = d_x(i,1), zi = d_x(i,2);

      const int jnum   = d_numneigh(i);
      const int* jlist = d_firstneigh(i);

      for (int jj = 0; jj < jnum; jj++) {
        int jcoded = jlist[jj];
        const double factor_lj = d_special_lj(sbmask(jcoded));
        const int j = (jcoded & NEIGHMASK);
        const int jtype = d_type(j);

        const double dx = xi - d_x(j,0);
        const double dy = yi - d_x(j,1);
        const double dz = zi - d_x(j,2);
        const double rsq = dx*dx + dy*dy + dz*dz;

        if (rsq >= d_cutsq(itype,jtype)) continue;

        double dcn = 0.0;
        if (rsq < l_cn_thr) {
          const double r     = sqrt(rsq);
          const double rcov  = (d_rcov(itype) + d_rcov(jtype)) * l_autoang; // d_rcov: View<double*,DeviceType>
          const double expt  = exp(-K1 * (rcov / r - 1.0));
          dcn = -K1 * rcov * expt / (rsq * (expt + 1.0) * (expt + 1.0));
        }

        const double r     = sqrt(rsq);
        const double fpair = factor_lj * dcn * (d_dc6(i) + d_dc6(j)) / r;

        const double fx = dx * fpair;
        const double fy = dy * fpair;
        const double fz = dz * fpair;

        Kokkos::atomic_add(&d_f(i,0),  fx);
        Kokkos::atomic_add(&d_f(i,1),  fy);
        Kokkos::atomic_add(&d_f(i,2),  fz);
        if (newton_pair || j < nlocal) {
          Kokkos::atomic_add(&d_f(j,0), -fx);
          Kokkos::atomic_add(&d_f(j,1), -fy);
          Kokkos::atomic_add(&d_f(j,2), -fz);
        }

        // ev_tally for virial from this stage can be added similarly if needed
        if (evflag) {
          // equivalent to ev_tally(i,j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz)
          const double evdw_here = evdwl; // already computed
          const double vxx = delx * fpair;
          const double vyy = dely * fpair;
          const double vzz = delz * fpair;
          const double vxy = delx * fpair; // or appropriate shear terms
          const double vxz = delx * fpair;
          const double vyz = dely * fpair;
        
          EV_ATOM(i).v[0]  += vxx;
          EV_ATOM(i).v[1]  += vyy;
          EV_ATOM(i).v[2]  += vzz;
          EV_ATOM(i).v[3]  += vxy;
          EV_ATOM(i).v[4]  += vxz;
          EV_ATOM(i).v[5]  += vyz;
        
          if (newton_pair || j < nlocal) {
            EV_ATOM(j).v[0]  += vxx;
            EV_ATOM(j).v[1]  += vyy; 
            EV_ATOM(j).v[2]  += vzz; 
            EV_ATOM(j).v[3]  += vxy;
            EV_ATOM(j).v[4]  += vxz;
            EV_ATOM(j).v[5]  += vxy;
          }
        }
      }
    });

  if (vflag_fdotr) virial_fdotr_compute(); // host-side utility; ok to call after kernels
}


template<class DeviceType>
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::set_funcpar(std:string &functional_name)
{
// TODO UGH import logic as is - doesn't need to be gpu-coded 
}


template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::init_one(int i, int j)
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
void PairDispersionD3Kokkos<DeviceType>::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style D3 requires atom IDs");

  // If you later re‑enable this, uncomment to enforce Newton pair requirement
  // if (force->newton_pair == 0)
  //   error->all(FLERR, "Pair style D3 requires newton pair on");

  // Request a half neighbor list
  neighbor->add_request(this);
}


template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_forward_comm(int n, int *list,
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
void PairDispersionD3Kokkos<DeviceType>::unpack_forward_comm(int n, int first,
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
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm(int n, int first,
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
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm(int n, int *list,
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
void PairDispersionD3Kokkos<DeviceType>::set_funcpar()
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


