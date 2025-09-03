#include "pair_dispersion_d3_kokkos.h"
#include "pair_dispersion_d3.h"
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
#ifdef D3_KK_DEBUG
#include <cstdio>
#endif


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
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp) : PairDispersionD3(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  //datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  nmax = 0; 
}

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::~PairDispersionD3Kokkos()
{
  if (copymode) return;
  
  // Clean up Kokkos allocations
  if (allocated) {
    memoryKK->destroy_kokkos(k_cn_v, cn);
    memoryKK->destroy_kokkos(k_dc6_v, dc6);
    
    if (eflag_atom) memoryKK->destroy_kokkos(k_eatom, eatom);
    if (vflag_atom) memoryKK->destroy_kokkos(k_vatom, vatom);
  }
  
  // Set our managed pointers to nullptr - others should already be nullptr from coeff()
  cn = nullptr;
  dc6 = nullptr;
  eatom = nullptr;
  vatom = nullptr;
  allocated = 0;
}


template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::init_one(int i, int j)
{
  // Let base class do mixing / set r0ab, cutsq, etc.
  double cut = PairDispersionD3::init_one(i,j);

  // If DualViews not yet allocated (e.g. restart w/ different ntypes), allocate here.
  int ntypes = atom->ntypes;
  if (!k_r0ab_v.span()) {
    k_mxci_v  = DAT::tdual_float_1d("k_mxci", ntypes+1);
    k_r2r4_v  = DAT::tdual_float_1d("k_r2r4", ntypes+1);
    k_rcov_v  = DAT::tdual_float_1d("k_rcov", ntypes+1);
    k_r0ab_v  = DAT::tdual_float_2d("k_r0ab", ntypes+1, ntypes+1);
    k_cutsq_v = DAT::tdual_float_2d("k_cutsq", ntypes+1, ntypes+1);
    k_c6ab_v  = tdual_float_5d("k_c6ab", ntypes+1, ntypes+1, 5,5,3);

    // Fill 1D (type) arrays once
    for (int t=0; t<=ntypes; ++t) {
      k_mxci_v.h_view(t)  = mxci[t];
      k_r2r4_v.h_view(t)  = r2r4[t];
      k_rcov_v.h_view(t)  = rcov[t];
    }
  }

  // Copy symmetric pair entries that base just finalized
  k_r0ab_v.h_view(i,j)  = r0ab[i][j];
  k_r0ab_v.h_view(j,i)  = r0ab[j][i];  // usually same, but keep symmetric
  
  // IMPORTANT: cutsq not yet set in base arrays; derive from returned cut
  const float cutsq_local = static_cast<float>(cut * cut);
  k_cutsq_v.h_view(i,j) = cutsq_local;
  k_cutsq_v.h_view(j,i) = cutsq_local;

  // C6 tensor (only once per (i,j); ensure symmetry if needed)
  for (int gi=0; gi<5; ++gi)
    for (int gj=0; gj<5; ++gj)
      for (int k=0; k<3; ++k) {
        k_c6ab_v.h_view(i,j,gi,gj,k) = c6ab[i][j][gi][gj][k];
        k_c6ab_v.h_view(j,i,gj,gi,k) = c6ab[j][i][gj][gi][k]; // keep transpose consistent
      }

  // If this is the final init_one call, push all to device & bind device views
  if (i == atom->ntypes && j == atom->ntypes) {
    k_mxci_v.modify_host();  k_mxci_v.template sync<DeviceType>();
    k_r2r4_v.modify_host();  k_r2r4_v.template sync<DeviceType>();
    k_rcov_v.modify_host();  k_rcov_v.template sync<DeviceType>();
    k_r0ab_v.modify_host();  k_r0ab_v.template sync<DeviceType>();
    k_cutsq_v.modify_host(); k_cutsq_v.template sync<DeviceType>();
    k_c6ab_v.modify_host();  k_c6ab_v.template sync<DeviceType>();

    d_mxci_v  = k_mxci_v.template view<DeviceType>();
    d_r2r4_v  = k_r2r4_v.template view<DeviceType>();
    d_rcov_v  = k_rcov_v.template view<DeviceType>();
    d_r0ab_v  = k_r0ab_v.template view<DeviceType>();
    d_cutsq_v = k_cutsq_v.template view<DeviceType>();
    d_c6ab_v  = k_c6ab_v.template view<DeviceType>();

 
#ifdef DEBUG_D3_CUTSQ
    if (!d_cutsq_v.data()) Kokkos::abort("d_cutsq_v not bound after final init_one");
#endif
  }
  return cut;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
 
  PairDispersionD3::coeff(narg,arg);
  
  //free all baseclass



}


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_style()
{
  // Initialize the base class first
  PairDispersionD3::init_style();

  // adjust neighbor list request for KOKKOS
  //int neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  //if (neighflag == FULL) request->enable_full();
  //neighbor->add_request(this,NeighConst::REQ_DEFAULT);
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_numbersKK()
{
  
  /*if (lmp->comm->me == 0) {
    printf("DEBUG: inum=%d, nlocal=%d, nall=%d, nmax=%d\n", inum, nlocal, nall, nmax);
    printf("DEBUG: d_cn_v.extent(0)=%zu, d_dc6_v.extent(0)=%zu\n", 
           d_cn_v.extent(0), d_dc6_v.extent(0));
  }*/
  atomKK->sync(execution_space,datamask_read);
  d_x           = atomKK->k_x.view<DeviceType>();
  d_f           = atomKK->k_f.view<DeviceType>();
  d_type        = atomKK->k_type.view<DeviceType>();
  nlocal      = atomKK->nlocal;
  nall        = atomKK->nlocal + atomKK->nghost;
  newton_pair = force->newton_pair; 
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;
 
  
  /*
  // Add debug output
  if (lmp->comm->me == 0) {
    printf("DEBUG: BEFORE inum=%d, nlocal=%d, nall=%d, nmax=%d\n", inum, nlocal, nall, nmax);
  }*/
   
 
  if (atomKK->nmax > nmax)
  {
    nmax = atomKK->nmax;
    memoryKK->grow_kokkos(k_cn_v, cn, nmax, "pair:cn");
    memoryKK->grow_kokkos(k_dc6_v, dc6, nmax, "pair:dc6");
    //allocated = 1; 
  }

  d_cn_v  = k_cn_v.template view<DeviceType>();
  d_dc6_v = k_dc6_v.template view<DeviceType>();
  d_mxci_v = k_mxci_v.template view<DeviceType>();
  d_r2r4_v = k_r2r4_v.template view<DeviceType>();
  d_rcov_v = k_rcov_v.template view<DeviceType>();
  d_r0ab_v = k_r0ab_v.template view<DeviceType>();
  d_c6ab_v = k_c6ab_v.template view<DeviceType>();
  //d_cutsq_v = k_cutsq_v.template view<DeviceType>(); 
  /* 
  // Add debug output after views are set
  if (lmp->comm->me == 0) {
    printf("DEBUG: after views  d_cn_v.extent(0)=%zu, d_dc6_v.extent(0)=%zu\n", 
           d_cn_v.extent(0), d_dc6_v.extent(0));
  }

  // Add early return if no atoms to process
  if (inum == 0 || nlocal == 0) {
    if (lmp->comm->me == 0) {
      printf("DEBUG: Early return - inum=%d, nlocal=%d\n", inum, nlocal);
    }
    return;
  }*/
  copymode = 1;
  // Zero out dc6 and cn
  if (newton_pair)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3KokkosCNDC6Initialise>(0, nall), *this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3KokkosCNDC6Initialise>(0, nlocal), *this);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3KokkosCNDC6Calc>(0, inum), *this);  
 
  k_cn_v.template modify<DeviceType>();
  
  communicationStage = 1;
  if (newton_pair) {
    k_cn_v.template modify<DeviceType>();
    //k_dc6_v.template modify<DeviceType>();
    comm->reverse_comm(this);
    k_cn_v.template sync<DeviceType>();
    //k_dc6_v.template sync<DeviceType>();
  }  
  k_cn_v.template modify<DeviceType>();
  //k_dc6_v.template modify<DeviceType>();
  comm->forward_comm(this);
  k_cn_v.template sync<DeviceType>();
  //k_dc6_v.template sync<DeviceType>();
  copymode = 0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3KokkosCNDC6Initialise, const int &i) const
{
  d_cn_v(i)  = 0.0;
  d_dc6_v(i) = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3KokkosCNDC6Calc, const int &ii) const
{
  if (ii >= inum) return;    
  const int     i      = d_ilist[ii];
  const int     itype  = d_type(i);
  const int     jnum   = d_numneigh[i];
  const F_FLOAT xi     = d_x(i,0);
  const F_FLOAT yi     = d_x(i,1);
  const F_FLOAT zi     = d_x(i,2);

  for (int jj = 0; jj < jnum; jj++)
  {
    int j                = d_neighbors(i, jj) & NEIGHMASK; 
    const int     jtype  = d_type(j); 
    const F_FLOAT xj     = d_x(j,0);
    const F_FLOAT yj     = d_x(j,1);
    const F_FLOAT zj     = d_x(j,2);
    
    const F_FLOAT xij    = xi - xj;
    const F_FLOAT yij    = yi - yj;
    const F_FLOAT zij    = zi - zj;

    const F_FLOAT rsq    = xij*xij + yij*yij + zij*zij; 

    // if the atoms are too far away don't consider the contribution
    if (rsq > cn_thr) continue; 

    const F_FLOAT rr      = sqrt(rsq);
    const F_FLOAT rcov_ij = (d_rcov_v(itype) + d_rcov_v(jtype)) * autoang;
    const F_FLOAT cn_ij   = 1.0f / (1.0f + expf(-K1 * ((rcov_ij / rr) - 1.0f)));

    d_cn_v(i) += cn_ij;
    if (newton_pair || j < nlocal) { Kokkos::atomic_add(&d_cn_v(j), cn_ij); }
  }
}

struct DC6Derive
{
  double num, den, dnum_i, dnum_j, dden_i, dden_j;
  double c6mem, r_save;

  KOKKOS_INLINE_FUNCTION
  static void init(DC6Derive &acc)
  {
    acc.num    = 0.0;
    acc.den    = 0.0;
    acc.dnum_i = 0.0;
    acc.dnum_j = 0.0;
    acc.dden_i = 0.0;
    acc.dden_j = 0.0;
    acc.c6mem  = -1.0e20;
    acc.r_save =  1.0e20; 
  }
  
  KOKKOS_INLINE_FUNCTION
  static void join(DC6Derive &dst, const DC6Derive &src)
  {
    dst.num    +=  src.num;
    dst.den    +=  src.den;
    dst.dnum_i +=  src.dnum_i;
    dst.dnum_j +=  src.dnum_j;
    dst.dden_i +=  src.dden_i;
    dst.dden_j +=  src.dden_j;
  
    if (src.r_save < dst.r_save) 
    {
      dst.r_save = src.r_save;
      dst.c6mem = src.c6mem;
    }
  }

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  static void accumulate_cell( const ViewType &d_c6ab_v,
                               int iat, int jat, int ci, int cj, 
                               double cni, double cnj, DC6Derive &acc)
  {
    // Add bounds checking
    if (iat >= d_c6ab_v.extent(0) || jat >= d_c6ab_v.extent(1) ||
        ci >= d_c6ab_v.extent(2) || cj >= d_c6ab_v.extent(3)) {
      return;
    }
    
    double c6_ref = d_c6ab_v(iat, jat, ci, cj, 0);
    c6_ref *= autoev * pow(autoang, 6);
    if (c6_ref <= 0.0) return; 
   
    const double cni_ref  = d_c6ab_v(iat, jat, ci, cj, 1);
    const double cnj_ref  = d_c6ab_v(iat, jat, ci, cj, 2);
  
    const double    dx_i  = cni - cni_ref;
    const double    dx_j  = cnj - cnj_ref; 

    const double       r  = dx_i*dx_i  + dx_j*dx_j;
  
    if ( r < acc.r_save) { acc.r_save = r; acc.c6mem = c6_ref; }
  
    double      expterm   = exp(K3*r);
    acc.num              += c6_ref * expterm;
    acc.den              += expterm;

    expterm              *= 2.0 * K3;
    const double  term_i  = expterm * dx_i;
    const double  term_j  = expterm * dx_j;
    acc.dnum_i           += c6_ref * term_i;
    acc.dden_i           += term_i;
    acc.dnum_j           += c6_ref * term_j;
    acc.dden_j           += term_j;
  }
};



KOKKOS_INLINE_FUNCTION
void operator+=(DC6Derive &lhs, const DC6Derive &rhs)
{
  DC6Derive::join(lhs, rhs);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::get_dC6KK
(
  const    int iat, const    int jat,
  const double cni, const double cnj,
  double &C6, double &dC6_dCNi, double &dC6_dCNj
) const
{
  const int Ci = d_mxci_v(iat) + 1;
  const int Cj = d_mxci_v(jat) + 1;

  DC6Derive acc;
  DC6Derive::init(acc);

  if (iat >= d_c6ab_v.extent(0) || jat >= d_c6ab_v.extent(1)) {
    C6 = 0.0; dC6_dCNi = 0.0; dC6_dCNj = 0.0;
    return;
  }

  for (int ci = 0; ci < Ci; ++ci) {
    if (ci >= d_c6ab_v.extent(2)) break;
    for (int cj = 0; cj < Cj; ++cj) {
      if (cj >= d_c6ab_v.extent(3)) break;
      DC6Derive::accumulate_cell(d_c6ab_v, iat, jat, ci, cj, cni, cnj, acc);
    }
  }

  if (acc.den > 1.0e-99) {
    C6       = acc.num / acc.den;
    dC6_dCNi = ((acc.dnum_i * acc.den) - (acc.dden_i * acc.num)) / (acc.den * acc.den);
    dC6_dCNj = ((acc.dnum_j * acc.den) - (acc.dden_j * acc.num)) / (acc.den * acc.den);
  } else {
    C6 = acc.c6mem;
    dC6_dCNi = 0.0;
    dC6_dCNj = 0.0;
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  #ifdef D3_KK_DEBUG
  static long long last_step = -1;
  static int calls_this_step = 0;
  if (update->ntimestep != last_step) { last_step = update->ntimestep; calls_this_step = 0; }
  ++calls_this_step;
  #endif


  using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  using TeamMember = typename TeamPolicy::member_type;

  int eflag = eflag_in;
  int vflag = vflag_in;
  
  if (atomKK->nmax > nmax) {
    nmax = atomKK->nmax;
    memoryKK->grow_kokkos(k_cn_v,  cn,  nmax, "pair:cn");
    memoryKK->grow_kokkos(k_dc6_v, dc6, nmax, "pair:dc6");
    if (eflag_atom) {
      memoryKK->destroy_kokkos(k_eatom, eatom);
      memoryKK->create_kokkos(k_eatom, eatom, nmax, "pair:eatom");
      d_eatom = k_eatom.view<DeviceType>();
    }
    if (vflag_atom) {
      memoryKK->destroy_kokkos(k_vatom, vatom);
      memoryKK->create_kokkos(k_vatom, vatom, nmax, "pair:vatom");
      d_vatom = k_vatom.view<DeviceType>();
    }
  }


  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  ev_init(eflag, vflag); 

  #ifdef D3_KK_DEBUG
  atomKK->sync(Host, F_MASK);
  double pre_sum = 0.0;
  for (int i=0;i<atomKK->nlocal;i++) {
    pre_sum += fabs(atom->f[i][0]) + fabs(atom->f[i][1]) + fabs(atom->f[i][2]);
    if (i < 2) {
      printf("D3KK DEBUG pre i=%d f=(%.6e %.6e %.6e) step=%lld call=%d\n",
             i, atom->f[i][0], atom->f[i][1], atom->f[i][2],
             (long long)update->ntimestep, calls_this_step);
    }
  }
  printf("D3KK DEBUG pre Stage0 sumF=%.6e step=%lld call=%d\n",
         pre_sum, (long long)update->ntimestep, calls_this_step);
  #endif

 
  calc_coordination_numbersKK();
 
  // Early return if no atoms to process
  if (list->inum == 0) {
    return;
  } 
  atomKK->sync(execution_space, X_MASK | TAG_MASK | TYPE_MASK); 
  atomKK->modified(execution_space, F_MASK); 
  if (eflag) atomKK->modified(execution_space, ENERGY_MASK | VIRIAL_MASK);
  if (vflag) atomKK->modified(execution_space, VIRIAL_MASK);


  


  auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh   = k_list->d_numneigh;
  d_neighbors  = k_list->d_neighbors;
  d_ilist      = k_list->d_ilist;
  inum         = list->inum;

  d_x    = atomKK->k_x.view<DeviceType>();
  d_f    = atomKK->k_f.view<DeviceType>();
  d_type = atomKK->k_type.view<DeviceType>();



  // clear dc6
  Kokkos::deep_copy(d_dc6_v, 0.0);
  
  d_cutsq_v = k_cutsq_v.template view<DeviceType>();
  
  #ifdef D3_KK_DEBUG
  {
    double h_min_cutsq=1e300, h_max_cutsq=-1e300;
    auto d_cutsq = d_cutsq_v;
    Kokkos::parallel_reduce("cutsq_min",
      Kokkos::RangePolicy<DeviceType>(1, atom->ntypes+1),
      KOKKOS_LAMBDA(const int i, double &loc_min){
        for (int j=1;j<=d_cutsq.extent(1)-1;j++){
          double v = d_cutsq(i,j);
          if (v>0.0 && v < loc_min) loc_min = v;
        }
      }, Kokkos::Min<double>(h_min_cutsq));
    Kokkos::parallel_reduce("cutsq_max",
      Kokkos::RangePolicy<DeviceType>(1, atom->ntypes+1),
      KOKKOS_LAMBDA(const int i, double &loc_max){
        for (int j=1;j<=d_cutsq.extent(1)-1;j++){
          double v = d_cutsq(i,j);
            if (v > loc_max) loc_max = v;
        }
      }, Kokkos::Max<double>(h_max_cutsq));
    Kokkos::fence();
    if (lmp->comm->me==0) {
      printf("D3KK DEBUG params s6=%.6g s8=%.6g alpha=%.6g rs6=%.6g rs8=%.6g cutsq[min,max]=[%.6g %.6g] step=%lld\n",
             s6,s8,alpha,rs6,rs8,h_min_cutsq,h_max_cutsq,(long long)update->ntimestep);
    }
  }
  #endif
 
  copymode=1;
  // -------------------------
  // Stage 1: dE/d(ij) + dc6
  // -------------------------
  /*#ifdef D3_KK_DEBUG
  long long paircount = 0;
  {
    auto d_x_local = d_x;
    auto d_type_local = d_type;
    auto d_numneigh_local = d_numneigh;
    auto d_neighbors_local = d_neighbors;
    auto d_cutsq_local = d_cutsq_v;
    Kokkos::parallel_reduce("stage1_paircount",
      Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii, long long &acc){
        int i = d_ilist[ii];
        int itype = d_type_local(i);
        int jn = d_numneigh_local[i];
        for (int jj=0;jj<jn;jj++){
          int jfull = d_neighbors_local(i,jj);
          int j = jfull & NEIGHMASK;
          int jtype = d_type_local(j);
          double dx = d_x_local(i,0)-d_x_local(j,0);
          double dy = d_x_local(i,1)-d_x_local(j,1);
            double dz = d_x_local(i,2)-d_x_local(j,2);
          double rsq = dx*dx+dy*dy+dz*dz;
          if (rsq < d_cutsq_local(itype,jtype)) acc++;
        }
      }, paircount);
  }
  Kokkos::fence();
  #endif */ 
  

  #ifdef D3_KK_DEBUG
  Kokkos::fence();
  double dc6_pre = 0.0;
  {
    auto d_dc6_v_local = d_dc6_v;   // copy view into captureable variable
    Kokkos::parallel_reduce(
      "dc6_pre",
      Kokkos::RangePolicy<DeviceType>(0, atomKK->nlocal),
      KOKKOS_LAMBDA(const int i, double &acc) {
        acc += fabs(d_dc6_v_local(i));
      },
      dc6_pre
    );
    Kokkos::fence();
  }
  printf("D3KK DEBUG before Stage1 dc6_L1=%.6e step=%lld call=%d\n",
         dc6_pre, (long long)update->ntimestep, calls_this_step);
  #endif  


  
      // Stage 1: only HALF + NEWTON=1 instantiations needed now
   if (evflag) {
     EV_FLOAT ev;
     Kokkos::parallel_reduce(
       Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdIJ<HALF,1,1>>(0, inum),
       *this, ev);
     if (eflag_global) eng_vdwl += ev.evdwl;
     if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
   } else {
     Kokkos::parallel_for(
       Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdIJ<HALF,1,0>>(0, inum),
       *this);
   }
  
  #ifdef D3_KK_DEBUG
  atomKK->sync(Host, F_MASK);
  double post1_sum=0.0;
  for (int i=0;i<atomKK->nlocal;i++) {
    post1_sum += std::fabs(atom->f[i][0]) + std::fabs(atom->f[i][1]) + std::fabs(atom->f[i][2]);
  }
  printf("D3KK DEBUG after Stage1 sumF=%.6e step=%lld call=%d\n",
         post1_sum, (long long)update->ntimestep, calls_this_step);
  #endif
   

  
   atomKK->modified(execution_space, F_MASK); 
   //Kokkos::fence();
   // Inter-stage comm: reverse then forward for dc6
   communicationStage = 2;
   k_dc6_v.template modify<DeviceType>();
   comm->reverse_comm(this);
   k_dc6_v.template sync<DeviceType>();
 
   k_dc6_v.template modify<DeviceType>();
   comm->forward_comm(this);
   k_dc6_v.template sync<DeviceType>();
   
   atomKK->sync(execution_space, F_MASK);

   #ifdef D3_KK_DEBUG
   Kokkos::fence();
   double dc6_post = 0.0;
   {
     auto d_dc6_v_local = d_dc6_v;
     Kokkos::parallel_reduce(
       "dc6_post",
       Kokkos::RangePolicy<DeviceType>(0, atomKK->nlocal),
       KOKKOS_LAMBDA(const int i, double &acc) {
         acc += fabs(d_dc6_v_local(i));
       },
       dc6_post
     );
     Kokkos::fence();
   }
   printf("D3KK DEBUG after dc6 comm L1=%.6e step=%lld call=%d\n",
          dc6_post, (long long)update->ntimestep, calls_this_step);
   #endif

    
    // Stage 2
   if (vflag) {
     EV_FLOAT ev;
     Kokkos::parallel_reduce(
       Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF,1,1>>(0, inum),
       *this, ev);
     //if (eflag_global) eng_vdwl += ev.evdwl;
     //if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
   } else {
     Kokkos::parallel_for(
       Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF,1,0>>(0, inum),
       *this);
   }
   #ifdef D3_KK_DEBUG
   atomKK->sync(Host, F_MASK);
   double post2_sum=0.0;
   for (int i=0;i<atomKK->nlocal;i++) {
     post2_sum += std::fabs(atom->f[i][0]) + std::fabs(atom->f[i][1]) + std::fabs(atom->f[i][2]);
   }
   printf("D3KK DEBUG after Stage2 sumF=%.6e step=%lld call=%d\n",
          post2_sum, (long long)update->ntimestep, calls_this_step);
   #endif    

/*

   if (evflag) {
    EV_FLOAT ev;
    if (newton_pair) {
      Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<DeviceType, TagPairDispDD3dEdIJ<HALF,1,1>>(inum, Kokkos::AUTO()),
        *this, ev);
    } else {
      Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<DeviceType, TagPairDispDD3dEdIJ<HALF,0,1>>(inum, Kokkos::AUTO()),
        *this, ev);
    }
    if (eflag_global) eng_vdwl += ev.evdwl;
    if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
  } else {
    if (newton_pair) {
      Kokkos::parallel_for(
        Kokkos::TeamPolicy<DeviceType, TagPairDispDD3dEdIJ<HALF,1,0>>(inum, Kokkos::AUTO()),
        *this);
    } else {
      Kokkos::parallel_for(
        Kokkos::TeamPolicy<DeviceType, TagPairDispDD3dEdIJ<HALF,0,0>>(inum, Kokkos::AUTO()),
        *this);
    }
  }
  // -------------------------
  // Comms between stages
  // -------------------------
  communicationStage = 2;
  if (newton_pair) {
    //k_cn_v.template modify<DeviceType>();
    k_dc6_v.template modify<DeviceType>();
    comm->reverse_comm(this);
    //k_cn_v.template sync<DeviceType>();
    k_dc6_v.template sync<DeviceType>();
  }
  // Always forward (match base code)
  //k_cn_v.template modify<DeviceType>();
  k_dc6_v.template modify<DeviceType>();
  comm->forward_comm(this);
  //k_cn_v.template sync<DeviceType>();
  k_dc6_v.template sync<DeviceType>();

  // -------------------------
  // Stage 2: dE/dxyz from dc6
  // -------------------------
  if (evflag) {
    EV_FLOAT ev;
    if (newton_pair) {
      Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF,1,1>>(0, inum), 
        *this, ev);
    } else {
      Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF,0,1>>(0, inum),
        *this, ev);
    }
    if (eflag_global) eng_vdwl += ev.evdwl;
    if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
  } else {
    if (newton_pair) {
      Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF,1,0>>(0, inum),
        *this);
    } else {
      Kokkos::parallel_for(
        Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF,0,0>>(0,inum), 
        *this);
    }
  }*/
  // forces modified again
  atomKK->modified(execution_space, datamask_modify);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  if (vflag_fdotr) pair_virial_fdotr_compute(this);
  //if (vflag_fdotr) virial_fdotr_compute();
  copymode = 0;
  if constexpr (!std::is_same_v<DeviceType, LMPHostType>) {
    atomKK->sync(Host, F_MASK);
  }
  
}

// EV version (RangePolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT &ev) const
{
  if (ii >= inum) return;
  const int i = d_ilist[ii];
  if (i >= nlocal) return;
  const int itype = d_type(i);
  if (itype <= 0 || itype >= d_mxci_v.extent(0)) return;

  const double xi = d_x(i,0);
  const double yi = d_x(i,1);
  const double zi = d_x(i,2);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) {
    const int jfull = d_neighbors(i,jj);
    const double factor_lj = special_lj[sbmask_disp(jfull)];
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);

    const double dx  = xi - d_x(j,0);
    const double dy  = yi - d_x(j,1);
    const double dz  = zi - d_x(j,2);
    const double rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= d_cutsq_v(itype,jtype)) continue;

    const double r      = sqrt(rsq);
    const double r2inv  = 1.0 / rsq;
    const double r4inv  = r2inv * r2inv;
    const double r6inv  = r4inv * r2inv;
    const double r8inv  = r6inv * r2inv;
    const double r10inv = r8inv * r2inv;

    const double cni = d_cn_v(i);
    const double cnj = d_cn_v(j);

    double C6=0.0, dC6_i=0.0, dC6_j=0.0;
    get_dC6KK(itype, jtype, cni, cnj, C6, dC6_i, dC6_j);
    if (C6 == 0.0) continue;

    const double C8 = 3.0 * C6 * d_r2r4_v(itype) * d_r2r4_v(jtype) * (autoang * autoang);

    const double r0     = r / d_r0ab_v(itype, jtype);
    const double alpha6 = alpha;
    const double alpha8 = alpha + 2.0;

    const double t6    = pow(rs6 / r0, alpha6);
    const double damp6 = 1.0 / (1.0 + 6.0 * t6);
    const double t8    = pow(rs8 / r0, alpha8);
    const double damp8 = 1.0 / (1.0 + 6.0 * t8);

    const double e6 = C6 * damp6 * r6inv;
    const double e8 = C8 * damp8 * r8inv;

    const double tmp6 = 6.0 * s6 * C6 * r8inv  * damp6;
    const double tmp8 = 8.0 * s8 * C8 * r10inv * damp8;

    const double fpair_no_damp = -(tmp6 + tmp8);
    const double fpair_damp    =  (tmp6 * alpha6 * t6 * damp6)
                                + (tmp8 * alpha8 * t8 * damp8 * 0.75);
    const double fpair = (fpair_no_damp + fpair_damp) * factor_lj;

    const double phi = -(s6 * e6 + s8 * e8) * factor_lj;

    const double rest = (s6 * e6 + s8 * e8) / C6;
    Kokkos::atomic_add(&d_dc6_v(i), rest * dC6_i);
    if (NEWTON_PAIR || j < nlocal)
      Kokkos::atomic_add(&d_dc6_v(j), rest * dC6_j);

    fix += dx * fpair;
    fiy += dy * fpair;
    fiz += dz * fpair;

    if (NEWTON_PAIR || j < nlocal) {
      Kokkos::atomic_add(&d_f(j,0), -(dx * fpair));
      Kokkos::atomic_add(&d_f(j,1), -(dy * fpair));
      Kokkos::atomic_add(&d_f(j,2), -(dz * fpair));
    }

    if (EVFLAG) this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, phi, fpair, dx, dy, dz);
  }

  Kokkos::atomic_add(&d_f(i,0), fix);
  Kokkos::atomic_add(&d_f(i,1), fiy);
  Kokkos::atomic_add(&d_f(i,2), fiz);
}

// NO-EV version (RangePolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  operator()(TagPairDispDD3dEdIJ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}

// Communication operators
template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackForwardCommCN, const int &i) const
{
  int j = d_sendlistV(i);
  bufV(i) = d_cn_v(j);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackForwardCommDC6, const int &i) const
{
  int j = d_sendlistV(i);
  bufV(i) = d_dc6_v(j);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackForwardCommCN, const int &i) const
{
  d_cn_v(i + first) = bufV(i);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackForwardCommDC6, const int &i) const
{
  d_dc6_v(i + first) = bufV(i);
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_forward_comm_kokkos(int n,
                                                                 DAT::tdual_int_1d k_sendlistV,
                                                                 DAT::tdual_xfloat_1d &buf,
                                                                 int /*pbc_flag*/,
                                                                 int * /*pbc*/)
{
  d_sendlistV = k_sendlistV.view<DeviceType>();
  bufV        = buf.view<DeviceType>();

  if (communicationStage == 1)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackForwardCommCN>(0,n), *this);
    k_cn_v.template modify<DeviceType>();
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackForwardCommDC6>(0,n), *this);
    k_dc6_v.template modify<DeviceType>();
  }
  return n;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_forward_comm_kokkos(int n,
                                                                    int first_in,
                                                                    DAT::tdual_xfloat_1d &buf)
{
  first = first_in;
  bufV = buf.view<DeviceType>();

  if (communicationStage == 1)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackForwardCommCN>(0,n), *this);
    k_cn_v.template modify<DeviceType>();
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackForwardCommDC6>(0,n), *this);
    k_dc6_v.template modify<DeviceType>(); 
  }
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm_kokkos(int n,
                                                                 int first_in,
                                                                 DAT::tdual_xfloat_1d &buf)
{
  bufV = buf.view<DeviceType>();
  first = first_in;

  if (communicationStage == 1)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackReverseCommCN>(0,n), *this);
    k_cn_v.template modify<DeviceType>();
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackReverseCommDC6>(0,n), *this);
    k_dc6_v.template modify<DeviceType>();
  }
  return n;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackReverseCommCN, const int &i) const
{
  bufV(i)    = d_cn_v(i + first);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackReverseCommDC6, const int &i) const
{
  bufV(i)    = d_dc6_v(i + first);
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm_kokkos(int n,
                                                                    DAT::tdual_int_1d k_sendlistV,
                                                                    DAT::tdual_xfloat_1d &buf)
{
  d_sendlistV = k_sendlistV.view<DeviceType>();
  bufV        = buf.view<DeviceType>();

  if (communicationStage == 1)
  {
    Kokkos::parallel_for( Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackReverseCommCN>(0,n), *this);
    k_cn_v.template modify<DeviceType>();
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for( Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackReverseCommDC6>(0,n), *this);
    k_dc6_v.template modify<DeviceType>();
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackReverseCommCN, const int &i) const
{
  const int j = d_sendlistV(i);
  Kokkos::atomic_add(&d_cn_v(j), bufV(i));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackReverseCommDC6, const int &i) const
{
  const int j = d_sendlistV(i);
  Kokkos::atomic_add(&d_dc6_v(j), bufV(i));
}

// base class override 
template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_forward_comm(int n,
                                                          int *list,
                                                          double *buf,
                                                          int /*pbc_flag*/,
                                                          int * /*pbc*/)
{
  if (communicationStage == 1)
  {
    k_cn_v.sync_host();
    int i,j;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      buf[i] = k_cn_v.h_view(j);
    }
  }
  if (communicationStage == 2)
  {
    k_dc6_v.sync_host();
    int i,j;

    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      buf[i] = k_dc6_v.h_view(j);
    }
  }
  return n;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_forward_comm(int n,
                                                             int first,
                                                             double *buf)
{
  if (communicationStage == 1)
  {
    k_cn_v.sync_host();
    for (int i = 0; i < n; i++ )
    {
      k_cn_v.h_view(i + first) = buf[i];
    }
    k_cn_v.modify_host();
  }
  if (communicationStage == 2)
  {
    k_dc6_v.sync_host();
    for (int i = 0; i < n; i++ )
    {
      k_dc6_v.h_view(i + first) = buf[i];
    }
    k_dc6_v.modify_host();
  }
}

template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm(int n,
                                                          int first,
                                                          double *buf)
{
  int m;
  if (communicationStage == 1)
  {
    k_cn_v.sync_host();

    int i, last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++) { buf[m++] = k_cn_v.h_view(i); }
  }

  if (communicationStage == 2)
  {
    k_dc6_v.sync_host();

    int i,last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++) { buf[m++] = k_dc6_v.h_view(i); }
  }
  return m;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm(int n,
                                                             int *list,
                                                             double *buf)
{
  if (communicationStage == 1)
  {
    k_cn_v.sync_host();

    int i,j,m;

    m = 0;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      k_cn_v.h_view(j) += buf[m++];
    }
    k_cn_v.modify_host();
  }

  if (communicationStage == 2)
  {
    k_dc6_v.sync_host();

    int i,j,m;

    m = 0;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      k_dc6_v.h_view(j) += buf[m++];
    }
    k_dc6_v.modify_host();
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int EFLAG = eflag_either;
  const int VFLAG = vflag_either;

  // Atomic views (HALF list needs atomics)
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  // Global energy (mirror PairComputeFunctor logic)
  if (EFLAG && eflag_global) {
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR) {
        ev.evdwl += epair;
      } else {
        if (i < nlocal) ev.evdwl += 0.5 * epair;
        if (j < nlocal) ev.evdwl += 0.5 * epair;
      }
    } else {
      ev.evdwl += 0.5 * epair;
    }
  }

  // Per-atom energy
  if (EFLAG && eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR || i < nlocal) v_eatom[i] += epairhalf;
      if (NEWTON_PAIR || j < nlocal) v_eatom[j] += epairhalf;
    } else {
      v_eatom[i] += epairhalf;
    }
  }

  if (!VFLAG) return;

  const E_FLOAT v0 = delx*delx*fpair;
  const E_FLOAT v1 = dely*dely*fpair;
  const E_FLOAT v2 = delz*delz*fpair;
  const E_FLOAT v3 = delx*dely*fpair;
  const E_FLOAT v4 = delx*delz*fpair;
  const E_FLOAT v5 = dely*delz*fpair;

  if (vflag_global) {
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR) {
        ev.v[0] += v0; ev.v[1] += v1; ev.v[2] += v2;
        ev.v[3] += v3; ev.v[4] += v4; ev.v[5] += v5;
      } else {
        if (i < nlocal) {
          ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
        }
        if (j < nlocal) {
          ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
        }
      }
    } else {
      ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
      ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
    }
  }

  if (vflag_atom) {
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR || i < nlocal) {
        v_vatom(i,0) += 0.5*v0; v_vatom(i,1) += 0.5*v1; v_vatom(i,2) += 0.5*v2;
        v_vatom(i,3) += 0.5*v3; v_vatom(i,4) += 0.5*v4; v_vatom(i,5) += 0.5*v5;
      }
      if (NEWTON_PAIR || j < nlocal) {
        v_vatom(j,0) += 0.5*v0; v_vatom(j,1) += 0.5*v1; v_vatom(j,2) += 0.5*v2;
        v_vatom(j,3) += 0.5*v3; v_vatom(j,4) += 0.5*v4; v_vatom(j,5) += 0.5*v5;
      }
    } else {
      v_vatom(i,0) += 0.5*v0; v_vatom(i,1) += 0.5*v1; v_vatom(i,2) += 0.5*v2;
      v_vatom(i,3) += 0.5*v3; v_vatom(i,4) += 0.5*v4; v_vatom(i,5) += 0.5*v5;
    }
  }
}

namespace LAMMPS_NS {
#ifdef KOKKOS_ENABLE_CUDA
template class PairDispersionD3Kokkos<LMPDeviceType>;
#endif
#ifdef KOKKOS_ENABLE_SERIAL
template class PairDispersionD3Kokkos<LMPHostType>;
#endif
}
    m = 0;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      k_dc6_v.h_view(j) += buf[m++];
    }
    k_dc6_v.modify_host();
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz) const
{
  const int EFLAG = eflag_either;
  const int VFLAG = vflag_either;

  // Atomic views (HALF list needs atomics)
  Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_eatom = k_eatom.view<DeviceType>();
  Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,
               typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > v_vatom = k_vatom.view<DeviceType>();

  // Global energy (mirror PairComputeFunctor logic)
  if (EFLAG && eflag_global) {
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR) {
        ev.evdwl += epair;
      } else {
        if (i < nlocal) ev.evdwl += 0.5 * epair;
        if (j < nlocal) ev.evdwl += 0.5 * epair;
      }
    } else {
      ev.evdwl += 0.5 * epair;
    }
  }

  // Per-atom energy
  if (EFLAG && eflag_atom) {
    const E_FLOAT epairhalf = 0.5 * epair;
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR || i < nlocal) v_eatom[i] += epairhalf;
      if (NEWTON_PAIR || j < nlocal) v_eatom[j] += epairhalf;
    } else {
      v_eatom[i] += epairhalf;
    }
  }

  if (!VFLAG) return;

  const E_FLOAT v0 = delx*delx*fpair;
  const E_FLOAT v1 = dely*dely*fpair;
  const E_FLOAT v2 = delz*delz*fpair;
  const E_FLOAT v3 = delx*dely*fpair;
  const E_FLOAT v4 = delx*delz*fpair;
  const E_FLOAT v5 = dely*delz*fpair;

  if (vflag_global) {
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR) {
        ev.v[0] += v0; ev.v[1] += v1; ev.v[2] += v2;
        ev.v[3] += v3; ev.v[4] += v4; ev.v[5] += v5;
      } else {
        if (i < nlocal) {
          ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
        }
        if (j < nlocal) {
          ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
        }
      }
    } else {
      ev.v[0] += 0.5*v0; ev.v[1] += 0.5*v1; ev.v[2] += 0.5*v2;
      ev.v[3] += 0.5*v3; ev.v[4] += 0.5*v4; ev.v[5] += 0.5*v5;
    }
  }

  if (vflag_atom) {
    if (NEIGHFLAG != FULL) {
      if (NEWTON_PAIR || i < nlocal) {
        v_vatom(i,0) += 0.5*v0; v_vatom(i,1) += 0.5*v1; v_vatom(i,2) += 0.5*v2;
        v_vatom(i,3) += 0.5*v3; v_vatom(i,4) += 0.5*v4; v_vatom(i,5) += 0.5*v5;
      }
      if (NEWTON_PAIR || j < nlocal) {
        v_vatom(j,0) += 0.5*v0; v_vatom(j,1) += 0.5*v1; v_vatom(j,2) += 0.5*v2;
        v_vatom(j,3) += 0.5*v3; v_vatom(j,4) += 0.5*v4; v_vatom(j,5) += 0.5*v5;
      }
    } else {
      v_vatom(i,0) += 0.5*v0; v_vatom(i,1) += 0.5*v1; v_vatom(i,2) += 0.5*v2;
      v_vatom(i,3) += 0.5*v3; v_vatom(i,4) += 0.5*v4; v_vatom(i,5) += 0.5*v5;
    }
  }
}

namespace LAMMPS_NS {
#ifdef KOKKOS_ENABLE_CUDA
template class PairDispersionD3Kokkos<LMPDeviceType>;
#endif
#ifdef KOKKOS_ENABLE_SERIAL
template class PairDispersionD3Kokkos<LMPHostType>;
#endif
}


