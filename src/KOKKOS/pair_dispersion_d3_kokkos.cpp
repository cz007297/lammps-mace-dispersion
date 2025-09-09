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
#ifdef D3K_DEBUG
#define D3K_DEBUG_NETF
#define D3K_DEBUG_BALANCE
#define D3K_DEBUG_CN
#define D3K_DEBUG_DC6
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
  datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
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

// ...existing code...
template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::init_one(int i, int j)
{
  // Let base class do mixing / set r0ab, cutsq, etc. for (i,j)
  double cut = PairDispersionD3::init_one(i,j);

  const int ntypes = atom->ntypes;

  // Allocate DualViews once (first call)
  if (!k_r0ab_v.span()) {
    k_mxci_v  = DAT::tdual_float_1d("k_mxci", ntypes+1);
    k_r2r4_v  = DAT::tdual_float_1d("k_r2r4", ntypes+1);
    k_rcov_v  = DAT::tdual_float_1d("k_rcov", ntypes+1);
    k_r0ab_v  = DAT::tdual_float_2d("k_r0ab", ntypes+1, ntypes+1);
    k_cutsq_v = DAT::tdual_float_2d("k_cutsq", ntypes+1, ntypes+1);

    // Inner grid dims are fixed to 5x5x3 in this build
    k_c6ab_v  = tdual_float_5d("k_c6ab", ntypes+1, ntypes+1, 5, 5, 3);

    // Fill 1D arrays (type properties) once on host
    for (int t = 0; t <= ntypes; ++t) {
      k_mxci_v.h_view(t)  = mxci[t];
      k_r2r4_v.h_view(t)  = r2r4[t];
      k_rcov_v.h_view(t)  = rcov[t];
    }
  }

  // For this (i,j) we can fill r0ab and cutsq immediately
  // r0ab comes from base; cutsq is derived from cut for this pair
  k_r0ab_v.h_view(i,j) = r0ab[i][j];
  k_r0ab_v.h_view(j,i) = r0ab[j][i];

  const float cutsq_local = static_cast<float>(cut * cut);
  k_cutsq_v.h_view(i,j) = cutsq_local;
  k_cutsq_v.h_view(j,i) = cutsq_local;

  // If this is the last init_one call, populate the FULL tables and sync once
  if (i == ntypes && j == ntypes) {
    // Fill all r0ab and cutsq from the base class arrays for every (t1,t2)
    for (int t1 = 1; t1 <= ntypes; ++t1) {
      for (int t2 = 1; t2 <= ntypes; ++t2) {
        k_r0ab_v.h_view(t1,t2) = r0ab[t1][t2];

        // Use the base Pair::cutsq if available (it has been set across init_one calls)
        // Fallback to previous value if uninitialized.
        const F_FLOAT csq = this->cutsq ? this->cutsq[t1][t2] : static_cast<double>(k_cutsq_v.h_view(t1,t2));
        k_cutsq_v.h_view(t1,t2) = static_cast<float>(csq);

        // Copy the C6 grid block for this (t1,t2)
        const int Gi = std::min<int>(mxci[t1] + 1, k_c6ab_v.extent(2)); // cap by 5
        const int Gj = std::min<int>(mxci[t2] + 1, k_c6ab_v.extent(3)); // cap by 5
        for (int gi = 0; gi < Gi; ++gi) {
          for (int gj = 0; gj < Gj; ++gj) {
            // channels: 0=c6_ref (in a.u.), 1=cni_ref, 2=cnj_ref
            k_c6ab_v.h_view(t1,t2,gi,gj,0) = c6ab[t1][t2][gi][gj][0];
            k_c6ab_v.h_view(t1,t2,gi,gj,1) = c6ab[t1][t2][gi][gj][1];
            k_c6ab_v.h_view(t1,t2,gi,gj,2) = c6ab[t1][t2][gi][gj][2];

            // Ensure symmetric block is present too (swap gi/gj when swapping t1/t2)
            k_c6ab_v.h_view(t2,t1,gj,gi,0) = c6ab[t2][t1][gj][gi][0];
            k_c6ab_v.h_view(t2,t1,gj,gi,1) = c6ab[t2][t1][gj][gi][1];
            k_c6ab_v.h_view(t2,t1,gj,gi,2) = c6ab[t2][t1][gj][gi][2];
          }
        }
      }
    }

    // Push all host mirrors to device once
    k_mxci_v.modify_host();   k_mxci_v.template sync<DeviceType>();
    k_r2r4_v.modify_host();   k_r2r4_v.template sync<DeviceType>();
    k_rcov_v.modify_host();   k_rcov_v.template sync<DeviceType>();
    k_r0ab_v.modify_host();   k_r0ab_v.template sync<DeviceType>();
    k_cutsq_v.modify_host();  k_cutsq_v.template sync<DeviceType>();
    k_c6ab_v.modify_host();   k_c6ab_v.template sync<DeviceType>();

    // Bind device views for kernels
    d_mxci_v  = k_mxci_v.template view<DeviceType>();
    d_r2r4_v  = k_r2r4_v.template view<DeviceType>();
    d_rcov_v  = k_rcov_v.template view<DeviceType>();
    d_r0ab_v  = k_r0ab_v.template view<DeviceType>();
    d_cutsq_v = k_cutsq_v.template view<DeviceType>();
    d_c6ab_v  = k_c6ab_v.template view<DeviceType>();
  }

  return cut;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairDispersionD3::coeff(narg,arg);
}


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_style()
{
  PairDispersionD3::init_style();

  // Adjust neighbor list request for KOKKOS
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> && !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  auto debug_netF = [&](const char* tag){
  #ifdef D3K_DEBUG_NETF
    auto vf = d_f;
    const int nloc = nlocal;
    const int nall_loc = nlocal + atom->nghost;
  
    double sx_o=0, sy_o=0, sz_o=0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nloc),
      KOKKOS_LAMBDA(int i, double& s){ s += vf(i,0); }, sx_o);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nloc),
      KOKKOS_LAMBDA(int i, double& s){ s += vf(i,1); }, sy_o);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nloc),
      KOKKOS_LAMBDA(int i, double& s){ s += vf(i,2); }, sz_o);
  
    double sx_ag=0, sy_ag=0, sz_ag=0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nall_loc),
      KOKKOS_LAMBDA(int i, double& s){ s += vf(i,0); }, sx_ag);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nall_loc),
      KOKKOS_LAMBDA(int i, double& s){ s += vf(i,1); }, sy_ag);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nall_loc),
      KOKKOS_LAMBDA(int i, double& s){ s += vf(i,2); }, sz_ag);
  
    double GO[3], GAG[3];
    double go[3]  = {sx_o,  sy_o,  sz_o};
    double gag[3] = {sx_ag, sy_ag, sz_ag};
    MPI_Allreduce(go,  GO,  3, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(gag, GAG, 3, MPI_DOUBLE, MPI_SUM, world);
    if (comm->me == 0) {
      fprintf(stderr, "[D3K] %s step %lld netF_owned=(%.6e %.6e %.6e) netF_all=(%.6e %.6e %.6e)\n",
              tag, (long long)update->ntimestep, GO[0], GO[1], GO[2], GAG[0], GAG[1], GAG[2]);
    }
  #endif
  };  
  
  std::unordered_map<std::string, int> dampingMap = {
      {"original", 1}, {"zerom", 2}, {"bj", 3}, {"bjm",4}};
  int dampingCode = dampingMap[damping_type];
  
  int eflag = eflag_in;
  int vflag = vflag_in;
  
  ev_init(eflag, vflag);
  
  if (eflag_atom)
  {
    if (!k_eatom.span() || k_eatom.extent(0) < nmax) 
    {
      memoryKK->destroy_kokkos(k_eatom, eatom);
      memoryKK->create_kokkos(k_eatom, eatom, nmax, "pair:eatom");
    }
    d_eatom = k_eatom.view<DeviceType>();  
  }
  if (vflag_atom)
  {
    if (!k_vatom.span() || k_vatom.extent(0) < nmax) 
    {
      memoryKK->destroy_kokkos(k_vatom, vatom);
      memoryKK->create_kokkos(k_vatom, vatom, nmax, "pair:vatom");
    }
    d_vatom = k_vatom.view<DeviceType>();
  }
  
  // FIRST SYNC AFTER EV_INIT
  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag ) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space, F_MASK);
 
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  
  // Ensure per-atom capacity first
  const int req_nmax = atom->nmax;
  if (!k_cn_v.span() || !k_dc6_v.span() || req_nmax > nmax) {
    nmax = req_nmax;
    memoryKK->grow_kokkos(k_cn_v,  cn,  nmax, "pair:cn");
    memoryKK->grow_kokkos(k_dc6_v, dc6, nmax, "pair:dc6");
  }

  //nmax = atomKK->nmax;

  d_cn_v  = k_cn_v.view<DeviceType>();
  d_dc6_v = k_dc6_v.view<DeviceType>();
  d_cutsq_v = k_cutsq_v.view<DeviceType>();

  d_x    = atomKK->k_x.view<DeviceType>();
  d_f    = atomKK->k_f.view<DeviceType>();
  d_type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall   = nlocal + atom->nghost;
  newton_pair = force->newton_pair; 


  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh   =  k_list->d_numneigh;
  d_neighbors  =  k_list->d_neighbors;
  d_ilist      =  k_list->d_ilist;
  inum = list->inum;

  //k_cn_v.template sync<DeviceType>();
  //k_dc6_v.template sync<DeviceType>();
  //k_cn_v.template modify<DeviceType>();
  //k_dc6_v.template modify<DeviceType>();


  need_dup = lmp->kokkos->need_dup<DeviceType>();
  
  if (need_dup) {
    dup_cn    =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_cn_v);
    dup_dc6   =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_dc6_v);
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom); 
  } else {

  ndup_cn    =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_cn_v);
  ndup_dc6   =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_dc6_v);
  ndup_f     =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_f);
  ndup_eatom =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
  ndup_vatom =  Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }







  copymode = 1; 

  // Zero out cn & dc6
  Kokkos::deep_copy(d_dc6_v, 0.0);
  Kokkos::deep_copy(d_cn_v, 0.0);
  k_dc6_v.template modify<DeviceType>();
  k_cn_v.template modify<DeviceType>(); 
 
  // Compute CN aka Kernel 0 

  Kokkos::parallel_for(policyInstance<TagPairDD3KokkosCNDC6Kernel<HALF>>::get(inum), *this); 

  if (need_dup) Kokkos::Experimental::contribute(d_cn_v, dup_cn);

  communicationStage = 1;
  // Communicated calculated kernel
  if (newton_pair) 
    {
      k_cn_v.template modify<DeviceType>(); 
      comm->reverse_comm(this);
      k_cn_v.template sync<DeviceType>();
    }
  k_cn_v.template modify<DeviceType>();
  comm->forward_comm(this);
  k_cn_v.template sync<DeviceType>();

  #if defined(D3K_DEBUG_CN) && !defined(KOKKOS_ENABLE_CUDA)
  if (comm->me==0) {
    int imax = std::min(5, nlocal);
    for (int i=0;i<imax;i++) printf("[CN ] i=%d cn=%.6e\n", i, (double)k_cn_v.h_view(i));
  }
  #endif
  // Main Kernels Switch Logic Here - NEWTON_PAIR hardcoded to one by design 
  // compute kernel 1
  switch (dampingCode) {
    case 1: { 
      if (evflag)
      {
        EV_FLOAT ev;
        Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdIJOriginalZeroDampKernel<HALF, 1, 1>>(0, inum),
          *this, ev);
        if (eflag_global) eng_vdwl += ev.evdwl;
        if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
      }
      else
      {
        Kokkos::parallel_for(
          policyInstance<TagPairDispDD3dEdIJOriginalZeroDampKernel<HALF, 1, 0>>::get(inum), *this);
      }
     } break;
    case 2: { 
      if (evflag)
      {
        EV_FLOAT ev;
        Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdIJModifiedZeroDampKernel<HALF, 1, 1>>(0, inum),
          *this, ev);
        if (eflag_global) eng_vdwl += ev.evdwl;
        if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
      }
      else
      {
        Kokkos::parallel_for(
          policyInstance<TagPairDispDD3dEdIJModifiedZeroDampKernel<HALF, 1, 0>>::get(inum), *this);
      }
     } break;
    case 3: { 
      if (evflag)
      {
        EV_FLOAT ev;
        Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdIJOriginalBJDampKernel<HALF, 1, 1>>(0, inum),
          *this, ev);
        //if (eflag_global) eng_vdwl += ev.evdwl;
        if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
      }
      else
      {
        Kokkos::parallel_for(
          policyInstance<TagPairDispDD3dEdIJOriginalBJDampKernel<HALF, 1, 0>>::get(inum), *this);
      }
     } break;
    case 4: { 
      if (evflag)
      {
        EV_FLOAT ev;
        Kokkos::parallel_reduce(
          Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdIJModifiedBJDampKernel<HALF, 1, 1>>(0, inum),
          *this, ev);
        if (eflag_global) eng_vdwl += ev.evdwl;
        if (vflag_global) for (int m=0; m<6; ++m) virial[m] += ev.v[m];
      }
      else
      {
        Kokkos::parallel_for(
          policyInstance<TagPairDispDD3dEdIJModifiedBJDampKernel<HALF, 1, 0>>::get(inum), *this);
      }
     } break;
    
  }
  debug_netF("after_dEdIJ_beforesync");

  if (need_dup) {
    Kokkos::Experimental::contribute(d_f, dup_f);
    Kokkos::Experimental::contribute(d_dc6_v, dup_dc6);
  }
  //atomKK->modified(execution_space, datamask_modify);
  //atomKK->sync(execution_space, datamask_modify); 
  //debug_netF("after_dEdIJ_aftersync_beforecomm");
  communicationStage = 2;
  if (newton_pair) 
    {
      k_dc6_v.template modify<DeviceType>(); 
      comm->reverse_comm(this);
      k_dc6_v.template sync<DeviceType>();
    }
  k_dc6_v.template modify<DeviceType>();
  comm->forward_comm(this);
  k_dc6_v.template sync<DeviceType>();

  //debug_netF("after_dEdIJ_aftersync_aftercomm");
  //atomKK->sync(execution_space, F_MASK);
  //debug_netF("after_dEdIJ_aftersync_aftercomm_aftersecondcomm");
  #if defined(D3K_DEBUG_DC6) && !defined(KOKKOS_ENABLE_CUDA)
  atomKK->sync(Host, 0); // ensure host mirrors fresh
  if (comm->me==0) {
    int imax = std::min(5, nlocal);
    for (int i=0;i<imax;i++) printf("[DC6] i=%d dc6=%.6e\n", i, (double)k_dc6_v.h_view(i));
  }
  #endif  


  // compute kernel 2
  if (evflag)
  {
    EV_FLOAT ev;
    Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType, TagPairDispDD3dEdXYZ<HALF, 1, 1>>(0, inum),
      *this, ev);
    // no energy contributions needed in kernel 2
  }
  else
  {
    Kokkos::parallel_for(
      policyInstance<TagPairDispDD3dEdXYZ<HALF, 1, 0>>::get(inum),*this);
  }
  debug_netF("after_dEdXYZ_beforesync");
  // final syncs
  //atomKK->modified(execution_space, datamask_modify);
  //debug_netF("after_dEdXYZ_beforesync_aftermod_beforesync");
  if (need_dup) {
    Kokkos::Experimental::contribute(d_f, dup_f);
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);
   
  if (eflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }
  if (vflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  //atomKK->sync(execution_space, F_MASK);
  copymode = 0;
  if constexpr (!std::is_same_v<DeviceType, LMPHostType>) {
    atomKK->sync(Host, F_MASK);
  } 
  //debug_netF("after_dEdXYZ_beforesync_aftermod_beforesync_aftersync");
  
  // Free allocated memory
  if (need_dup) {
    dup_cn         = {};
    dup_dc6        = {};
    dup_f          = {};
    dup_eatom      = {};
    dup_vatom      = {};
  }
  

}


template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3KokkosCNDC6Kernel<NEIGHFLAG>, const int &ii) const
{
 
  // The cn array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  auto v_cn_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_cn), decltype(ndup_cn)>::get(dup_cn, ndup_cn);
  auto a_cn_scv = v_cn_scv.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  if (ii >= inum) return;    
  const int     i      = d_ilist[ii];
  const int     itype  = d_type(i);
  if (itype <= 0 || itype >= d_rcov_v.extent(0)) return; 
  const int     jnum   = d_numneigh[i];
  const F_FLOAT xi     = d_x(i,0);
  const F_FLOAT yi     = d_x(i,1);
  const F_FLOAT zi     = d_x(i,2);

  for (int jj = 0; jj < jnum; jj++)
  {
    int j                = d_neighbors(i, jj) & NEIGHMASK; 
    const int     jtype  = d_type(j); 
    if (jtype <= 0 || jtype >= d_rcov_v.extent(0)) continue;
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

    a_cn_scv(i) += cn_ij;
    if (newton_pair || j < nlocal) a_cn_scv(j) += cn_ij; 
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
  static void Operator( const ViewType &d_c6ab_v,
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
   
    const F_FLOAT  cni_ref  = d_c6ab_v(iat, jat, ci, cj, 1);
    const F_FLOAT  cnj_ref  = d_c6ab_v(iat, jat, ci, cj, 2);
  
    const F_FLOAT    dx_i  = cni - cni_ref;
    const F_FLOAT    dx_j  = cnj - cnj_ref; 

    const F_FLOAT       r  = dx_i*dx_i  + dx_j*dx_j;
  
    if ( r < acc.r_save) { acc.r_save = r; acc.c6mem = c6_ref; }
  
    double      expterm   = exp(K3*r);
    acc.num              += c6_ref * expterm;
    acc.den              += expterm;

    expterm              *= 2.0 * K3;
    const F_FLOAT  term_i  = expterm * dx_i;
    const F_FLOAT  term_j  = expterm * dx_j;
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

KOKKOS_INLINE_FUNCTION
static int sbmask_disp(const int jfull) {
  return (jfull >> SBBITS) & 3;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::dC6KK
(
  const    int iat, const    int jat,
  const F_FLOAT cni, const F_FLOAT cnj,
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
      DC6Derive::Operator(d_c6ab_v, iat, jat, ci, cj, cni, cnj, acc);
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
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDispDD3dEdXYZ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  auto v_f_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f_scv = v_f_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>(); 

  
  if (ii >= inum) return;
  const int i = d_ilist[ii];
  if (i >= nlocal) return ; 
  const F_FLOAT  xi = d_x(i,0);
  const F_FLOAT  yi = d_x(i,1);
  const F_FLOAT  zi = d_x(i,2);
  const int itype = d_type(i);

  if (itype <= 0 || itype >= d_mxci_v.extent(0)) return;
  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) 
  {
    const int jfull = d_neighbors(i,jj);
    const F_FLOAT factor_lj = special_lj[sbmask_disp(jfull)]; 
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);

    const F_FLOAT  dx = xi - d_x(j,0);
    const F_FLOAT  dy = yi - d_x(j,1);
    const F_FLOAT  dz = zi - d_x(j,2);
    const F_FLOAT  rsq = dx*dx + dy*dy + dz*dz;

    if (rsq < d_cutsq_v(itype, jtype)) 
    {
      const F_FLOAT r = sqrt(rsq);
      
      F_FLOAT dcn = 0.0;
      if (rsq < cn_thr)
      { 
        const F_FLOAT  rcovij  = (d_rcov_v(itype) + d_rcov_v(jtype))*autoang;
        const F_FLOAT  expterm = exp(-K1 * (rcovij / r - 1.0));
        dcn = -K1 * rcovij * expterm / (rsq * (expterm + 1.0) * (expterm + 1.0));
      } 

      const F_FLOAT  fpair1 = dcn * (d_dc6_v(i) + d_dc6_v(j)) / r ; 
      const F_FLOAT  fpair  = fpair1*factor_lj;
     
      const F_FLOAT  fx = dx * fpair;
      const F_FLOAT  fy = dy * fpair; 
      const F_FLOAT  fz = dz * fpair;
    
      fix += fx;
      fiy += fy;
      fiz += fz;

      if (NEWTON_PAIR || j < nlocal) {
        a_f_scv(j,0) -= fx;
        a_f_scv(j,1) -= fy;
        a_f_scv(j,2) -= fz;
      }

      if (EVFLAG) {this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, 0.0, fpair, dx, dy, dz);} 
    }
  }
  
  a_f_scv(i, 0) += fix;
  a_f_scv(i, 1) += fiy;
  a_f_scv(i, 2) += fiz;
}

// Thin wrapper: NO-EV kernel
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDispDD3dEdXYZ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int& ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(TagPairDispDD3dEdXYZ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
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
  
  auto v_eatom_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_eatom),decltype(ndup_eatom)>::get(dup_eatom,ndup_eatom);
  auto a_eatom_scv = v_eatom_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  auto v_vatom_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom_scv = v_vatom_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

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
      if (NEWTON_PAIR || i < nlocal) a_eatom_scv(i) += epairhalf;
      if (NEWTON_PAIR || j < nlocal) a_eatom_scv(j) += epairhalf;
    } else {
      a_eatom_scv(i) += epairhalf;
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
        a_vatom_scv(i,0) += 0.5*v0; a_vatom_scv(i,1) += 0.5*v1; a_vatom_scv(i,2) += 0.5*v2;
        a_vatom_scv(i,3) += 0.5*v3; a_vatom_scv(i,4) += 0.5*v4; a_vatom_scv(i,5) += 0.5*v5;
      }
      if (NEWTON_PAIR || j < nlocal) {
        a_vatom_scv(j,0) += 0.5*v0; a_vatom_scv(j,1) += 0.5*v1; a_vatom_scv(j,2) += 0.5*v2;
        a_vatom_scv(j,3) += 0.5*v3; a_vatom_scv(j,4) += 0.5*v4; a_vatom_scv(j,5) += 0.5*v5;
      }
    } else {
      a_vatom_scv(i,0) += 0.5*v0; a_vatom_scv(i,1) += 0.5*v1; a_vatom_scv(i,2) += 0.5*v2;
      a_vatom_scv(i,3) += 0.5*v3; a_vatom_scv(i,4) += 0.5*v4; a_vatom_scv(i,5) += 0.5*v5;
    }
  }
}



template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT &ev) const
{
  auto v_f_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f_scv = v_f_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>(); 

  auto v_dc6_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_dc6), decltype(ndup_dc6)>::get(dup_dc6, ndup_dc6);
  auto a_dc6_scv = v_dc6_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();  

  if (ii >= inum) return;
  const int i = d_ilist[ii];
  if (i >= nlocal) return;
  const int itype = d_type(i);
  if (itype <= 0 || itype >= d_mxci_v.extent(0)) return;

  const F_FLOAT xi = d_x(i,0);
  const F_FLOAT yi = d_x(i,1);
  const F_FLOAT zi = d_x(i,2);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) {
    const int jfull = d_neighbors(i,jj);
    const F_FLOAT factor_lj = special_lj[sbmask_disp(jfull)];
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);
    if (jtype <= 0 || jtype >= d_mxci_v.extent(0)) continue; 

    const F_FLOAT dx  = xi - d_x(j,0);
    const F_FLOAT dy  = yi - d_x(j,1);
    const F_FLOAT dz  = zi - d_x(j,2);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= d_cutsq_v(itype,jtype)) continue;

    const F_FLOAT r      = sqrt(rsq);
    const F_FLOAT r2inv  = 1.0 / rsq;
    const F_FLOAT r4inv  = r2inv * r2inv;
    const F_FLOAT r6inv  = r4inv * r2inv;
    const F_FLOAT r8inv  = r6inv * r2inv;
    const F_FLOAT r10inv = r8inv * r2inv;

    const F_FLOAT cni = d_cn_v(i);
    const F_FLOAT cnj = d_cn_v(j);

    double C6=0.0, dC6_i=0.0, dC6_j=0.0;
    dC6KK(itype, jtype, cni, cnj, C6, dC6_i, dC6_j);
    if (C6 == 0.0) continue;

    const F_FLOAT C8 = 3.0 * C6 * d_r2r4_v(itype) * d_r2r4_v(jtype) * (autoang * autoang);

    const F_FLOAT r0     = r / d_r0ab_v(itype, jtype);
    const F_FLOAT alpha6 = alpha;
    const F_FLOAT alpha8 = alpha + 2.0;

    const F_FLOAT t6    = pow(rs6 / r0, alpha6);
    const F_FLOAT damp6 = 1.0 / (1.0 + 6.0 * t6);
    const F_FLOAT t8    = pow(rs8 / r0, alpha8);
    const F_FLOAT damp8 = 1.0 / (1.0 + 6.0 * t8);

    const F_FLOAT e6 = C6 * damp6 * r6inv;
    const F_FLOAT e8 = C8 * damp8 * r8inv;

    const F_FLOAT tmp6 = 6.0 * s6 * C6 * r8inv  * damp6;
    const F_FLOAT tmp8 = 8.0 * s8 * C8 * r10inv * damp8;

    const F_FLOAT fpair_no_damp = -(tmp6 + tmp8);
    const F_FLOAT fpair_damp    =  (tmp6 * alpha6 * t6 * damp6)
                                + (tmp8 * alpha8 * t8 * damp8 * 0.75);
    const F_FLOAT fpair = (fpair_no_damp + fpair_damp) * factor_lj;

    const F_FLOAT phi = -(s6 * e6 + s8 * e8) * factor_lj;

    const F_FLOAT rest = (s6 * e6 + s8 * e8) / C6;
    
    a_dc6_scv(i) += rest*dC6_i;
    if (NEWTON_PAIR || j < nlocal)
      a_dc6_scv(j) += rest*dC6_j;

    const F_FLOAT fx = dx * fpair;
    const F_FLOAT fy = dy * fpair;
    const F_FLOAT fz = dz * fpair;
    
    fix += fx;
    fiy += fy;
    fiz += fz; 

    if (NEWTON_PAIR || j < nlocal) {
      a_f_scv(j,0) -= fx;
      a_f_scv(j,1) -= fy;
      a_f_scv(j,2) -= fz;
    }

    if (EVFLAG) this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, phi, fpair, dx, dy, dz);
  }

  a_f_scv(i, 0) += fix;
  a_f_scv(i, 1) += fiy;
  a_f_scv(i, 2) += fiz;
}
// NO-EV version (RangePolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  operator()(TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}


template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT &ev) const
{
  auto v_f_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f_scv = v_f_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>(); 

  auto v_dc6_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_dc6), decltype(ndup_dc6)>::get(dup_dc6, ndup_dc6);
  auto a_dc6_scv = v_dc6_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();  

  if (ii >= inum) return;
  const int i = d_ilist[ii];
  if (i >= nlocal) return;
  const int itype = d_type(i);
  if (itype <= 0 || itype >= d_mxci_v.extent(0)) return;

  const F_FLOAT xi = d_x(i,0);
  const F_FLOAT yi = d_x(i,1);
  const F_FLOAT zi = d_x(i,2);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) {
    const int jfull = d_neighbors(i,jj);
    const F_FLOAT factor_lj = special_lj[sbmask_disp(jfull)];
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);
    if (jtype <= 0 || jtype >= d_mxci_v.extent(0)) continue;

    const F_FLOAT dx  = xi - d_x(j,0);
    const F_FLOAT dy  = yi - d_x(j,1);
    const F_FLOAT dz  = zi - d_x(j,2);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= d_cutsq_v(itype,jtype)) continue;

    const F_FLOAT r      = sqrt(rsq);
    const F_FLOAT r2inv  = 1.0 / rsq;
    const F_FLOAT r4inv  = r2inv * r2inv;
    const F_FLOAT r6inv  = r4inv * r2inv;
    const F_FLOAT r8inv  = r6inv * r2inv;
    const F_FLOAT r10inv = r8inv * r2inv;

    const F_FLOAT cni = d_cn_v(i);
    const F_FLOAT cnj = d_cn_v(j);

    double C6=0.0, dC6_i=0.0, dC6_j=0.0;
    dC6KK(itype, jtype, cni, cnj, C6, dC6_i, dC6_j);
    if (C6 == 0.0) continue;

    const F_FLOAT C8 = 3.0 * C6 * d_r2r4_v(itype) * d_r2r4_v(jtype) * (autoang * autoang);

    const F_FLOAT r0     = d_r0ab_v(itype, jtype);
    const F_FLOAT alpha6 = alpha;
    const F_FLOAT alpha8 = alpha + 2.0;

    const F_FLOAT t6     = pow((r / (rs6*r0))+rs8*r0, -alpha6);
    const F_FLOAT t8     = pow((r/r0)+rs8*r0, -alpha8);
    const F_FLOAT damp6  = 1.0 / ( 1.0 + (6.0*t6));
    const F_FLOAT damp8  = 1.0 / ( 1.0 + (6.0*t8));
   
    const F_FLOAT e6     = C6 * damp6 * r6inv;
    const F_FLOAT e8     = C8 * damp8 * r8inv;

    const F_FLOAT tmp6   = 6.0 * s6 * C6 * r8inv  * damp6;   
    const F_FLOAT tmp8   = 8.0 * s8 * C8 * r10inv * damp8;

    const F_FLOAT fpair1   = -(tmp6 + tmp8);
 
    const F_FLOAT fp26     = tmp6 * alpha6 * t6 * damp6 * r / (r+rs6*rs8*r0*r0);
    const F_FLOAT fp28     = tmp8 * alpha8 * t8 * damp8 * r / (r+rs6*r0*r0);

    const F_FLOAT fpair2   = fp26 + (0.75 *fp28);
    const F_FLOAT fpair    = (fpair1 + fpair2) * factor_lj; 
    
    const F_FLOAT phi = -(s6 * e6 + s8 * e8) * factor_lj;

    const F_FLOAT rest = (s6 * e6 + s8 * e8) / C6;
    
    a_dc6_scv(i) += rest*dC6_i;
    if (NEWTON_PAIR || j < nlocal)
      a_dc6_scv(j) += rest*dC6_j;

    const F_FLOAT fx = dx * fpair;
    const F_FLOAT fy = dy * fpair;
    const F_FLOAT fz = dz * fpair;

    fix += fx; fiy += fy; fiz += fz; 
    if (NEWTON_PAIR || j < nlocal) {
      a_f_scv(j,0) -= fx;
      a_f_scv(j,1) -= fy;
      a_f_scv(j,2) -= fz;
    }

    if (EVFLAG) this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, phi, fpair, dx, dy, dz);
  }

  a_f_scv(i, 0) += fix;
  a_f_scv(i, 1) += fiy;
  a_f_scv(i, 2) += fiz;
}
// NO-EV version (RangePolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  operator()(TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}




template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT &ev) const
{

  F_FLOAT fix = 0.0, fiy = 0.0, fiz = 0.0;

  auto v_f_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f_scv = v_f_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>(); 

  auto v_dc6_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_dc6), decltype(ndup_dc6)>::get(dup_dc6, ndup_dc6);
  auto a_dc6_scv = v_dc6_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();  

  if (ii >= inum) return;
  const int i = d_ilist[ii];
  if (i >= nlocal) return;

  const int itype = d_type(i);
  if (itype <= 0 || itype >= d_mxci_v.extent(0)) return;

  const F_FLOAT xi = d_x(i,0);
  const F_FLOAT yi = d_x(i,1);
  const F_FLOAT zi = d_x(i,2);


  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) {
    const int jfull = d_neighbors(i,jj);
    const F_FLOAT factor_lj = special_lj[sbmask_disp(jfull)];
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);
    if (jtype <= 0 || jtype >= d_mxci_v.extent(0)) continue;

    const F_FLOAT  dx  = xi - d_x(j,0);
    const F_FLOAT  dy  = yi - d_x(j,1);
    const F_FLOAT  dz  = zi - d_x(j,2);
    const F_FLOAT  rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= d_cutsq_v(itype,jtype)) continue;

    const F_FLOAT  r      = sqrt(rsq);
    const F_FLOAT  cni = d_cn_v(i);
    const F_FLOAT  cnj = d_cn_v(j);

    F_FLOAT C6=0.0, dC6_i=0.0, dC6_j=0.0;
    dC6KK(itype, jtype, cni, cnj, C6, dC6_i, dC6_j);
    if (C6 == 0.0) continue;

    const F_FLOAT   C8 = 3.0 * C6 * d_r2r4_v(itype) * d_r2r4_v(jtype) * (autoang * autoang);
    const F_FLOAT   r0     = sqrt(C8/C6);
    const F_FLOAT   r4     = rsq*rsq;
    const F_FLOAT   r6     = rsq*rsq*rsq;
    const F_FLOAT   r8     = rsq*rsq*rsq*rsq;

    // ip - inner product 
    const F_FLOAT   iptmp  = ((a1*r0)+a2);
    const F_FLOAT   expt6  = iptmp*iptmp*iptmp*iptmp*iptmp*iptmp;
    const F_FLOAT   expt8  = iptmp*iptmp*iptmp*iptmp*iptmp*iptmp*iptmp*iptmp;
    const F_FLOAT   t6     = r6 + expt6 ;
    const F_FLOAT   t8     = r8 + expt8 ; 
    const F_FLOAT   e6     = C6/t6;
    const F_FLOAT   e8     = C8/t8;
    const F_FLOAT   tmp6   = (6.0 * s6 * C6 * r4) / (t6*t6);
    const F_FLOAT   tmp8   = (8.0 * s8 * C8 * r6) / (t8*t8);
    const F_FLOAT   fpairsum = -(tmp6 + tmp8);
    const F_FLOAT   fpair    = fpairsum * factor_lj;
    const F_FLOAT   phi = -(s6 * e6 + s8 * e8) * factor_lj;
    const F_FLOAT   rest = (s6 * e6 + s8 * e8) / C6;
    

    const F_FLOAT  fx = dx * fpair;
    const F_FLOAT  fy = dy * fpair;
    const F_FLOAT  fz = dz * fpair;

    fix += fx; fiy += fy; fiz += fz;
    if (NEWTON_PAIR || j < nlocal) {
      a_f_scv(j,0) -= fx;
      a_f_scv(j,1) -= fy;
      a_f_scv(j,2) -= fz;
    }

    a_dc6_scv(i) += rest*dC6_i;
    if (NEWTON_PAIR || j < nlocal) a_dc6_scv(j) += rest*dC6_j;

    if (EVFLAG) {
      if (eflag_global) { 
        ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<nlocal)))?1.0:0.5)*phi;
      } 
      if (vflag_either || eflag_atom ) this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, phi, fpair, dx, dy, dz);
    }

  }

  a_f_scv(i, 0) += fix;
  a_f_scv(i, 1) += fiy;
  a_f_scv(i, 2) += fiz;
}


// NO-EV version (RangePolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  operator()(TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}


template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii, EV_FLOAT &ev) const
{
   
  #if defined(D3K_DEBUG_BALANCE)
   F_FLOAT sumJx = 0, sumJy = 0, sumJz = 0;
  #endif
  // same logic as Original, different parameters setup in funcpar

  auto v_f_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  auto a_f_scv = v_f_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>(); 

  auto v_dc6_scv = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_dc6), decltype(ndup_dc6)>::get(dup_dc6, ndup_dc6);
  auto a_dc6_scv = v_dc6_scv.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();  

  if (ii >= inum) return;
  const int i = d_ilist[ii];
  if (i >= nlocal) return;
  const int itype = d_type(i);
  if (itype <= 0 || itype >= d_mxci_v.extent(0)) return;

  const F_FLOAT xi = d_x(i,0);
  const F_FLOAT yi = d_x(i,1);
  const F_FLOAT zi = d_x(i,2);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) {
    const int jfull = d_neighbors(i,jj);
    const F_FLOAT factor_lj = special_lj[sbmask_disp(jfull)];
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);
    if (jtype <= 0 || jtype >= d_mxci_v.extent(0)) continue;

    const F_FLOAT dx  = xi - d_x(j,0);
    const F_FLOAT dy  = yi - d_x(j,1);
    const F_FLOAT dz  = zi - d_x(j,2);
    const F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    if (rsq >= d_cutsq_v(itype,jtype)) continue;

    const F_FLOAT r      = sqrt(rsq);
    const F_FLOAT cni = d_cn_v(i);
    const F_FLOAT cnj = d_cn_v(j);

    double C6=0.0, dC6_i=0.0, dC6_j=0.0;
    dC6KK(itype, jtype, cni, cnj, C6, dC6_i, dC6_j);
    if (C6 == 0.0) continue;

    const F_FLOAT C8 = 3.0 * C6 * d_r2r4_v(itype) * d_r2r4_v(jtype) * (autoang * autoang);

    const F_FLOAT r0     = sqrt(C8/C6);
    const F_FLOAT r4     = rsq*rsq;
    const F_FLOAT r6     = rsq*rsq*rsq;
    const F_FLOAT r8     = rsq*rsq*rsq*rsq;

    // ip - inner product 
    const F_FLOAT iptmp  = (a1*r0+a2);
    const F_FLOAT expt6  = iptmp*iptmp*iptmp*iptmp*iptmp*iptmp;
    const F_FLOAT expt8  = iptmp*iptmp*iptmp*iptmp*iptmp*iptmp*iptmp*iptmp;
    const F_FLOAT t6     = r6 + expt6 ;
    const F_FLOAT t8     = r8 + expt8 ; 

    const F_FLOAT e6     = C6/t6;
    const F_FLOAT e8     = C8/t8;


    const F_FLOAT tmp6   = (6.0 * s6 * C6 * r4) / (t6*t6);
    const F_FLOAT tmp8   = (8.0 * s8 * C8 * r6) / (t8*t8);

    const F_FLOAT fpairsum = -(tmp6 + tmp8);
    const F_FLOAT fpair    = fpairsum * factor_lj;


    const F_FLOAT phi = -(s6 * e6 + s8 * e8) * factor_lj;
    const F_FLOAT rest = (s6 * e6 + s8 * e8) / C6;
    
    a_dc6_scv(i) += rest*dC6_i;
    if (NEWTON_PAIR || j < nlocal)
      a_dc6_scv(j) += rest*dC6_j;

    const F_FLOAT fx = dx * fpair;
    const F_FLOAT fy = dy * fpair;
    const F_FLOAT fz = dz * fpair;
    #if defined(D3K_DEBUG_BALANCE)
    sumJx += fx; sumJy += fy; sumJz += fz;
    #endif
    
    fix += fx; fiy += fy; fiz += fz;
    if (NEWTON_PAIR || j < nlocal) {
      a_f_scv(j,0) -= fx;
      a_f_scv(j,1) -= fy;
      a_f_scv(j,2) -= fz;
    }

    if (EVFLAG) this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, phi, fpair, dx, dy, dz);
  }
  #if defined(D3K_DEBUG_BALANCE) && !defined(KOKKOS_ENABLE_CUDA)
  const double mx = fabs((double)fix - (double)sumJx);
  const double my = fabs((double)fiy - (double)sumJy);
  const double mz = fabs((double)fiz - (double)sumJz);
  if (mx>1e-12 || my>1e-12 || mz>1e-12) {
    printf("BJ DAMPING [BAL] step=%lld i=%d mx=%.3e my=%.3e mz=%.3e jnum=%d\n",
           (long long)update->ntimestep, i, mx, my, mz, d_numneigh(i));
  }
  #endif


  a_f_scv(i, 0) += fix;
  a_f_scv(i, 1) += fiy;
  a_f_scv(i, 2) += fiz;
}


// NO-EV version (RangePolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int &ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  operator()(TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(), ii, ev);
}

template<typename DeviceType>
template<class TAG>
struct PairDispersionD3Kokkos<DeviceType>::policyInstance
{
  static auto get(int inum)
  {
    auto policy = Kokkos::RangePolicy<DeviceType, TAG>(0, inum);
    return policy;
  }
};



namespace LAMMPS_NS {
#ifdef KOKKOS_ENABLE_CUDA
template class PairDispersionD3Kokkos<LMPDeviceType>;
#endif
#ifdef KOKKOS_ENABLE_SERIAL
template class PairDispersionD3Kokkos<LMPHostType>;
#endif
}


