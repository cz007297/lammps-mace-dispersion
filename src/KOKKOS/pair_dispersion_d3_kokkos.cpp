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
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp) : PairDispersionD3(lmp)
{
  int nci_, ncj_;
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
  //  memoryKK->destroy_kokkos(k_eatom,eatom);
  //  memoryKK->destroy_kokkos(k_vatom,vatom);
  }

}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairDispersionD3::coeff(narg,arg);
  
  int ntypes = atom->ntypes; 
  int nmax   = atom->nmax;
 
  DAT::tdual_float_1d   k_mxci_v("k_mxci", ntypes+1); 
  DAT::tdual_float_1d   k_cn_v("k_cn", nmax); 
  DAT::tdual_float_1d   k_dc6_v("k_dc6", nmax);
  DAT::tdual_float_1d   k_r2r4_v("k_r2r4", ntypes+1);
  DAT::tdual_float_1d   k_rcov_v("k_rcov", ntypes+1);
  DAT::tdual_float_2d   k_r0ab_v("k_r0ab", ntypes+1, ntypes+1);
  tdual_float_5d   k_c6ab_v("k_c6ab", ntypes+1, ntypes+1, 5, 5, 3);
 
  t_host_float_2d  h_r0ab_v.h_view; 
  t_host_float_1d  h_r2r4_v.h_view;
  t_host_float_1d  h_rcov_v.h_view;
  t_host_float_5d  h_c6ab_v.h_view;
  t_host_int_1d    h_mxci_v.h_view;

  for (int i; i <= ntypes; i++)
  {
    h_r2r4_v = r2r4[i];
    h_rcov_v = rcov[i];
    h_mxci_v = mxci[i];
    for (int j; j <= ntypes; j++)
    {
      h_r0ab_v = r0ab[i][j];
      for (int gi; gi < 5; gi++)
      {
        for (int gj; gj < 5; gj++)
        {
          for (int k; k < 3; k++)
          {
            h_c6ab(i, j, gi, gj, k) = c_6ab[i][j][gi][gj][k];
          }
        }
      }
    }
  }
 
  k_r2r4_v.modify_host();
  k_rcov_v.modify_host(); 
  k_r0ab_v.modify_host(); 
  k_c6ab_v.modify_host();
  k_mxci_v.modify_host();
  
  k_mxci_v.template sync<DeviceType>();
  k_c6ab_v.template sync<DeviceType>();  
  k_rcov_v.template sync<DeviceType>();
  k_r0ab_v.template sync<DeviceType>();
  k_c6ab_v.template sync<DeviceType>();
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_numberKK()
{
  atomKK->sync(execution_space,datamask_read);
  x           = atomKK->k_x.view<DeviceType>();
  f           = atomKK->k_f.view<DeviceType>();
  type        = atomKK->k_type.view<DeviceType>();
  nlocal      = atomKK->nlocal;
  nall        = atomKK->nghost + nlocal;
  newton_pair = force->newton_pair; 

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  if (atomKK->nmax > nmax)
  {
    nmax = atomKK-> nmax 
    memoryKK->grow_kokkos(k_cn, cn, "pair:cn");
    memoryKK->grow_kokkos(k_dc6, dc6, "pair:dc6"); 
  }


  d_cn_v  = k_cn_v.template view<DeviceType>();
  d_dc6_v = k_dc6_v.template view<DeviceType>();
   

  // Zero out dc6 and dcn
  if (newton_pair)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3KokkosCNDC6Initialise>(0, nmax), *this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3KokkosCNDC6Initialise>(0, nlocal), *this);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3KokkosCNDC6Calc>(0, inum), *this);  
 
  k_cn.template modify<DeviceType>();
  
  communicationStage = 1;
  if (newton_pair) 
  {
    k_cn_v.template modify<DeviceType>();
    k_dc6_v.template modify<DeviceType>();
    comm->reverse_comm(this);
    k_cn_v.template sync<DeviceType>();
    k_dc6_v.template sync<DeviceType>();
  }  
  else
  {
    k_cn_v.template modify<DeviceType>();
    k_dc6_v.template modify<DeviceType>();
    comm->forward_comm(this);
    k_cn_v.template modify<DeviceType>();
    k_dc6_v.template modify<DeviceType>();
  }
  copymode = 0;
  }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3KokkosCNDC6Initialise, const int &i) const
{
  d_cn[i]  = 0.0;
  d_dc6[i] = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>operator()(TagPairDD3KokkosCNDC6Calc, const int &ii) const
{
  const int     i      = d_ilist[ii];
  const int     itype  = type(i);
  const int     jnum   = d_numneigh[i];
  const F_FLOAT xi     = x(i,0);
  const F_FLOAT yi     = x(i,1);
  const F_FLOAT zi     = x(i,2);

  for (int jj = 0; << j_num; jj++)
  {
    int j                = d_neighbors(i, jj) & NEIGHMASK; 
    const int     jtype  = type(j); 
    const F_FLOAT xj     = x(j,0);
    const F_FLOAT yj     = x(j,1);
    const F_FLOAT zj     = x(j,2);
    
    const F_FLOAT xij    = xi - xj;
    const F_FLOAT yij    = yi - yj;
    const F_FLOAT zij    = zi - zj;

    const F_FLOAT rsq    = xij*xij + yij*yij + zij*zij; 

    // if the atoms are too far away don't consider the contribution
    if (rsq > l_cn_thr) continue; 

    const F_FLOAT rr      = sqrt(rsq);
    const F_FLOAT rcov_ij = (d_rcov_v(itype) + d_rcov_v(jtype)) * autoang ;
    const F_FLOAT cn_ij   = F_FLOAT(1.0) / (F_FLOAT(1.0) + exp(-K1 * ((rcov_ij / rr - F_FLOAT(1.0)))));

    d_cn(i) += cn_ij;
    if (newton_pair || j < nlocal } { Kokkos::atomic_add(&d_cn(j), cn_ij); }
}   


struct DC6Derive
{
  double num, den; // ones to accumulate
  double dnum_i, dnum_j;
  double dden_i, dden_j;
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
  static void join(volatile DC6Derive &dst, const volatile DC6Derive &src)
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
      dst.c6mem  = src.c6mem;
    }
  }

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  static void accumulate_cell( const ViewType &d_c6ab_v,
                               int iat, int jat, int ci, int cj, 
                               double cni, double cnj, DC6Derive &acc)
  {
    double c6_ref         = d_c6ab_v(iat, jat, ci, cj, 0);
    c6_ref               *= autoev * pow(autoang, 6);
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

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::get_dC6KK
(
  const TeamMember &team,
  const    int iat, const    int jat,
  const double cni, const double cnj,
  double &C6, double &dC6_dCNi, double &dC6_dCNj 
) const
{
  const int Ci = d_mxci_v(iat) + 1;
  const int Cj = d_mxci_v(jat) + 1;

  DC6Derive team_acc;
  DC6Derive::init(team_acc);
  
  // Outer loop splits Ci across team threads
  Kokkos::parallel_reduce(
    Kokkos::TeamThreadRange(team, Ci),
    [&](const int ci, DC6Derive &acc_outer)
    {
      DC6Derive row;
      DC6Derive::init(row);
      
      // Inner loop splits Cj across vector lanes 
      Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(team, Cj),
        [&](const int cj, DC6Derive &acc_inner)
        {
          DC6Derive::accumulate_cell(d_c6ab_v, iat, jat, ci, cj, cni, cnj, acc_inner);
        },
        row
       );
       DC6Derive::join(acc_outer, row);
    },
    team_acc
  );
  
  Kokkos::single(Kokkos::PerTeam(team), [&]()
  {
    if (team_acc.den > 1.0e-99)
    {
      C6         = team_acc.num / team_acc.den ; 
      dC6_dCNi   = ((team_acc.dnum_i * team_acc.den) - (team_acc.dden_i * team_acc.num)) / (team_acc.den * team_acc.den) ;
      dC6_dCNj   = ((team_acc.dnum_j * team_acc.den) - (team_acc.dden_j * team_acc.num)) / (team_acc.den * team_acc.den) ;
    }
    else{ C6 = team_acc.c6mem; dC6_dCNi = 0.0; dC6_dCNj = 0.0; }
  });
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag, int vflag)
{
  // --- setup ---
  using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  using TeamMember = typename TeamPolicy::member_type;

  calc_coordination_numbersKK();
  atomKK->sync(exec_space, X_MASK | F_MASK | TYPE_MASK);
  atomKK->modified(exec_space, F_MASK);
  ev_init(eflag, vflag);

  auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh   = k_list->d_numneigh;
  d_neighbors  = k_list->d_neighbors;
  d_ilist      = k_list->d_ilist;
  inum         = list->inum;

  d_x    = atomKK->k_x.view<DeviceType>();
  d_f    = atomKK->k_f.view<DeviceType>();
  d_type = atomKK->k_type.view<DeviceType>();

  // clear dc6
  Kokkos::deep_copy(k_dc6_v, 0.0);

  // -------------------------
  // Stage 1: dE/d(ij) + dc6
  // -------------------------
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
  k_cn_v.template modify<DeviceType>();
  k_dc6_v.template modify<DeviceType>();
  if (newton_pair) {
    comm->reverse_comm(this);
  } else {
    comm->forward_comm(this);
  }
  k_cn_v.template sync<DeviceType>();
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
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

// EV version (TeamPolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  const TagPairDispDD3dEdIJ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>&,
  const TeamMember& team, EV_FLOAT& ev) const
{
  const int ii = team.league_rank();
  if (ii >= inum) return;

  const int i = d_ilist[ii];
  const double xi = d_x(i,0);
  const double yi = d_x(i,1);
  const double zi = d_x(i,2);
  const int itype = d_type(i);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) {
    const int jfull = d_neighbors(i,jj);
    const double factor_lj = d_special_lj[sbmask(jfull)];
    const int j    = jfull & NEIGHMASK;
    const int jtype = d_type(j);

    const double dx  = xi - d_x(j,0);
    const double dy  = yi - d_x(j,1);
    const double dz  = zi - d_x(j,2);
    const double rsq = dx*dx + dy*dy + dz*dz;

    if (rsq < d_cutsq(itype,jtype) && rsq > 0.0) {
      const double r      = sqrt(rsq);
      const double r2inv  = 1.0 / rsq;
      const double r4inv  = r2inv * r2inv;
      const double r6inv  = r4inv * r2inv;
      const double r8inv  = r6inv * r2inv;
      const double r10inv = r8inv * r2inv;

      const double cni = d_cn_v(i);
      const double cnj = d_cn_v(j);

      double C6 = 0.0, dC6_i = 0.0, dC6_j = 0.0;
      // Use your team-parallel interpolation/derivatives
      get_dC6KK(team, itype, jtype, cni, cnj, C6, dC6_i, dC6_j);
      if (C6 == 0.0) continue;

      const double C8 = 3.0 * C6 * d_r2r4_v(itype) * d_r2r4_v(jtype) * (autoang * autoang);

      // Dimensionless r0 and D3 exponents
      const double r0     = r / d_r0ab_v(itype, jtype);
      const double alpha6 = alpha;
      const double alpha8 = alpha + 2.0;

      // t_n = (rs_n / r0)^{alpha_n}, damp_n = 1 / (1 + n * t_n)
      const double t6    = pow(rs6 / r0, alpha6);
      const double damp6 = 1.0 / (1.0 + 6.0 * t6);
      const double t8    = pow(rs8 / r0, alpha8);
      const double damp8 = 1.0 / (1.0 + 8.0 * t8);

      // Energies without s6/s8/sign
      const double e6 = C6 * damp6 * r6inv;
      const double e8 = C8 * damp8 * r8inv;

      // Force pieces
      // tmp6 = 6 s6 C6 r^-8 damp6, tmp8 = 8 s8 C8 r^-10 damp8
      const double tmp6 = 6.0 * s6 * C6 * r8inv  * damp6;
      const double tmp8 = 8.0 * s8 * C8 * r10inv * damp8;

      // fpair (per-component multiplier): fpair = (s6 de6/dr + s8 de8/dr) / r
      // = [ -(tmp6 + tmp8)            ] + ddamp contributions:
      //   + [ tmp6 * alpha6 * t6 * damp6 ] + [ tmp8 * alpha8 * t8 * damp8 ]
      const double fpair_no_damp = -(tmp6 + tmp8);
      const double fpair_damp    =  (tmp6 * alpha6 * t6 * damp6)
                                  + (tmp8 * alpha8 * t8 * damp8);
      const double fpair = (fpair_no_damp + fpair_damp) * factor_lj;

      // Pair energy (with sign and s6/s8)
      const double phi = -(s6 * e6 + s8 * e8) * factor_lj;

      // dE/dC6 to propagate via dC6/dCN
      // dE/dC6 = -[ s6 * e6/C6 + s8 * e8/C6 ] since C8 ∝ C6
      const double rest = -(s6 * e6 + s8 * e8) / C6;
      Kokkos::atomic_add(&d_dc6_v(i), rest * dC6_i);
      if (NEWTON_PAIR || j < nlocal) {
        Kokkos::atomic_add(&d_dc6_v(j), rest * dC6_j);
      }

      // Accumulate i forces in registers
      fix += dx * fpair;
      fiy += dy * fpair;
      fiz += dz * fpair;

      // Accumulate j forces atomically
      if (NEWTON_PAIR || j < nlocal) {
        Kokkos::atomic_add(&d_f(j,0), -dx * fpair);
        Kokkos::atomic_add(&d_f(j,1), -dy * fpair);
        Kokkos::atomic_add(&d_f(j,2), -dz * fpair);
      }

      if (EVFLAG) {
        if (eflag) {
          const double weight =
            ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && (NEWTON_PAIR || j < nlocal)) ? 1.0 : 0.5;
          ev.evdwl += weight * phi;
        }
        if (vflag_either || eflag_atom) {
          this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, phi, fpair, dx, dy, dz);
        }
      }
    }
  }

  // Write back i's force with one atomic per component
  Kokkos::atomic_add(&d_f(i,0), fix);
  Kokkos::atomic_add(&d_f(i,1), fiy);
  Kokkos::atomic_add(&d_f(i,2), fiz);
}

// NO-EV thin wrapper (TeamPolicy)
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  const TagPairDispDD3dEdIJ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>& tag,
  const TeamMember& team) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(tag, team, ev);
}



template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  const TagPairDispDD3dEdXYZ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>&,
  const int& ii, EV_FLOAT& ev) const
{
  const int i = d_ilist[ii];
  const double xi = d_x(i,0);
  const double yi = d_x(i,1);
  const double zi = d_x(i,2);
  const int itype = d_type(i);

  double fix = 0.0, fiy = 0.0, fiz = 0.0;

  const int jnum = d_numneigh[i];
  for (int jj = 0; jj < jnum; ++jj) 
  {
    const int jfull = d_neighbors(i,jj);
    const double factor_lj = d_special_lj[sbmask(jfull)];
    const int j = jfull & NEIGHMASK;
    const int jtype = d_type(j);

    const double dx = xi - d_x(j,0);
    const double dy = yi - d_x(j,1);
    const double dz = zi - d_x(j,2);
    const double rsq = dx*dx + dy*dy + dz*dz;

    if (rsq < d_cutsq(itype,jtype) && rsq > 0.0) 
    {
      const double r      = sqrt(rsq);
      
      if (rsq < cn_thr)
      { 
        const double rcovij  = (d_rcov_v(itype) + d_rcov_v(jtype))*autoang;
        const double expterm = exp(-K1 * (rcovij / (r - 1.0)));
        const double dcn     = -K1 * rcovij * expterm / (rsq * (expterm + 1.0) * (expterm + 1.0));
      }
      else dcn = 0.0 ; 

      const double fpair1 = dcn * (d_dc6_v(i) + d_dc6_v(j)) / r ; 
      const double fpair  = fpair1*factor_lj;
     
      // i forces in registers
      fix += dx * fpair;
      fiy += dy * fpair;
      fiz += dz * fpair;

      // j forces atomically
      if (NEWTON_PAIR || j < nlocal) 
      {
        Kokkos::atomic_add(&d_f(j,0), -dx * fpair);
        Kokkos::atomic_add(&d_f(j,1), -dy * fpair);
        Kokkos::atomic_add(&d_f(j,2), -dz * fpair);
      }

      // EV tally
      if (EVFLAG) { this->template ev_tally<NEIGHFLAG,NEWTON_PAIR>(ev, i, j, 0.0, fpair, dx, dy, dz); }   
      }
    }
  // one atomic per component for i
  Kokkos::atomic_add(&d_f(i,0), fix);
  Kokkos::atomic_add(&d_f(i,1), fiy);
  Kokkos::atomic_add(&d_f(i,2), fiz);
}

// Thin wrapper: NO-EV kernel
template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(
  const TagPairDispDD3dEdXYZ<NEIGHFLAG,NEWTON_PAIR,EVFLAG>& tag,
  const int& ii) const
{
  EV_FLOAT ev; // unused if EVFLAG==0
  this->template operator()<NEIGHFLAG,NEWTON_PAIR,EVFLAG>(tag, ii, ev);
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
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackForwardCommDC6>(0,n), *this);
  }
  return n;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackForwardCommCN, const int &i) const
{
  int j = d_sendlistV(i);
  bufV(i) = d_cnV(j);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackForwardCommDC6, const int &i) const
{
  int j = d_sendlistV(i);
  bufV(i) = d_dc6V(j);
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
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackForwardCommDC6>(0,n), *this);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackForwardCommCN, const int &i) const
{
  d_cnV(i + first) = bufV(i);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackForwardCommDC6, const int &i) const
{
  d_dc6V(i + first) = bufV(i);
}




// reverse pack 


template<class DeviceType>
int PairDispersionD3Kokkos<DeviceType>::pack_reverse_comm_kokkos(int n,
                                                                 int first_in,
                                                                 DAT::tdual_xfloat_1d &buf)
{
  bufV = buf.view<DeviceType>();
  //d_cn       = k_cn.view<DeviceType>();
  //d_dc6      = k_dc6.view<DeviceType>();
  //d_cn  = k_cn.template view<DeviceType>();
  //d_dc6 = d_dc6.template view<DeviceType>();

  first = first_in;

  if (communicationStage == 1)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackReverseCommCN>(0,n), *this);
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairDD3PackReverseCommDC6>(0,n), *this);
  }
  return n;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackReverseCommCN, const int &i) const
{
  bufV(i)    = d_cnV(i + first);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3PackReverseCommDC6, const int &i) const
{
  bufV(i)    = d_dc6V(i + first);
}


// reverse unpack 


template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::unpack_reverse_comm_kokkos(int n,
                                                                    DAT::tdual_int_1d k_sendlistV,
                                                                    DAT::tdual_xfloat_1d &buf)
{
  d_sendlistV = k_sendlistV.view<DeviceType>();
  bufV        = buf.view<DeviceType>();
  //d_cn       = k_cn.view<DeviceType>();
  //d_dc6      = k_dc6.view<DeviceType>();
  //d_cn  = k_cn.template view<DeviceType>();
  //d_dc6 = d_dc6.template view<DeviceType>();

  if (communicationStage == 1)
  {
    Kokkos::parallel_for( Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackReverseCommCN>(0,n), *this);
  }
  if (communicationStage == 2)
  {
    Kokkos::parallel_for( Kokkos::RangePolicy<DeviceType, TagPairDD3UnpackReverseCommDC6>(0,n), *this);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackReverseCommCN, const int &i) const
{
  const int j = d_sendlistV(i);
  Kokkos::atomic_add(&d_cnV(j), bufV(i));
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairDispersionD3Kokkos<DeviceType>::operator()(TagPairDD3UnpackReverseCommDC6, const int &i) const
{
  const int j = d_sendlistV(i);
  Kokkos::atomic_add(&d_dc6V(j), bufV(i));
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
    k_cnV.sync_host();
    int i,j;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      buf[i] = h_cnV[j];
    }
  }
  if (communicationStage == 2)
  {
    k_dc6V.sync_host();
    int i,j;

    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      buf[i] = h_dc6V[j];
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
    k_cnV.sync_host();
    for (int i = 0; i < n; i++ )
    {
      h_cnV[i +first] = buf[i];
    }
    k_cnV.modify_host();
  }
  if (communicationStage == 2)
  {
    k_dc6V.sync_host();
    for (int i = 0; i < n; i++ )
    {
      h_dc6V[i +first] = buf[i];
    }
    k_dc6V.modify_host();
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
    k_cnV.sync_host();

    int i, last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++) { buf[m++] = h_cnV[i]; }
  }

  if (communicationStage == 2)
  {
    k_dc6V.sync_host();

    int i,last;

    m = 0;
    last = first + n;
    for ( i = first; i < last; i++) { buf[m++] = h_dc6V[i]; }
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
    k_cnV.sync_host();

    int i,j,m;

    m = 0;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      h_cnV[j] += buf[m++];
    }
    k_cnV.modify_host();
  }

  if (communicationStage == 2)
  {
    k_dc6V.sync_host();

    int i,j,m;

    m = 0;
    for ( i = 0; i < n; i++ )
    {
      j = list[i];
      h_dc6V[j] += buf[m++];
    }
    k_dc6V.modify_host();
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





