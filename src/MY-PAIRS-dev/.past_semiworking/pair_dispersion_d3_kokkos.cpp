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
#define CHECK_DUALVIEW_DEVICE_ALLOC(dv, name) \
  if (!(dv.view_device().data())) { \
    std::ostringstream oss; \
    oss << "Device view for '" << name << "' is uninitialized.\n" \
        << "Triggered at " << __FILE__ << ":" << __LINE__; \
    throw std::runtime_error(oss.str()); \
  }



using namespace LAMMPS_NS;
static constexpr int NUM_ELEMENTS=94;
static constexpr int N_PARS_COLS=5;  // number columns in C6 table
static constexpr int N_PARS_ROWS=32395; // number of rows C6 table

#include <d3_parameters.h>

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::PairDispersionD3Kokkos(LAMMPS *lmp)
:PairDispersionD3(lmp)
{
  comm_forward = 2;
  comm_reverse = 2;
  
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom ; 
//  Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType> k_params;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
  damping_type = "zero_damping";
}

struct DC6 
{
  double C6;
  double dC6i;
  double dC6j;

  KOKKOS_INLINE_FUNCTION
  DC6() : C6(0), dC6i(0), dC6j(0) {}
};

template<class DeviceType, class MXCI_View, class C6AB_View>
KOKKOS_INLINE_FUNCTION
DC6 get_dC6_res(
    int iat, int jat,
    double cni, double cnj,
    const MXCI_View  &d_mxciV,
    const C6AB_View  &d_c6abV)
{
  DC6 out;

  double c6_ref, cni_ref, cnj_ref;
  double c6mem  = -1.0e20;
  double r_save =  1.0e20;
  double r, expterm, term;
  double num = 0.0, den = 0.0;
  double KKnum_i = 0.0, KKnum_j = 0.0;
  double KKden_i = 0.0, KKden_j = 0.0;

  for (int ci = 0; ci <= d_mxciV(iat); ci++) {
    for (int cj = 0; cj <= d_mxciV(jat); cj++) {
      c6_ref = d_c6abV(iat, jat, ci, cj, 0) * autoev * pow(autoang, 6);

      if (c6_ref > 0.0) {
        cni_ref = d_c6abV(iat, jat, ci, cj, 1);
        cnj_ref = d_c6abV(iat, jat, ci, cj, 2);

        r = (cni - cni_ref) * (cni - cni_ref) +
            (cnj - cnj_ref) * (cnj - cnj_ref);

        if (r < r_save) { r_save = r; c6mem = c6_ref; }

        expterm = exp(K3 * r);
        num += c6_ref * expterm;
        den += expterm;

        expterm = expterm * 2.0 * K3;

        term     = expterm * (cni - cni_ref);
        KKnum_i += c6_ref * term;
        KKden_i += term;

        term     = expterm * (cnj - cnj_ref);
        KKnum_j += c6_ref * term;
        KKden_j += term;
      }
    }
  }

  if (den > 1.0E-99) {
    out.C6   = num / den;
    out.dC6i = ((KKnum_i * den) - (KKden_i * num)) / (den * den);
    out.dC6j = ((KKnum_j * den) - (KKden_j * num)) / (den * den);
  } else {
    out.C6   = c6mem;
    out.dC6i = 0.0;
    out.dC6j = 0.0;
  }
  return out;
}

template <class DualViewType, class ExecSpace, class FillerFunc>
inline void init_dualview_from_host(DualViewType &dv, const FillerFunc &filler)
{
  auto hv = dv.h_view;
  using HostView = std::decay_t<decltype(hv)>;
  constexpr int rank = HostView::rank;

  if constexpr (rank == 1) {
    for (size_t i = 0; i < hv.extent(0); ++i) filler(hv, static_cast<int>(i));
  } else if constexpr (rank == 2) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        filler(hv, static_cast<int>(i), static_cast<int>(j));
  } else if constexpr (rank == 3) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          filler(hv, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k));
  } else if constexpr (rank == 4) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          for (size_t l = 0; l < hv.extent(3); ++l)
            filler(hv, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k), static_cast<int>(l));
  } else if constexpr (rank == 5) {
    for (size_t i = 0; i < hv.extent(0); ++i)
      for (size_t j = 0; j < hv.extent(1); ++j)
        for (size_t k = 0; k < hv.extent(2); ++k)
          for (size_t l = 0; l < hv.extent(3); ++l)
            for (size_t m = 0; m < hv.extent(4); ++m)
              filler(hv, static_cast<int>(i), static_cast<int>(j), static_cast<int>(k),
                         static_cast<int>(l), static_cast<int>(m));
  } else {
    static_assert(rank <= 5, "Rank > 5 not implemented");
  }

  dv.modify_host();
  dv.template sync<ExecSpace>();
}


template<class DeviceType>
struct ev_tally {
  using AT = ArrayTypes<DeviceType>;
  using exec_space = typename DeviceType::execution_space;
  using View1D_EF = typename AT::t_efloat_1d;
  using View2D_EF = typename AT::t_efloat_2d;
  using View1D_EFLOAT_6 = Kokkos::View<E_FLOAT[6], DeviceType>; 
  const EV_FLOAT ev; 
  View1D_EFLOAT_6  d_vglobalV; // <--- accompanies EV_FLOAT &ev

  const int i, j;
  const F_FLOAT epair, fpair, delx, dely, delz;
  const int nlocal, l_newtonpair, l_neighflag;
  const int eflag, eflag_atom, vflag_either, vflag_global, vflag_atom;
  View1D_EF d_eatom;
  View2D_EF d_vatom;

  KOKKOS_INLINE_FUNCTION
  ev_tally() = default;
  KOKKOS_INLINE_FUNCTION ev_tally(
  const EV_FLOAT &ev_, View1D_EFLOAT_6 d_vglobalV_,
  const int &i_, const int &j_,
  const F_FLOAT &epair_, const F_FLOAT &fpair_,
  const F_FLOAT &delx_, const F_FLOAT &dely_, const F_FLOAT &delz_,
  const int &nlocal_, const int &l_newtonpair_, const int &l_neighflag_,
  const int &eflag_, const int &eflag_atom_, const int &vflag_either_,
  const int &vflag_global_, const int &vflag_atom_,
  View1D_EF d_eatom_, View2D_EF d_vatom_)
: ev(ev_), d_vglobalV(d_vglobalV_),
  i(i_), j(j_), epair(epair_), fpair(fpair_), delx(delx_), dely(dely_), delz(delz_),
  nlocal(nlocal_), l_newtonpair(l_newtonpair_), l_neighflag(l_neighflag_),
  eflag(eflag_), eflag_atom(eflag_atom_),
  vflag_either(vflag_either_), vflag_global(vflag_global_), vflag_atom(vflag_atom_),
  d_eatom(d_eatom_), d_vatom(d_vatom_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()() const {
    
    const int NEWTON_PAIR = l_newtonpair;
    
    if (eflag) {
      const E_FLOAT epairhalf = 0.5 * epair;
      if (l_neighflag || i < nlocal) d_eatom(i) += epairhalf;
      if (l_neighflag || j < nlocal) d_eatom(j) += epairhalf;
    }
    if (vflag_either) {
      const E_FLOAT v0 = delx * delx * fpair;
      const E_FLOAT v1 = dely * dely * fpair;
      const E_FLOAT v2 = delz * delz * fpair;
      const E_FLOAT v3 = delx * dely * fpair;
      const E_FLOAT v4 = delx * delz * fpair;
      const E_FLOAT v5 = dely * delz * fpair;
      if (vflag_global) {
        if (NEWTON_PAIR || i < nlocal) {
          Kokkos::atomic_add(&d_vglobalV(0), 0.5 * v0);
          Kokkos::atomic_add(&d_vglobalV(1), 0.5 * v1);
          Kokkos::atomic_add(&d_vglobalV(2), 0.5 * v2);
          Kokkos::atomic_add(&d_vglobalV(3), 0.5 * v3);
          Kokkos::atomic_add(&d_vglobalV(4), 0.5 * v4);
          Kokkos::atomic_add(&d_vglobalV(5), 0.5 * v5);
        }
        if (NEWTON_PAIR || j < nlocal) {
          Kokkos::atomic_add(&d_vglobalV(0), 0.5 * v0);
          Kokkos::atomic_add(&d_vglobalV(1), 0.5 * v1);
          Kokkos::atomic_add(&d_vglobalV(2), 0.5 * v2);
          Kokkos::atomic_add(&d_vglobalV(3), 0.5 * v3);
          Kokkos::atomic_add(&d_vglobalV(4), 0.5 * v4);
          Kokkos::atomic_add(&d_vglobalV(5), 0.5 * v5);
        }
      }
      if (vflag_atom) {
        if (NEWTON_PAIR || i < nlocal) {
          d_vatom(i, 0) += 0.5 * v0;
          d_vatom(i, 1) += 0.5 * v1;
          d_vatom(i, 2) += 0.5 * v2;
          d_vatom(i, 3) += 0.5 * v3;
          d_vatom(i, 4) += 0.5 * v4;
          d_vatom(i, 5) += 0.5 * v5;
        }
        if (NEWTON_PAIR || j < nlocal) {
          d_vatom(j, 0) += 0.5 * v0;
          d_vatom(j, 1) += 0.5 * v1;
          d_vatom(j, 2) += 0.5 * v2;
          d_vatom(j, 3) += 0.5 * v3;
          d_vatom(j, 4) += 0.5 * v4;
          d_vatom(j, 5) += 0.5 * v5;
        }
      }
    }
  }
};


//template<class DeviceType, int NEIGHFLAG, int NEWTON_PAIR>
template<class DeviceType>
struct PairDispD3Kernel_dEdIJ {
  using exec_space      = typename DeviceType::execution_space;
  using policy_type     = Kokkos::RangePolicy<exec_space>;
  using index_type      = typename policy_type::index_type;
  using AT              = ArrayTypes<DeviceType>;
  using View1D_EF       = typename AT::t_efloat_1d;
  using View2D_EF       = typename AT::t_efloat_2d;
  using View2D_neigh    = typename AT::t_neighbors_2d;
  using View2D_F        = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using View1D_F        = Kokkos::View<F_FLOAT*, DeviceType>;
  using View1D_int      = Kokkos::View<int*, DeviceType>;
  using View2D_params   = Kokkos::View<params_d3**, Kokkos::LayoutRight, DeviceType>;
  using View5D_c6ab     = Kokkos::View<double*****, Kokkos::LayoutRight, DeviceType>;
  using View1D_EFLOAT_6 = Kokkos::View<E_FLOAT[6], DeviceType>; 
  // views
  View2D_F      d_x;
  View2D_F      d_f;
  View1D_F      d_cnV;
  View1D_F      d_dc6V;
  View1D_int    d_typeV;
  Few<int, 4>   f_special_lj;
  View1D_int    d_mxciV;
  View5D_c6ab   d_c6abV;
  View2D_params d_paramsV;
  View1D_F      d_r2r4V;
  View1D_int    d_ilistV;
  View1D_int    d_numneighV;
  View2D_neigh  d_neighborsV;

  // scalars
  const int     l_eflag;
  const int     l_eflagatom;
  const int     l_newton_pair;
  const int     l_nlocal;
  const int     l_neighflag; 
  F_FLOAT l_autoang;
  // for virial
  const EV_FLOAT      ev;
  View1D_EFLOAT_6     d_vglobalV; 
  const int vflag_either, vflag_global, vflag_atom;
  View1D_EF d_eatom;
  View2D_EF d_vatom;

  PairDispD3Kernel_dEdIJ(
    View2D_F         d_x_,
    View2D_F         d_f_,
    View1D_F         d_cnV_,
    View1D_F         d_dc6V_,
    View1D_int       d_typeV_,
    Few<int, 4>      const& f_special_lj_,
    View1D_int       d_mxciV_,
    View5D_c6ab      d_c6abV_,
    View2D_params    d_paramsV_,
    View1D_F         d_r2r4V_,
    View1D_int       d_ilistV_,
    View1D_int       d_numneighV_,
    View2D_neigh     d_neighborsV_,
    int              const& l_eflag_,
    int              const& l_eflagatom_,
    int              const& l_newton_pair_,
    int              const& l_nlocal_,
    int              const& l_neighflag_,
    F_FLOAT          const& l_autoang_,
    View1D_EFLOAT_6  d_vglobalV_,
    EV_FLOAT         const& ev_,
    int              const& vflag_either_,
    int              const& vflag_global_,
    int              const& vflag_atom_,
    View1D_EF        d_eatom_,
    View2D_EF        d_vatom_)
  : d_x(d_x_), d_f(d_f_), d_cnV(d_cnV_), d_dc6V(d_dc6V_), d_typeV(d_typeV_),
    f_special_lj(f_special_lj_), d_mxciV(d_mxciV_), d_c6abV(d_c6abV_), d_paramsV(d_paramsV_),
    d_r2r4V(d_r2r4V_), d_ilistV(d_ilistV_), d_numneighV(d_numneighV_),
    d_neighborsV(d_neighborsV_), l_eflag(l_eflag_), l_eflagatom(l_eflagatom_), l_newton_pair(l_newton_pair_),
    l_nlocal(l_nlocal_), l_neighflag(l_neighflag_), l_autoang(l_autoang_),
    d_vglobalV(d_vglobalV_), ev(ev_), vflag_either(vflag_either_),
    vflag_global(vflag_global_), vflag_atom(vflag_atom_), 
    d_eatom(d_eatom_), d_vatom(d_vatom_) 
  {}

  //template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type &ii) const {
    const int   i     = d_ilistV[ii];
    if (i >= l_nlocal) return;
    if (i < 0 || i >= d_x.extent_int(0)) Kokkos::abort("bad i");
    const int   itype = d_typeV(i);
    const auto  icn   = d_cnV(i);
    const int   jnum  = d_numneighV(ii);
    //const auto  neigh_i = k_list->get_neighbors(i);

    for (int jj = 0; jj < jnum; jj++) {
      const int jenc   = d_neighborsV(ii,jj);
      const int j      = jenc & NEIGHMASK;
      if (j < 0 || j >= d_x.extent_int(0)) continue;
      const int sbmask = jenc >> SBBITS;
      const int jtype  = d_typeV(j);
      const auto jcn   = d_cnV(j);
      const F_FLOAT factor = f_special_lj[sbmask];

      const F_FLOAT delx = d_x(i,0) - d_x(j,0);
      const F_FLOAT dely = d_x(i,1) - d_x(j,1);
      const F_FLOAT delz = d_x(i,2) - d_x(j,2);
      const F_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

      const auto &p = d_paramsV(itype, jtype);
      if (rsq < p.cutsq) {
        const DC6 t_dc6_res = get_dC6_res<DeviceType>(itype, jtype, icn, jcn, d_mxciV, d_c6abV);
        const F_FLOAT C6 = t_dc6_res.C6;
        const F_FLOAT C8 = F_FLOAT(3.0) * C6 * d_r2r4V(itype) * d_r2r4V(jtype) * l_autoang * l_autoang;

        const F_FLOAT r        = sqrt(rsq);
        const F_FLOAT r2inv    = F_FLOAT(1.0) / rsq;
        const F_FLOAT r6inv    = r2inv * r2inv * r2inv;
        const F_FLOAT r8inv    = r6inv * r2inv;

        const F_FLOAT r0       = r / p.r0_ab;
        const F_FLOAT alpha6   = p.alpha;
        const F_FLOAT alpha8   = alpha6 + F_FLOAT(2.0);

        const F_FLOAT t6    = pow(p.rs6 / r0, alpha6);
        const F_FLOAT damp6 = F_FLOAT(1.0) / (F_FLOAT(1.0) + F_FLOAT(6.0) * t6);

        const F_FLOAT t8    = pow(p.rs8 / r0, alpha8);
        const F_FLOAT damp8 = F_FLOAT(1.0) / (F_FLOAT(1.0) + F_FLOAT(6.0) * t8);

        const F_FLOAT e6    = C6 * damp6 * r6inv;
        const F_FLOAT e8    = C8 * damp8 * r8inv;

        const F_FLOAT tmp6  = F_FLOAT(6.0) * p.s6 * C6 * r2inv * r6inv * damp6;
        const F_FLOAT tmp8  = F_FLOAT(8.0) * p.s8 * C8 * r2inv * r8inv * damp8;

        const F_FLOAT fpair1 = -tmp6 - tmp8;
        const F_FLOAT fpair2 = tmp6 * alpha6 * t6 * damp6 + (F_FLOAT(3.0)/F_FLOAT(4.0)) * tmp8 * alpha8 * t8 * damp8;
        const F_FLOAT fpair  = (fpair1 + fpair2) * factor;

        // energy tally (local)
        F_FLOAT l_evdwl = F_FLOAT(0.0);
        if (l_eflag) l_evdwl = -(p.s6 * e6 + p.s8 * e8) * factor;

        const F_FLOAT rest = (p.s6 * e6 + p.s8 * e8) / C6;

        Kokkos::atomic_add(&d_dc6V(i), rest * t_dc6_res.dC6i);
        if (l_newton_pair || j < l_nlocal)
          Kokkos::atomic_add(&d_dc6V(j), rest * t_dc6_res.dC6j);

        // forces
        const F_FLOAT fx = fpair * delx;
        const F_FLOAT fy = fpair * dely;
        const F_FLOAT fz = fpair * delz;

        d_f(i,0) += fx;  d_f(i,1) += fy;  d_f(i,2) += fz;
        Kokkos::atomic_add(&d_f(j,0), -fx);
        Kokkos::atomic_add(&d_f(j,1), -fy);
        Kokkos::atomic_add(&d_f(j,2), -fz);

        if (l_eflag || vflag_either)
        {
          ev_tally<DeviceType> tally{
                                 ev, d_vglobalV,
                                 i, j, l_evdwl, fpair, delx, dely, delz,
                                 l_nlocal, l_newton_pair, l_neighflag, // <-- swap order
                                 l_eflag, l_eflagatom,                // <-- add eflag_atom
                                 vflag_either, vflag_global, vflag_atom,
                                 d_eatom, d_vatom};
                               tally();
        }
      }
    }
  }
};

//template<class DeviceType, int NEIGHFLAG, int NEWTON_PAIR>
template<class DeviceType>
struct PairDispD3Kernel_dEdXYZ {
  using exec_space      = typename DeviceType::execution_space;
  using policy_type     = Kokkos::RangePolicy<exec_space>;
  using index_type      = typename policy_type::index_type;
  using AT = ArrayTypes<DeviceType>;  
  using View1D_EF       = typename AT::t_efloat_1d;
  using View2D_EF       = typename AT::t_efloat_2d;
  using View2D_neigh    = typename AT::t_neighbors_2d; 
  using View2D_F        = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using View1D_F        = Kokkos::View<F_FLOAT*, DeviceType>;
  using View1D_int      = Kokkos::View<int*, DeviceType>;
  using View2D_params   = Kokkos::View<params_d3**, Kokkos::LayoutRight, DeviceType>;
  using View1D_EFLOAT_6 = Kokkos::View<E_FLOAT[6], DeviceType>; 
  // views
  View2D_F      d_x;
  View2D_F      d_f;
  View1D_F      d_dc6V;
  View1D_int    d_typeV;
  Few<int,4>    f_special_lj;
  View1D_F      d_rcovV;
  View2D_params d_paramsV;   // to reuse per-type cutoffs

  // neighbor list
  View1D_int    d_ilistV;
  View1D_int    d_numneighV;
  View2D_neigh  d_neighborsV;

  // scalars
  const int     l_newton_pair;
  const int     l_evflag;
  const int     l_nlocal;
  const int     l_neighflag;
  F_FLOAT l_autoang;
  F_FLOAT l_cn_thr;
  F_FLOAT K1;
  //for virial
  EV_FLOAT         ev;
  View1D_EFLOAT_6  d_vglobalV; 
  const int  vflag_either, vflag_global, vflag_atom;
  View1D_EF  d_eatom;
  View2D_EF  d_vatom;

  PairDispD3Kernel_dEdXYZ(
    View2D_F          d_x_,
    View2D_F          d_f_,
    View1D_F          d_dc6V_,
    View1D_int        d_typeV_,
    Few<int, 4>       const&  f_special_lj_,
    View1D_F          d_rcovV_,
    View2D_params     d_paramsV_,
    View1D_int        d_ilistV_,
    View1D_int        d_numneighV_,
    View2D_neigh      d_neighborsV_, 
    int               const& l_newton_pair_,
    int               const& l_evflag_,
    int               const& l_nlocal_,
    int               const& l_neighflag_,
    F_FLOAT           const& l_autoang_,
    F_FLOAT           const& l_cn_thr_,
    F_FLOAT           const& K1_,
    View1D_EFLOAT_6   d_vglobalV_,
    EV_FLOAT          const& ev_,
    int               const& vflag_either_,
    int               const& vflag_global_,
    int               const& vflag_atom_,
    View1D_EF         d_eatom_,
    View2D_EF         d_vatom_)
  : d_x(d_x_), d_f(d_f_), d_dc6V(d_dc6V_), d_typeV(d_typeV_), f_special_lj(f_special_lj_),
    d_rcovV(d_rcovV_), d_paramsV(d_paramsV_), d_ilistV(d_ilistV_), d_numneighV(d_numneighV_),
    d_neighborsV(d_neighborsV_), l_newton_pair(l_newton_pair_), l_evflag(l_evflag_),
    l_nlocal(l_nlocal_), l_neighflag(l_neighflag_), l_autoang(l_autoang_), l_cn_thr(l_cn_thr_), K1(K1_),
    d_vglobalV(d_vglobalV_), ev(ev_), vflag_either(vflag_either_),
    vflag_global(vflag_global_), vflag_atom(vflag_atom_), 
    d_eatom(d_eatom_), d_vatom(d_vatom_)
  {}

  //template<int NEIGHFLAG, int NEWTON_PAIR>
  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type &ii) const {
    const int i     = d_ilistV[ii];
    if (i >= l_nlocal) return;
    if (i < 0 || i >= d_x.extent_int(0)) Kokkos::abort("bad i") ;
    const int itype = d_typeV(i);
    const int jnum  = d_numneighV(ii);
    //const auto neigh_i   = k_list->get_neighbors(i);
    const F_FLOAT rcov_i = d_rcovV(itype);

    for (int jj = 0; jj < jnum; jj++) {
      const int jenc   = d_neighborsV(ii,jj);
      const int j      = jenc & NEIGHMASK;
      if (j < 0 || j >= d_x.extent_int(0)) continue;
      const int sbmask = jenc >> SBBITS;
      const int jtype  = d_typeV(j);
      const F_FLOAT rcov_j = d_rcovV(jtype);
      const F_FLOAT factor = f_special_lj[sbmask];

      const F_FLOAT delx = d_x(i,0) - d_x(j,0);
      const F_FLOAT dely = d_x(i,1) - d_x(j,1);
      const F_FLOAT delz = d_x(i,2) - d_x(j,2);
      const F_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

      const auto &p = d_paramsV(itype, jtype); // use same per-type cutoffs
      if (rsq < p.cutsq) {
        F_FLOAT dcn = F_FLOAT(0.0);
        if (rsq < l_cn_thr) {
          const F_FLOAT rcovij  = (rcov_i + rcov_j) * l_autoang;
          const F_FLOAT invr    = F_FLOAT(1.0) / sqrt(rsq);
          const F_FLOAT expterm = exp(-K1 * (rcovij * invr - F_FLOAT(1.0)));
          const F_FLOAT denom   = (expterm + F_FLOAT(1.0));
          dcn = -K1 * rcovij * expterm / (rsq * denom * denom);
        }

        const F_FLOAT invr  = F_FLOAT(1.0) / sqrt(rsq);
        const F_FLOAT fpair = dcn * (d_dc6V(i) + d_dc6V(j)) * invr * factor;

        const F_FLOAT fx = fpair * delx;
        const F_FLOAT fy = fpair * dely;
        const F_FLOAT fz = fpair * delz;

        d_f(i,0) += fx; d_f(i,1) += fy; d_f(i,2) += fz;

        if (l_newton_pair || j < l_nlocal) {
          Kokkos::atomic_add(&d_f(j,0), -fx);
          Kokkos::atomic_add(&d_f(j,1), -fy);
          Kokkos::atomic_add(&d_f(j,2), -fz);
        }

        if (l_evflag || vflag_either) 
        {
          ev_tally<DeviceType> tally{
                                 ev, d_vglobalV,
                                 i, j, F_FLOAT(0.0), fpair, delx, dely, delz,
                                 l_nlocal, l_newton_pair, l_neighflag,
                                 /*eflag*/ 0, /*eflag_atom*/ 0,
                                 vflag_either, vflag_global, vflag_atom,
                                 d_eatom, d_vatom};
                               tally();
        }
      }
    }
  }
};

template<class DeviceType>
PairDispersionD3Kokkos<DeviceType>::~PairDispersionD3Kokkos()
{
  if (copymode) return;
  if (allocated) 
  {
    //memoryKK->destroy_kokkos(k_eatom, eatom);
    //memoryKK->destroy_kokkos(k_vatom, vatom);
    //memoryKK->destroy_kokkos(k_cutsq, cutsq); 
    //memoryKK->destroy_kokkos(k_r0ab, r0ab);
    //memoryKK->destroy_kokkos(k_c6ab, c6ab);
    memoryKK->destroy_kokkos(k_r2r4V, r2r4);
    memoryKK->destroy_kokkos(k_rcovV, rcov); 
  }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::allocateKK()
{
    // do not recall allocate because it is called when calling 
    // base class PairDispersionD3::coeff() in our coeff
    //PairDispersionD3::allocate();

    const int ntypes = atom->ntypes;
    const int nmax_atoms = atom->nmax;

    // Per-type arrays
    //memoryKK->create_kokkos(k_mxci, mxci, ntypes+1, "pair:mxci");
    //memoryKK->create_kokkos(k_r2r4, r2r4, ntypes+1, "pair:r2r4");
    memoryKK->create_kokkos(k_rcovV, rcov, ntypes+1, "pair:rcov");

    // Per-atom arrays: use nmax, not ntypes
    memoryKK->create_kokkos(k_x, atom->x, nmax_atoms, 3, "pair:x");
    memoryKK->create_kokkos(k_f, atom->f, nmax_atoms, 3, "pair:f");

    // Only create k_type if not already allocated
    if (!k_typeV.view_device().data()) {
        memoryKK->create_kokkos(k_typeV, atom->type, nmax_atoms, "pair:type");
    }
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::settings(int narg, char **arg)
{
  //PairDispersionD3::settings(int narg, char **arg);
  PairDispersionD3::settings(narg, arg); 
  l_rthr = this->rthr;
  l_cn_thr = this->cn_thr;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::coeff(int narg, char **arg)
{
  // Call base class 
  PairDispersionD3::coeff(narg, arg);
  
  tdual_float_1   k_r2r4 = tdual_float_1("k_r2r4", ntypes);
  tdual_float_1   k_rcov = tdual_float_1("k_rcov", 
  tdual_float_2d  k_r0ab = tdual_float_2d("k_r0ab", ntypes, ntypes);
  t_host_float_2d k_r0ab.h_view;

   
  allocateKK();
}

template<class DeviceType>
double PairDispersionD3Kokkos<DeviceType>::init_one(int i, int j) {
  double cut = PairDispersionD3::init_one(i, j);
  //sync_coeffs_to_device();
  // Optionally capture cutsq for this pair here into k_params.h_view(i,j).cutsq
  return cut;
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_style() {
  PairDispersionD3::init_style();

  // Make sure neighflag matches the Kokkos execution space
  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

  if (neighflag == FULL)
    request->enable_full();
  
  //sync_coeffs_to_device();
}

/*
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::init_style() {
  PairDispersionD3::init_style();
 
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                         !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
   
}
*/
template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::sync_coeffs_to_device()
{
  const int ntypes = atom->ntypes;
   
  if (ntypes == 0 || !r2r4 || !mxci || !c6ab) {
    error->all(FLERR,"dispersion/d3/kk: coeff() not called before init_style (arrays unallocated)");
  }

  //allocateKK();
  // Allocate params + 1D r2r4
  k_paramsV    = Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType>("d3_params", ntypes+1, ntypes+1);
  k_r2r4V      = Kokkos::DualView<F_FLOAT*, DeviceType>("r2r4", ntypes+1); 
  k_mxciV      = Kokkos::DualView<int*, DeviceType>("t_mxci", ntypes+1); 
  k_c6abV      = Kokkos::DualView<double*****, Kokkos::LayoutRight, DeviceType>("c6ab", ntypes+1, ntypes+1, 5, 5, 3);
  // ---- fill params on host ----
  {
    auto hvV = k_paramsV.h_view;
    for (int i = 1; i <= ntypes; i++) {
      k_r2r4V.h_view(i) = r2r4[i];
      for (int j = 1; j <= ntypes; j++) {
        auto &p = hvV(i,j);
        p.cutsq = cutsq[i][j];
        p.cut   = std::sqrt(p.cutsq);
        p.s6    = s6;
        p.s8    = s8;
        p.rs6   = rs6;
        p.rs8   = rs8;
        p.alpha = alpha;
        p.r0_ab = r0ab[i][j];
      }
    }
  }
  k_paramsV.modify_host();
  k_paramsV.template sync<DeviceType>();

  // ---- r2r4 via filler ----
  auto fill_r2r4 = [&](auto &hvV, int i) { hvV(i) = r2r4[i]; };
  init_dualview_from_host<decltype(k_r2r4V), typename DeviceType::execution_space>(k_r2r4V, fill_r2r4);
  

  // ---- mxci + c6ab ----
  int gi_max = 0, gj_max = 0;
  for (int t = 1; t <= ntypes; t++) {
    if (mxci[t] < 0 || mxci[t] >= 5)
    {
      error->all(FLERR, "mxci[%d] out of range: %d (must be 0..4)", t, mxci[t]);
    }
    gi_max = std::max(gi_max, mxci[t]);
    gj_max = std::max(gj_max, mxci[t]);
  }
  gi_max++; gj_max++;

  init_dualview_from_host<decltype(k_mxciV), typename DeviceType::execution_space>(k_mxciV, [&](auto &hvV, int i) { hvV(i) = mxci[i]; });

  
  init_dualview_from_host<decltype(k_c6abV), typename DeviceType::execution_space>(k_c6abV, [&](auto &hvV, int i, int j, int gi, int gj, int k) 
  {
    if (gi <= mxci[i] && gj <= mxci[j]) { hvV(i,j,gi,gj,k) = c6ab[i][j][gi][gj][k];} else { hvV(i,j,gi,gj,k) = 0.0 ;} 
  });

}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::calc_coordination_numberKK() {
  // 1) Sync atoms needed
  atomKK->sync(execution_space, datamask_read);

  // 2) Grow per-atom buffers to capacity before taking views
  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    k_cnV  = typename AT::tdual_double_1d("pair:cn",  nmax);
    k_dc6V = typename AT::tdual_double_1d("pair:dc6", nmax);
   
    h_cnV  = k_cnV.h_view;
    h_dc6V = k_dc6V.h_view;
    //d_cn  = k_cn.template view<DeviceType>();
    //d_dc6 = k_dc6.template view<DeviceType>();
  }

  // 3) Get device neighbor list Views/handles
  auto* k_listV = static_cast<NeighListKokkos<DeviceType>*>(list);
  printf("inum=%d, gnum=%d, d_ilist.extent(0)=%d, d_numneigh.extent(0)=%d, d_neighbors.extent(0)=%d\n",
       list ? list->inum : -1,
       list ? list->gnum : -1,
       (int)k_listV->d_ilist.extent(0),
       (int)k_listV->d_numneigh.extent(0),
       (int)k_listV->d_neighbors.extent(0));
  auto d_ilistV       = k_listV->d_ilist;        // View<int* , ... , Device>
  auto d_numneighV    = k_listV->d_numneigh;     // View<int* , ... , Device>
  auto d_neighborsV   = k_listV->d_neighbors;
  //auto d_firstneigh = k_list->d_firstneigh;  // View<int** , ... , Device> (2-D neighbors)
  //if
  // Optional fail-fast checks
  if (!d_ilistV.data() || !d_numneighV.data()) {
    error->all(FLERR, "Device neighbor list not available");
  }
 
  CHECK_DUALVIEW_DEVICE_ALLOC(k_rcovV, "k_rcov"); 
  // Ensure the device-side allocation exists
  k_x.template sync<DeviceType>();
  k_cnV.template sync<DeviceType>();
  k_dc6V.template sync<DeviceType>();
  k_typeV.template sync<DeviceType>();
  k_rcovV.template sync<DeviceType>(); 

 

  // 4) Take device views of your arrays
  CHECK_DUALVIEW_DEVICE_ALLOC(k_x, "k_x");
  //auto d_x    = k_x.template view<DeviceType>();
  auto d_x          = atomKK->k_x.view<DeviceType>();
  auto d_typeV      = atomKK->k_type.view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_cnV, "k_cn");
  auto d_cnV   = k_cnV.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_dc6V, "k_dc6");
  auto d_dc6V  = k_dc6V.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_typeV, "k_type");
  //auto d_type = k_type.template view<DeviceType>();  
  CHECK_DUALVIEW_DEVICE_ALLOC(k_rcovV, "k_rcov");
  auto d_rcovV = k_rcovV.template view<DeviceType>(); 

  const int nlocal      = atom->nlocal;
  const int nall        = nlocal + atom->nghost;
  const int newton_pair = force->newton_pair;
  const int inum        = list->inum;

  copymode = 1;
  
  // 5) Zero
  Kokkos::parallel_for(
    "ZeroCN_DC6",
    Kokkos::RangePolicy<typename DeviceType::execution_space>(0, newton_pair ? nall : nlocal),
    KOKKOS_LAMBDA(const int i) {
      d_cnV(i)  = F_FLOAT(0);
      d_dc6V(i) = F_FLOAT(0);
    });

  // 6) Compute using only device Views
  const F_FLOAT l_cn_thr = static_cast<F_FLOAT>(cn_thr);
  Kokkos::parallel_for(
    "PairDispersionD3KokkosCalcCNKK",
    Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
    KOKKOS_LAMBDA(const int &ii) {
      const int i     = d_ilistV[ii];
      const int itype = d_typeV(i);
      const int jnum  = d_numneighV(ii);
      const auto rcov_i = d_rcovV(itype);
      //const auto neigh_i = k_list->get_neighbors(i);
      for (int jj = 0; jj < jnum; ++jj) {
        const int jenc = d_neighborsV(ii, jj);
        const int j = jenc & NEIGHMASK;
        const int jtype = d_typeV(j);
        const auto rcov_j = d_rcovV(jtype);

        const F_FLOAT delx = d_x(i,0) - d_x(j,0);
        const F_FLOAT dely = d_x(i,1) - d_x(j,1);
        const F_FLOAT delz = d_x(i,2) - d_x(j,2);
        const F_FLOAT rsq  = delx*delx + dely*dely + delz*delz;

        if (rsq > l_cn_thr) continue;

        const F_FLOAT rr      = sqrt(rsq);
        const F_FLOAT rcov_ij = (rcov_i + rcov_j) * autoang;
        const F_FLOAT cn_ij   = F_FLOAT(1) / (F_FLOAT(1) + exp(-K1 * ((rcov_ij/rr) - F_FLOAT(1))));

        d_cnV(i) += cn_ij;
        if (newton_pair || j < nlocal) Kokkos::atomic_add(&d_cnV(j), cn_ij);
      }
    });

  // 7) Comm as before
  communicationStage = 1;
  if (newton_pair)
  { 
    k_cnV.template modify<DeviceType>();
    comm->reverse_comm(this);
    k_cnV.template sync<DeviceType>(); 
  }
  else
  { 
    k_cnV.template modify<DeviceType>();
    comm->forward_comm(this);
    k_cnV.template sync<DeviceType>();
  }
  
  copymode = 0; 
}

template<class DeviceType>
void PairDispersionD3Kokkos<DeviceType>::compute(int eflag, int vflag)
{
  eflag = eflag;
  vflag = vflag; 
  
  if (!coeffs_synced) 
  {
    sync_coeffs_to_device();
    coeffs_synced = true;
  }
   
  // maybe move elsewhere   
  //sync_coeffs_to_device();
  // init energy/virial flags
  ev_init(eflag, vflag);

  
  if (eflag_atom)
  {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, atom->nmax, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>(); 
  }

  if (vflag_atom)
  {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, atom->nmax, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>(); 
  }
  // 1) coordination numbers (fills k_cn and zeros k_dc6)
  calc_coordination_numberKK();

  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag ) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  // common flags and sizes
  // potential error source eflag =! evflag
  const int  l_eflag       = static_cast<int>(eflag);
  const int  l_evflag      = static_cast<int>(evflag);
  const int  l_newton_pair = static_cast<int>(force->newton_pair); 
  const int  l_nlocal      = atom->nlocal;
  const int  inum          = list->inum;

  // neighbor list
  auto* k_listV       = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto  d_ilistV     = k_listV->d_ilist;
  auto  d_numneighV  = k_listV->d_numneigh;
  auto  d_neighborsV = k_listV->d_neighbors; 
   
  if (inum > 0) {
    if (d_ilistV.extent_int(0)    != inum || d_numneighV.extent_int(0) != inum || d_neighborsV.extent_int(0)!= inum) 
    {
      error->all(FLERR, "Half-list device neighbor arrays not sized to inum");
    }
  }

   
  Few<int,4> f_special_lj;
  f_special_lj[0] = force->special_lj[0];
  f_special_lj[1] = force->special_lj[1];
  f_special_lj[2] = force->special_lj[2];
  f_special_lj[3] = force->special_lj[3];
 
  // device views
  CHECK_DUALVIEW_DEVICE_ALLOC(k_x, "k_x");
  //auto d_x          = k_x.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_f, "k_f");
  //auto d_f          = k_f.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_typeV, "k_type");
  //auto d_type       = k_type.template view<DeviceType>();
  // no longer a dual view, unnecessary check
  //CHECK_DUALVIEW_DEVICE_ALLOC(d_special_lj, "d_special_lj");
  //auto d_special_lj = k_special_lj.template view<DeviceType>();

  auto d_x      = atomKK->k_x.view<DeviceType>();
  auto d_f      = atomKK->k_f.view<DeviceType>();
  auto d_typeV  = atomKK->k_type.view<DeviceType>();
  

  CHECK_DUALVIEW_DEVICE_ALLOC(k_cnV, "k_cn"); 
  auto d_cnV       = k_cnV.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_dc6V, "k_dc6");
  auto d_dc6V      = k_dc6V.template view<DeviceType>();
 
  auto h_cnV         = k_cnV.h_view;
  auto h_dc6V        = k_dc6V.h_view;
 
  CHECK_DUALVIEW_DEVICE_ALLOC(k_paramsV, "k_params"); 
  auto d_paramsV     = k_paramsV.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_r2r4V, "k_r2r4"); 
  auto d_r2r4V       = k_r2r4V.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_rcovV, "k_rcov");
  auto d_rcovV       = k_rcovV.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_mxciV, "k_mxci");
  auto d_mxciV       = k_mxciV.template view<DeviceType>();
  CHECK_DUALVIEW_DEVICE_ALLOC(k_c6abV, "k_c6ab");
  auto d_c6abV       = k_c6abV.template view<DeviceType>();


  /*
  for (int i = 0; i < atom->nlocal; i++)
  {
    const int t = k_type.h_view(i);
    if (t < 1 || t > atom->ntypes)
    {
      error->all(FLERR, "Invalid atom type %d at local index %d (ntypes=%d)", t, i, atom->ntypes);
    }
  }*/
  // constants used on device
  const F_FLOAT l_autoang  = static_cast<F_FLOAT>(autoang);
  const F_FLOAT l_cn_thr   = static_cast<F_FLOAT>(cn_thr);  // squared distance threshold
  const F_FLOAT l_K1       = static_cast<F_FLOAT>(K1);
  
  const EV_FLOAT ev; 
  View1D_EFLOAT_6  d_vglobalV("vglobal");
  Kokkos::parallel_for(
  "ZeroVGlobal",
  Kokkos::RangePolicy<typename DeviceType::execution_space>(0,6),
  KOKKOS_LAMBDA (const int k) {
    d_vglobalV(k) = E_FLOAT(0);
  }); 
  const int l_eflagatom = static_cast<int>(eflag_atom); 
  const int l_neighflag = 1; // KOKKOS uses half lists
  copymode = 1;
  // 2) main pair interaction kernel: forces + energy + dC6 accumulation
  {
    PairDispD3Kernel_dEdIJ<DeviceType> f1(
      d_x, d_f, d_cnV, d_dc6V, d_typeV, f_special_lj, d_mxciV, d_c6abV,
      d_paramsV, d_r2r4V, d_ilistV, d_numneighV, d_neighborsV,
      l_eflag, l_eflagatom, l_newton_pair, l_nlocal, l_neighflag, l_autoang,
      d_vglobalV, ev, vflag_either, vflag_global, vflag_atom, d_eatom, d_vatom);

    Kokkos::parallel_for(
      "PairDispD3Kokkos_dEdIJ",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
      f1);

    // mark modified on device
    k_f.template modify<DeviceType>();
   // k_dc6.template modify<DeviceType>();
  }

  // 3) communicate dC6 (same pattern as the CPU version)
  communicationStage = 2;
  if (l_newton_pair)
  {
    k_dc6V.template modify<DeviceType>();
    comm->reverse_comm(this);
    k_dc6V.template sync<DeviceType>();
  } else
  {
    k_dc6V.template modify<DeviceType>();
    comm->forward_comm(this);
    k_dc6V.template sync<DeviceType>();
  }

  // 4) gradient from dCN/dr using accumulated dC6 (adds to forces)
  {
    PairDispD3Kernel_dEdXYZ<DeviceType> f2(
      d_x, d_f, d_dc6V, d_typeV, f_special_lj, d_rcovV, d_paramsV,
      d_ilistV, d_numneighV, d_neighborsV,
      l_newton_pair, l_evflag, l_nlocal, l_neighflag, l_autoang, l_cn_thr, l_K1,
      d_vglobalV, ev, vflag_either, vflag_global, vflag_atom, d_eatom, d_vatom);

    Kokkos::parallel_for(
      "PairDispD3Kokkos_dEdXYZ",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
      f2);

    // mark modified on device
    k_f.template modify<DeviceType>();
  }
  
  E_FLOAT h_vglobal[6];
  Kokkos::View<E_FLOAT[6], Kokkos::HostSpace> h_vglobalV(h_vglobalV);
  Kokkos::deep_copy(h_vglobalV, d_vglobalV);

  for (int m = 0; m < 6; ++m) this->virial[m] += h_vglobal[m];
  
  // optional virial via f · r
  if (vflag_fdotr) virial_fdotr_compute();
  copymode = 0; 
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

// unpack 

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
template class PairDispersionD3Kokkos<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_SERIAL
template class PairDispersionD3Kokkos<Kokkos::Serial>;
#endif
}
