#ifdef PAIR_CLASS
//clang-format off
PairStyle(dispersion/d3/kk, PairDispersionD3Kokkos<LMPDeviceType>);
PairStyle(dispersion/d3/device, PairDispersionD3Kokkos<LMPDeviceType>);
PairStyle(dispersion/d3/host, PairDispersionD3Kokkos<LMPHostType>);
#else

#ifndef LMP_PAIR_DISPERSION_D3_KOKKOS_H
#define LMP_PAIR_DISPERSION_D3_KOKKOS_H

#include "pair_dispersion_d3.h"
#include "kokkos_type.h"
#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neigh_list_kokkos.h"

using namespace LAMMPS_NS;

struct TagPairDD3PackForwardCommCN{};
struct TagPairDD3PackForwardCommDC6{};
struct TagPairDD3UnpackForwardCommCN{};
struct TagPairDD3UnpackForwardCommDC6{};
struct TagPairDD3PackReverseCommCN{};
struct TagPairDD3PackReverseCommDC6{};
struct TagPairDD3UnpackReverseCommCN{};
struct TagPairDD3UnpackReverseCommDC6{};

template< int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJ{};

template< int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdXYZ{};

template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3, KokkosBase
{

  public:
    enum {EnabledNeighFlags=HALF};
    using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
    using TeamMember = typename TeamPolicy::member_type;
    typedef ArrayTypes<DeviceType> AT;
   
    PairDispersionD3Kokkos(class LAMMPS *);
    ~PairDispersionD3Kokkos();
    
    void compute(int eflag, int vflag) override; 
    void coeff(int, char**) override;
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJ<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const TeamMember& team, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJ<NEIGHFLAG, NEWTON_PAIR, EVFLAG>,const TeamMember& team) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdXYZ<NEIGHFLAG, NEWTON_PAIR, EVFLAG>,  const int& ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdXYZ<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int& ii ) const; 
  
    void get_dC6KK(const TeamMember& team,
                   int itype, int jtype,
                   double cni, double cnj, 
                   double &C6, double &dC6_i, double &dC6_j) const ;
    
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackForwardCommCN,    const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackForwardCommDC6,   const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackForwardCommCN,  const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackForwardCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackReverseCommCN,    const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackReverseCommDC6,   const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackReverseCommCN,  const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackReverseCommDC6, const int &i) const;

  protected:
    typedef typename Kokkos::DualView<float*****, DeviceType> tdual_float_5d;
    
    typename AT::t_x_array x;
    typename AT::t_f_array f;
    typename AT::t_int_1d type;
    
    DAT::tdual_int_1d k_mxci_v;
    DAT::tdual_float_1d k_cn_v;
    DAT::tdual_float_1d k_dc6_v; 
    DAT::tdual_float_1d k_r2r4_v;
    DAT::tdual_float_1d k_rcov_v;
    DAT::tdual_float_2d k_r0ab_v;
};

#endif
#endif
    
