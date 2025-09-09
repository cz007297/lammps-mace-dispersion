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

namespace LAMMPS_NS{

// Communication tags
struct TagPairDD3PackForwardCommCN{};
struct TagPairDD3PackForwardCommDC6{};
struct TagPairDD3UnpackForwardCommCN{};
struct TagPairDD3UnpackForwardCommDC6{};
struct TagPairDD3PackReverseCommCN{};
struct TagPairDD3PackReverseCommDC6{};
struct TagPairDD3UnpackReverseCommCN{};
struct TagPairDD3UnpackReverseCommDC6{};

// Computation tags
struct TagPairDD3KokkosCNDC6Initialise{};
//struct TagPairDD3KokkosCNDC6Calc{};
template<int NEIGHFLAG>
struct TagPairDD3KokkosCNDC6Kernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJOriginalZeroDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJModifiedZeroDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJOriginalBJDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJModifiedBJDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdXYZ{};

template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3, public KokkosBase
{
  public:
    enum {EnabledNeighFlags=HALF};
    typedef ArrayTypes<DeviceType> AT;
    typedef DeviceType device_type;
    typename AT::t_x_array_const_um x;
    typename AT::t_f_array_const_um f;
    typedef EV_FLOAT value_type;
   
    PairDispersionD3Kokkos(class LAMMPS *);
    ~PairDispersionD3Kokkos();
    
    void compute(int eflag, int vflag) override; 
    void coeff(int, char**) override;
    void init_style() override;  
    double init_one(int i, int j) override;  
    // Main computation operators
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdXYZ<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int& ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdXYZ<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int& ii) const; 
    
    // Coordination number calculation operators
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3KokkosCNDC6Initialise, const int &i) const;
   
    template<int NEIGHFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3KokkosCNDC6Kernel<NEIGHFLAG>, const int &ii) const;
  
    // C6 coefficient calculation
    KOKKOS_INLINE_FUNCTION
    void dC6KK(int itype, int jtype,
                   double cni, double cnj, 
                   double &C6, double &dC6_i, double &dC6_j) const;
    
    // Communication operators
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackForwardCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackForwardCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackForwardCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackForwardCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackReverseCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackReverseCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackReverseCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackReverseCommDC6, const int &i) const;

    // Communication interface methods
    int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&, int, int*) override;
    void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
    int pack_reverse_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
    void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&) override;
    
    int pack_forward_comm(int, int*, double*, int, int*) override;
    void unpack_forward_comm(int, int, double*) override;
    int pack_reverse_comm(int, int, double*) override;
    void unpack_reverse_comm(int, int*, double*) override;

  protected:
    int need_dup;
    typedef typename Kokkos::DualView<float*****, DeviceType> tdual_float_5d;
    
    // Coordination number calculation
    void calc_coordination_numbersKK();
    
    // Device views
    typename AT::t_x_array d_x;
    typename AT::t_f_array d_f;
    typename AT::t_int_1d d_type;
    typename AT::t_neighbors_2d d_neighbors;
    typename AT::t_int_1d d_ilist, d_numneigh;
   
    // energy & velocity views
    DAT::tdual_efloat_1d k_eatom;
    DAT::tdual_virial_array k_vatom;
    typename AT::t_efloat_1d d_eatom;
    typename AT::t_virial_array d_vatom;
    
    // Parameter arrays
    DAT::tdual_float_1d k_mxci_v;
    typename AT::t_float_1d d_mxci_v;
    
    DAT::tdual_float_1d k_cn_v;
    typename AT::t_float_1d d_cn_v;
    
    DAT::tdual_float_1d k_dc6_v; 
    typename AT::t_float_1d d_dc6_v;
    
    DAT::tdual_float_1d k_r2r4_v;
    typename AT::t_float_1d d_r2r4_v;
    
    DAT::tdual_float_1d k_rcov_v;
    typename AT::t_float_1d d_rcov_v;
    
    DAT::tdual_float_2d k_r0ab_v;
    typename AT::t_float_2d d_r0ab_v;
    
    DAT::tdual_float_2d k_cutsq_v;
    typename AT::t_float_2d d_cutsq_v;
    
    tdual_float_5d k_c6ab_v;
    typename tdual_float_5d::t_dev d_c6ab_v;
    
    // Communication variables
    typename AT::t_int_1d d_sendlistV;
    typename AT::t_xfloat_1d bufV;
    int communicationStage;
    int first;
    
    // Local variables
    int nlocal, nall, inum, nmax;
    int newton_pair;
    ExecutionSpace execution_space;

    friend void pair_virial_fdotr_compute<PairDispersionD3Kokkos>(PairDispersionD3Kokkos*);
    
    template<int NEIGHFLAG, int NEWTON_PAIR>
    KOKKOS_INLINE_FUNCTION
    void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
                  const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

    // definitions for scatterviews
    using KKDeviceType = typename KKDevice<DeviceType>::value;
    
    template<typename DataType, typename Layout>
    using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;
    
    template<typename DataType, typename Layout> 
    using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  
    DupScatterView<F_FLOAT*, typename DAT::tdual_float_1d::array_layout>         dup_cn;
    DupScatterView<F_FLOAT*, typename DAT::tdual_float_1d::array_layout>         dup_dc6;
    DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout>           dup_f; 
    DupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout>            dup_eatom;
    DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout>      dup_vatom;
   
    NonDupScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout>         ndup_cn;
    NonDupScatterView<F_FLOAT*, typename DAT::t_ffloat_1d::array_layout>         ndup_dc6;
    NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout>        ndup_f;
    NonDupScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout>         ndup_eatom;
    NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout>   ndup_vatom;
    
    template<class TAG>
    struct policyInstance;


};
}
#endif
#endif

