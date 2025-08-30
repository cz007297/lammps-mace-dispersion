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


static constexpr double K1 = 16.0;
static constexpr double K3 = -4.0;
static constexpr double autoang =  0.52917725 ;
static constexpr double autoev  = 27.21140795 ;

namespace LAMMPS_NS
{

struct params_d3 {
  F_FLOAT cut;
  F_FLOAT cutsq;
  F_FLOAT s6;
  F_FLOAT s8;
  F_FLOAT rs6;
  F_FLOAT rs8;
  F_FLOAT alpha;
  F_FLOAT r0_ab;
  F_FLOAT pad[1];
};

struct TagPairDD3PackForwardCommCN{};
struct TagPairDD3PackForwardCommDC6{};
struct TagPairDD3UnpackForwardCommCN{};
struct TagPairDD3UnpackForwardCommDC6{};
struct TagPairDD3PackReverseCommCN{};
struct TagPairDD3PackReverseCommDC6{};
struct TagPairDD3UnpackReverseCommCN{};
struct TagPairDD3UnpackReverseCommDC6{};


template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3
{
  
  public:
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
    // flags for lammps 
    enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
    
    typedef ArrayTypes<DeviceType> AT; 
    using View1D_EFLOAT_6 = Kokkos::View<E_FLOAT[6], DeviceType>;
    // override baseclass methods we are not using 
    PairDispersionD3Kokkos(class LAMMPS *);
    ~PairDispersionD3Kokkos() override;
    //void allocate() override; 
    void allocateKK(); 
    double init_one(int, int) override;
    void init_style() override; 
    void coeff(int narg, char **arg) override;  
    void compute(int eflag, int vflag) override; 
    void settings(int narg, char **arg) override; 
    void calc_coordination_numberKK(); 
    void sync_coeffs_to_device();
    int pack_forward_comm(int, int*, double*, int, int*) override;
    void unpack_forward_comm(int, int, double*) override;
    int pack_reverse_comm(int, int, double*) override;
    void unpack_reverse_comm(int, int*, double*) override;
                              
    int    pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlistV,
                                    DAT::tdual_xfloat_1d &buf,
                                    int /*pbc_flag*/, int  * /*pbc*/);
    void unpack_forward_comm_kokkos(int n, int first, 
                                    DAT::tdual_xfloat_1d &buf);
    int    pack_reverse_comm_kokkos(int n, int first, 
                                    DAT::tdual_xfloat_1d &buf);
    void unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d k_sendlistV,
                                    DAT::tdual_xfloat_1d &buf);
  protected:
  bool coeffs_synced = false;
  typename AT::tdual_xfloat_1d      buf;
  typename AT::tdual_xfloat_2d      k_x; 
  typename AT::tdual_ffloat_2d      k_f;
  typename AT::tdual_int_1d         k_listV, k_typeV, k_ilistV, k_numneighV;//, k_special_lj; 
  typename AT::tdual_double_1d      k_C6DevV, k_cnV, k_dc6V, k_rcovV;
  typename AT::tdual_neighbors_2d   k_neighborsV; 
  
  DAT::tdual_efloat_1d        k_eatom;
  DAT::tdual_virial_array     k_vatom;
  typename AT::t_efloat_1d    d_eatom;
  typename AT::t_virial_array d_vatom;
  
  double d_special_lj[4];  
 
  int communicationStage, neighflag ;
  double l_rthr, l_cn_thr; 
  
  // corresponding views for mxci & c6ab
  Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType> k_paramsV;
  Kokkos::DualView<F_FLOAT*, DeviceType>                         k_r2r4V;
  Kokkos::DualView<int*, DeviceType>                             k_mxciV;
  Kokkos::DualView<double*****, Kokkos::LayoutRight, DeviceType> k_c6abV;

  void repack_host_to_dual(int ntypes, int gi_max, int gj_max);

  int first;
  DAT::tdual_int_1d k_sendlistV;  
  typename AT::t_xfloat_1d_um bufV;
  typename AT::t_int_1d d_sendlistV;
  typename AT::t_ffloat_1d d_cnV;
  typename AT::t_ffloat_1d d_dc6V;
  HAT::t_ffloat_1d h_cnV;
  HAT::t_ffloat_1d h_dc6V;
   
};

}


#endif 
#endif
