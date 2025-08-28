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
    // flags for lammps 
    enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
    
    typedef ArrayTypes<DeviceType> AT; 
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
                              
    int    pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                    DAT::tdual_xfloat_1d &buf,
                                    int /*pbc_flag*/, int  * /*pbc*/);
    void unpack_forward_comm_kokkos(int n, int first, 
                                    DAT::tdual_xfloat_1d &buf);
//    int    pack_reverse_comm_kokkos(int n, int first, 
//                                    DAT::tdual_xfloat_1d &k_buf);
//    void unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d k_list,
//                                    DAT::tdual_xfloat_1d &k_buf);
 
    //double *get_dC6(int, int, double, double) override;
    //double get_dC6(int iat, int jat, double cni, double cnj) override; 
    // Holds all per-typepair constants for D3 zero damping
    /*struct params_d3 
    {
    // Common scalars
      F_FLOAT cut;      // cutoff distance
      F_FLOAT cutsq;    // squared cutoff
      F_FLOAT s6;       // scaling factor for C6 term
      F_FLOAT s8;       // scaling factor for C8 term (if used)
      F_FLOAT rs6;      // damping range scaling for C6
      F_FLOAT rs8;      // damping range scaling for C8
      F_FLOAT alpha;    // damping exponent

      // Typepair‐specific data
      F_FLOAT r0_ab;    // reference R0 for pair (i,j)
//      F_FLOAT c6_ab;    // C6 coefficient for (i,j)
//      F_FLOAT c8_ab;    // C8 coefficient for (i,j) if applicable

      // Padding for alignment if you like (helps coalescing)
      F_FLOAT pad[1];
     }; */
  protected:
  //Kokkos::DualView<F_FLOAT*,         Kokkos::LayoutRight,DeviceType> k_r2r4;   // per-type
  //Kokkos::DualView<F_FLOAT*,         Kokkos::LayoutRight,DeviceType> k_cn; 
  
  
  typename AT::tdual_xfloat_1d      buf;
  typename AT::tdual_xfloat_2d      k_x; 
  typename AT::tdual_ffloat_2d      k_f;
  typename AT::tdual_int_1d         k_list, k_type, k_ilist, k_numneigh, k_special_lj; 
  //typename AT::tdual_double_1d      k_C6Dev, k_cn, k_dc6,  k_r2r4, k_rcov, k_cn_thr, k_rthr;
  typename AT::tdual_double_1d      k_C6Dev, k_cn, k_dc6, k_rcov;
  //typename AT::tdual_int_1d         k_mxci; 
  typename AT::tdual_neighbors_2d   k_neighbors; 
  //typename AT::tdual_neighbors_2d   k_firstneigh; 
  
  int communicationStage, neighflag ;
  double k_rthr, k_cn_thr; 
  
  // corresponding views for mxci & c6ab
  // in your header
  Kokkos::DualView<params_d3**, Kokkos::LayoutRight, DeviceType> k_params;
  Kokkos::DualView<F_FLOAT*, DeviceType>                         k_r2r4;
  Kokkos::DualView<int*, DeviceType>                             k_mxci;
  Kokkos::DualView<double*****, Kokkos::LayoutRight, DeviceType> k_c6ab;

  void repack_host_to_dual(int ntypes, int gi_max, int gj_max);

  int first;
  DAT::tdual_int_1d k_sendlist;  
  typename AT::t_xfloat_1d_um buf_v;
  typename AT::t_int_1d d_sendlist;
  typename AT::t_ffloat_1d d_cn;
  typename AT::t_ffloat_1d d_dc6;
  HAT::t_ffloat_1d h_cn;
  HAT::t_ffloat_1d h_dc6;
   
};

}


#endif 
#endif
