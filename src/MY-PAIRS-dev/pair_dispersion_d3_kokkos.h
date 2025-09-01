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
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3 {

 public:
   using view_1d    = Kokkos::View<double*>;
   using view_2d    = Kokkos::View<double**>;
   using view_5d    = Kokkos::View<double*****>;
   using view_i2d   = Kokkos::View<int**>;

   PairDispersionD3(class LAMMPS *);
  ~PairDispersionD3() override;

   void allocate() override;
   void settings(int, char **) override;
   void read_r0ab(int *atomic_numbers, int ntypes) override;
   void read_c6ab(int *atomic_numbers, int ntypes) override; 
   void calc_coordination_number() override; 
   void find_atomic_number(std::string &key) override; 
   void is_int_in_array(int array[], int size, int value) override;
   void set_limit_in_pars_array(int &idx_atom_1, int &idx_atom_2, int &idx_i,                                           int &idx_j) override; 
   void coeff(int narg, char **arg) override ; 
   void pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,                                        int * /*pbc*/) override ;
   void unpack_forward_comm(int n, int first, double *buf) override ; 
   void pack_reverse_comm(int n, int first, double *buf) override ; 
   void unpack_reverse_comm(int n, int *list, double *buf) override ;  

   // do the rest
   
   

 private:
   view_1d k_r2r4, k_rcov, k_mxci, k_cn, k_dc6  ; 
   view_2d k_r0ab ; 
   view_5d k_c6ab ; 
 
   double s6, s8, s18, rs6, rs18, a1, a2, alpha, alpha6, alpha8; 
   double k_rthr, k_cn_thr; 

   int nmax ; 
   std::string damping_type;

   void unpack_reverse_comm(int n, int *list, double *buf);
   void pack_reverse_comm(int n, int first, double *buf);
   void unpack_forward_comm(int n, int first, double *buf); 
   void pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int* /*pbc*/);
   void init_style();
   void init_one(int i, int j); 
   void compute_dC6(int iat, int jat, double cni, double cnj); 
   void coeff(int narg, char **arg); 
   void calc_coordination_number();
   void read_c6ab(const int *atomic_numbers, int ntypes);
   void read_r0ab(const int *atomic_numbers, int ntypes);
   void settings(int narg, char **arg); 
   void allocate(); 
   KOKKOS_INLINE_FUNCTION
   void get_dC6(int iat, int jat, double cni, double cnj, 
                const view_5d k_c6ab, const view_1d k_mxci, 
                double l_autoev, double l_autoang, double l_K3, 
                double *res);
   void set_limit_in_pars_array(ViewType &view, double limit); 
   void is_int_in_array(int value, const host_view_1d<int> &view); 
   void compute_dC6(int iat, int jat, double cni, double cnj);  
   void compute_kokkos(int eflag, int vflag); 
 
#endif
