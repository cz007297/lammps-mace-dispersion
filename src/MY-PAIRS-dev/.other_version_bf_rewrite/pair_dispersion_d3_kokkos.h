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

namespace LAMMPS_NS
{
template<class DeviceType>
struct PairDispersionD3Kokkos_CalcCN {
  // Type aliases for brevity
  using view_1d_t = Kokkos::View<double*, DeviceType>;
  using view_2d_t = Kokkos::View<double**, DeviceType>;
  using view_1i_t = Kokkos::View<int*, DeviceType>;
  using view_2i_t = Kokkos::View<int**, DeviceType>;

  // Data the kernel needs
  view_2d_t d_x;
  view_1i_t d_type;
  view_1i_t d_ilist;
  view_1i_t d_numneigh;
  view_2i_t d_firstneigh;
  view_1d_t d_cn;
  view_1d_t d_rcov;

  int nlocal;
  int newton_pair;
  double cn_thr;
  double K1;
  double autoang;

  // Device‑side operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ii) const {
    const int i      = d_ilist(ii);
    const int itype  = d_type(i);
    const int jnum   = d_numneigh(i);
    const int* jlist = &d_firstneigh(i,0);

    for (int jj = 0; jj < jnum; jj++) {
      const int j      = jlist[jj] & NEIGHMASK;
      const int jtype  = d_type(j);

      const double delx = d_x(i,0) - d_x(j,0);
      const double dely = d_x(i,1) - d_x(j,1);
      const double delz = d_x(i,2) - d_x(j,2);
      const double rsq  = delx*delx + dely*dely + delz*delz;

      if (rsq > cn_thr) continue;

      const double rr      = sqrt(rsq);
      const double rcov_ij = (d_rcov(itype) + d_rcov(jtype)) * autoang;
      const double cn_ij   = 1.0 / (1.0 + exp(-K1 * ((rcov_ij / rr) - 1.0)));

      Kokkos::atomic_add(&d_cn(i), cn_ij);
      if (newton_pair || j < nlocal) {
        Kokkos::atomic_add(&d_cn(j), cn_ij);
      }
    }
  }
};
template<class DeviceType>
struct DevOpC6Functor {
  // Use your existing aliases
  using view_1d  = Kokkos::View<double*, DeviceType>;
  using view_5d  = Kokkos::View<double*****, DeviceType>;
  using view_1i  = Kokkos::View<int*, DeviceType>;

  int iat;
  int jat;
  double cni;
  double cnj;
  view_5d d_c6ab;
  view_1i k_mxci;
  double l_autoev;
  double l_autoang;
  view_1d c6_res; // length 3 on device

  KOKKOS_INLINE_FUNCTION
  void operator()(const int /*idx*/) const {
    double c6mem   = -1.0e20;
    double r_save  =  1.0e20;
    double num     =  0.0, den = 0.0;
    double d_num_i =  0.0, d_num_j = 0.0;
    double d_den_i =  0.0, d_den_j = 0.0;

    for (int ci = 0; ci <= k_mxci(iat); ci++) {
      for (int cj = 0; cj <= k_mxci(jat); cj++) {
        double c6_ref = d_c6ab(iat,jat,ci,cj,0)
                        * l_autoev * pow(l_autoang, 6);
        if (c6_ref > 0.0) {
          double cni_ref = d_c6ab(iat,jat,ci,cj,1);
          double cnj_ref = d_c6ab(iat,jat,ci,cj,2);

          double r = (cni - cni_ref) * (cni - cni_ref)
                   + (cnj - cnj_ref) * (cnj - cnj_ref);

          if (r < r_save) {
            r_save = r;
            c6mem  = c6_ref;
          }

          double expterm = exp(K3 * r);
          num += c6_ref * expterm;
          den += expterm;

          expterm *= 2.0 * K3;

          double term = expterm * (cni - cni_ref);
          d_num_i += c6_ref * term;
          d_den_i += term;

          term = expterm * (cnj - cnj_ref);
          d_num_j += c6_ref * term;
          d_den_j += term;
        }
      }
    }

    if (den > 1.0E-99) {
      c6_res(0) = num / den;
      c6_res(1) = ((d_num_i * den) - (d_den_i * num)) / (den * den);
      c6_res(2) = ((d_num_j * den) - (d_den_j * num)) / (den * den);
    } else {
      c6_res(0) = c6mem;
      c6_res(1) = 0.0;
      c6_res(2) = 0.0;
    }
  }
};


template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3 
{
   using view_1d      = Kokkos::View<double*, DeviceType>;
   using view_2d      = Kokkos::View<double**, DeviceType>;
   using view_5d      = Kokkos::View<double*****, DeviceType>;
   using view_1i      = Kokkos::View<int*, DeviceType>; 
   using view_2i      = Kokkos::View<int**, DeviceType>;
   using host_view_1d = Kokkos::View<int*, Kokkos::HostSpace>;

   using view_1dconst = Kokkos::View<const double*, Kokkos::LayoutRight, DeviceType>;
   
   PairDispersionD3Kokkos(class LAMMPS *);
  ~PairDispersionD3Kokkos() override;

   void allocate() override;
   void settings(int narg, char **arg) override;
   int  find_atomic_number(std::string &key) override; 
   std::vector<int>  is_int_in_array(int array[], int size, int value) override;
   void read_r0ab(int *atomic_numbers, int ntypes) override;
   void set_limit_in_pars_array(int &idx_atom_1, int &idx_atom_2, int &idx_i, int &idx_j) override; 
   void read_c6ab(int *atomic_numbers, int ntypes) override; 
   void coeff(int narg, char **arg) override ; 
   void calc_coordination_number() override; 
   double get_dC6(int iat, int jat, double cni, double cnj) override;
   void compute(int eflag, int vflag) override;
   // void set_funcpar(std::string &functional_name) override ; 
   void k_set_funcpar();  
   double init_one(int i, int j) override; 
   void init_style() override; 
   int pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) override ;
   void unpack_forward_comm(int n, int first, double *buf) override ; 
   int pack_reverse_comm(int n, int first, double *buf) override ; 
   void unpack_reverse_comm(int n, int *list, double *buf) override ;  
   double k_compute_dC6(int iat, int jat, double cni, double cnj); 
   void k_computeKK(int eflag, int vflag);   
   void k_calc_coordination_number();
   void k_dC6(int iat, int jat, double cni, double cnj,
              view_5d d_c6ab, view_1d k_mxci,
              double l_autoev, double l_autoang, double K3,
              view_1d c6_res);
   

 private:
   view_1d k_r2r4, k_rcov, k_mxci, k_cn, k_dc6  ; 
   view_2d k_r0ab, k_cutsq; 
   view_5d k_c6ab ; 
   
   //int     d_inum ; 
   //view_1i d_ilist, d_numneight; 
   //view_2i d_firstneight ; 


   double s6, s8, s18, rs6, rs18, a1, a2, alpha, alpha6, alpha8; 
   double k_rthr, k_cn_thr; 

   int nmax ; 
   std::string damping_type;

   void k_unpack_reverse_comm(int n, int *list, double *buf);
   int  k_pack_reverse_comm(int n, int first, double *buf);
   void k_unpack_forward_comm(int n, int first, double *buf); 
   int  k_pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int* /*pbc*/);
   void k_init_style();
   double k_init_one(int i, int j); 
   void k_coeff(int narg, char **arg); 
   void k_read_c6ab(const int *atomic_numbers, int ntypes);
   void k_read_r0ab(const int *atomic_numbers, int ntypes);
   void k_settings(int narg, char **arg); 
   void k_allocate(); 
   void k_set_limit_in_pars_array(view_1d &parsview, double limit); 
   bool k_is_int_in_array(int value, const host_view_1d &view); 
};      // class PairDispersionD3Kokkos 
}       // namespace LAMMPS_NS
#endif  // LMP_PAIR_DISPERSION_D3_KOKKOS_H
#endif  // PAIR_CLASS 
