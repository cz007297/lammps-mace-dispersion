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
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT; 
  PairDispersionD3Kokkos(class LAMMPS *);
  ~PairDispersionD3Kokkos() override; 

  void calc_coordination_number() override;
  void get_dC6(int iat, int jat, double cni, double cnj); override;
  void compute(int, int) override; 
  void allocate() override;

  //void settings(int, char **) override; 
  //void coeff(int, char **) override; 
  //void init_style() override; 
  //double init_one(int, int) override; 
  
  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_reverse_comm(int, int, double *) override;
  
  void unpack_forward_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 private:
  
