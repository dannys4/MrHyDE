/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDMACRODATA_H
#define SUBGRIDMACRODATA_H

#include "trilinos.hpp"
#include "preferences.hpp"

class SubGridMacroData {
public:
  
  SubGridMacroData() {} ;
  
  ~SubGridMacroData() {} ;
  
  SubGridMacroData(DRV & macronodes_, Kokkos::View<int****,HostDevice> & macrosideinfo_,
                   LIDView macroLIDs_,
                   Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation_)
  : macronodes(macronodes_), macrosideinfo(macrosideinfo_), macroLIDs(macroLIDs_),
  macroorientation(macroorientation_) {
    
  }
  
  void setMacroIDs(const size_t & numElem) {
    
    macroIDs = Kokkos::View<int*,AssemblyDevice>("macro elem IDs",numElem);
    
    size_t numMacro = macronodes.extent(0);
    size_t numEperM = numElem/numMacro;
    size_t mID = 0;
    size_t prog = 0;
    for (size_t i=0; i<numMacro; i++) {
      for (size_t j=0; j<numEperM; j++) {
        macroIDs(prog) = i;
        prog++;
      }
    }
    
  }
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV macronodes;
  Kokkos::View<int****,HostDevice> macrosideinfo;
  LIDView macroLIDs;
  Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> macroorientation;
  Kokkos::View<int**,HostDevice> bcs;
  Kokkos::View<int*,AssemblyDevice> macroIDs;
};
#endif

