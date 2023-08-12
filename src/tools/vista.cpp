/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "vista.hpp"

using namespace MrHyDE;
    
#ifndef MrHyDE_NO_AD
template<class EvalT>
Vista<EvalT>::Vista(View_EvalT2 vdata) {
  viewdata_ = vdata;
  is_AD_ = true;
  is_view_ = true;
}
#endif
    
template<class EvalT>
Vista<EvalT>::Vista(View_Sc2 vdata) {
  viewdata_Sc_ = vdata;
  viewdata_ = View_EvalT2("2D view",vdata.extent(0),vdata.extent(1));
  is_view_ = true;
  is_AD_ = false;
}

//#ifndef MrHyDE_NO_AD
//template<class EvalT>
//Vista<EvalT>::Vista(EvalT & data_) {
//  viewdata_ = View_EvalT2("2D view",1,1);
//  deep_copy(viewdata_,data_);
//  is_view_ = false;
//  is_AD_ = true;
//}
//#endif

template<class EvalT> 
Vista<EvalT>::Vista(ScalarT & data_) {
  viewdata_ = View_EvalT2("2D view",1,1);
  deep_copy(viewdata_,data_);
  is_view_ = false;
  is_AD_ = false;
}

#ifndef MrHyDE_NO_AD
template<class EvalT>
void Vista<EvalT>::update(View_EvalT2 vdata) {
  viewdata_ = vdata;
}
#endif
    
template<class EvalT>
void Vista<EvalT>::update(View_Sc2 vdata) {
  viewdata_Sc_ = vdata;
}
    
#ifndef MrHyDE_NO_AD
template<class EvalT>
void Vista<EvalT>::update(EvalT & data_) {
  deep_copy(viewdata_,data_);
}
#endif

template<class EvalT>   
void Vista<EvalT>::updateSc(ScalarT & data_) {
  deep_copy(viewdata_,data_);
}

//template<class EvalT>   
//void Vista<EvalT>::updateParam(AD & pdata_) {
  // nothing for now
//}

//KOKKOS_INLINE_FUNCTION    
//template<>
//ScalarT Vista<ScalarT>::operator()(const size_type & i0, const size_type & i1) const {
//  if (is_view_) {
//    if (is_AD_) {
//      return viewdata_(i0,i1);
//    }
//    else {
//#ifndef MrHyDE_NO_AD
//      viewdata_(i0,i1) = viewdata_Sc_(i0,i1);
//      return viewdata_(i0,i1);
//#else
//      return viewdata_Sc_(i0,i1);
//#endif
 //   }
//  }
//  else {
//    return viewdata_(0,0);
//  }/
//}
    
/*
template<class EvalT>
KOKKOS_INLINE_FUNCTION 
typename Kokkos::View<EvalT**,ContLayout,AssemblyDevice>::reference_type Vista<EvalT>::operator()(const size_type & i0, const size_type & i1) const {
  if (is_view_) {
    if (is_AD_) {
      return viewdata_(i0,i1);
    }
    else {
#ifndef MrHyDE_NO_AD
      //viewdata_(i0,i1).val() = viewdata_Sc_(i0,i1);
      viewdata_(i0,i1) = viewdata_Sc_(i0,i1);
      return viewdata_(i0,i1);
#else
      return viewdata_Sc_(i0,i1);
#endif
    }
  }
  else {
    return viewdata_(0,0);
  }
}
*/
/*
template<>
KOKKOS_INLINE_FUNCTION 
Kokkos::View<AD**,ContLayout,AssemblyDevice>::reference_type Vista<EvalT>::operator()(const size_type & i0, const size_type & i1) const {
  if (is_view_) {
    if (is_AD_) {
      return viewdata_(i0,i1);
    }
    else {
#ifndef MrHyDE_NO_AD
      //viewdata_(i0,i1).val() = viewdata_Sc_(i0,i1);
      viewdata_(i0,i1) = viewdata_Sc_(i0,i1);
      return viewdata_(i0,i1);
#else
      return viewdata_Sc_(i0,i1);
#endif
    }
  }
  else {
    return viewdata_(0,0);
  }
}
*/
/*
template<>
KOKKOS_INLINE_FUNCTION 
Kokkos::View<AD**,ContLayout,AssemblyDevice>::reference_type Vista<EvalT>::operator()(const size_type & i0, const size_type & i1) const {
  if (is_view_) {
    if (is_AD_) {
      return viewdata_(i0,i1);
    }
    else {
#ifndef MrHyDE_NO_AD
      //viewdata_(i0,i1).val() = viewdata_Sc_(i0,i1);
      viewdata_(i0,i1) = viewdata_Sc_(i0,i1);
      return viewdata_(i0,i1);
#else
      return viewdata_Sc_(i0,i1);
#endif
    }
  }
  else {
    return viewdata_(0,0);
  }
}*/

template<class EvalT>
bool Vista<EvalT>::isView() {
  return is_view_;
}

template<class EvalT>
bool Vista<EvalT>::isAD() {
  return is_AD_;
}
    
template<class EvalT>
Kokkos::View<EvalT**,ContLayout,AssemblyDevice> Vista<EvalT>::getData() {
  return viewdata_;
}

template<class EvalT>
View_Sc2 Vista<EvalT>::getDataSc() {
  return viewdata_Sc_;
}
    
template<class EvalT>
void Vista<EvalT>::print() {
  std::cout << "Printing Vista -------" <<std::endl;
  std::cout << "  Is View: " << is_view_ << std::endl;
  std::cout << "  Is AD: " << is_AD_ << std::endl;    
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

// Avoid redefining since ScalarT=AD if no AD
#ifndef MrHyDE_NO_AD
template class MrHyDE::Vista<ScalarT>;
#endif

// Custom AD type
template class MrHyDE::Vista<AD>;

// Standard built-in types
template class MrHyDE::Vista<AD2>;
template class MrHyDE::Vista<AD4>;
template class MrHyDE::Vista<AD8>;
template class MrHyDE::Vista<AD16>;
template class MrHyDE::Vista<AD18>;
template class MrHyDE::Vista<AD24>;
template class MrHyDE::Vista<AD32>;