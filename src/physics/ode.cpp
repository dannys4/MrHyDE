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

#include "ode.hpp"
using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

ODE::ODE(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_) {
  
  label = "ode";
  myvars.push_back("q");
  mybasistypes.push_back("HVOL");
  
}

// ========================================================================================
// ========================================================================================

void ODE::defineFunctions(Teuchos::ParameterList & fs,
                          Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  functionManager->addFunction("ODE source",fs.get<string>("ODE source","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void ODE::volumeResidual() {
  
  View_AD2_sv source;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("ODE source","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  int q_basis = wkset->usebasis[qnum];
  auto basis = wkset->basis[q_basis];
  auto res = wkset->res;
  auto sol_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),qnum,0,0);  
  // Simply solves q_dot = f(q,t)
  auto off = Kokkos::subview(offsets,qnum,Kokkos::ALL());
  parallel_for("ODE volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
    res(e,off(0)) += sol_dot(e) - source(e,0);
  });
}

// ========================================================================================
// ========================================================================================

void ODE::boundaryResidual() {
  
}

// ========================================================================================
// ========================================================================================

void ODE::edgeResidual() {}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void ODE::computeFlux() {
}

// ========================================================================================
// ========================================================================================

void ODE::setVars(vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "q") {
      qnum = i;
    }
  }
}
