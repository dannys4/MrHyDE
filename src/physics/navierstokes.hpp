/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef NAVIERSTOKES_H
#define NAVIERSTOKES_H

#include "physics_base.hpp"

static void navierstokesHelp() {
  cout << "********** Help and Documentation for the Navier Stokes Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class navierstokes : public physicsbase {
public:
  
  navierstokes() {} ;
  
  ~navierstokes() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  navierstokes(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
               const size_t & numip_side_, const int & numElem_,
               Teuchos::RCP<FunctionManager> & functionManager_,
               const size_t & blocknum_);
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual();
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual();
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux();

  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  AD computeTau(const AD & localdiff, const AD & xvl, const AD & yvl, const AD & zvl, const ScalarT & h) const;
  
private:
  
  size_t numip, numip_side, blocknum;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int ux_num, uy_num, uz_num, pr_num, e_num;
  
  bool isTD, useSUPG, usePSPG;
  
  int test;
  
  std::string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  vector<ScalarT> pik;
  bool pin_pr, have_energy;
  ScalarT pin_tol, pin_scale, T_ambient, beta;
  
  int verbosity;
  bool useScalarRespFx;
  
  FDATA dens, visc, source_ux, source_pr, source_uy, source_uz;
  
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::navierstokes::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::navierstokes::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::navierstokes::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::navierstokes::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::navierstokes::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::navierstokes::computeFlux() - evaluation of flux");
  
};

#endif
