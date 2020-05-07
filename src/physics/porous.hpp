/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef POROUS_H
#define POROUS_H

#include "physics_base.hpp"

class porous : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  porous() {} ;
  
  ~porous() {};
  
  porous(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  // ========================================================================================
  // ========================================================================================
  
  void defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<FunctionManager> & functionManager_);
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual();
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual();

  // ========================================================================================
  // ========================================================================================

  void edgeResidual();

  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================

  void computeFlux();

  // ========================================================================================
  // ========================================================================================

  void setVars(std::vector<string> & varlist_);

private:

  int spaceDim;
  int pnum;
  bool addBiot;
  ScalarT biot_alpha, formparam;
  
  FDATA perm, porosity, viscosity, densref, pref, comp, gravity, source;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous::computeFlux() - evaluation of flux");
  
};

#endif
