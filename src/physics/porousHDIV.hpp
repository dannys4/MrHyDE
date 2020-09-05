/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef POROUSHDIV_H
#define POROUSHDIV_H

#include "physics_base.hpp"

static void porousHDIVHelp() {
  cout << "********** Help and Documentation for the Porous (HDIV) Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}


class porousHDIV : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  porousHDIV() {} ;
  
  ~porousHDIV() {};
  
  porousHDIV(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
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
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux();
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist);
  
  // ========================================================================================
  // ========================================================================================
  
  void updatePerm(FDATA Kinv_xx, FDATA Kinv_yy, FDATA Kinv_zz);
  
private:
  
  int spaceDim;
  
  int pnum=-1, unum=-1, auxpnum=-1, auxunum=-1;
  int dxnum,dynum,dznum;
  bool isTD, addBiot, usePermData, useWells;
  ScalarT biot_alpha;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::computeFlux() - evaluation of interface flux");
  
};

#endif
