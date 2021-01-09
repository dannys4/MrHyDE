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

#ifndef ODE_H
#define ODE_H

#include "physics_base.hpp"

namespace MrHyDE {
  
  class ODE : public physicsbase {
  public:
    
    ODE() {} ;
    
    ~ODE() {};
    
    // ========================================================================================
    // ========================================================================================
    
    ODE(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);
    
    
  private:
    
    int qnum;
    View_AD2 dqdt;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::ODE::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::ODE::volumeResidual() - evaluation of residual");
    
  };
  
}

#endif
