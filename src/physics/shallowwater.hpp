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

#ifndef MRHYDE_SHALLOWWATER_H
#define MRHYDE_SHALLOWWATER_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief shallowwater physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "source Hu" is the source Hu.
   *   - "flux left" is the flux left.
   *   - "flux right" is the flux right.
   *   - "flux bottom" is the flux bottom.
   *   - "source Hv" is the source Hv.
   *   - "viscosity" is the viscosity.
   *   - "Neumann source Hv" is the Neumann source Hv.
   *   - "source H" is the source H.
   *   - "flux top" is the flux top.
   *   - "bathymetry" is the bathymetry.
   *   - "bottom friction" is the bottom friction.
   *   - "Neumann source Hu" is the Neumann source Hu.
   *   - "bathymetry_y" is the bathymetry_y.
   *   - "Coriolis" is the Coriolis.
   *   - "bathymetry_x" is the bathymetry_x.
   */
  class shallowwater : public physicsbase {
  public:
    
    shallowwater() {} ;
    
    ~shallowwater() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    shallowwater(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
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
    
    void setWorkset(Teuchos::RCP<Workset<AD> > & wkset_);

//    void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
  private:
    
    int H_num, Hu_num, Hv_num;
    ScalarT gravity;
    
    ScalarT formparam;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwater::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwater::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwater::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwater::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwater::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwater::computeFlux() - evaluation of flux");
    
    
  };
  
}

#endif
