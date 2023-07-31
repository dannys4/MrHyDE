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

#ifndef MRHYDE_POROUSWG_H
#define MRHYDE_POROUSWG_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief Weak Galerkin porous media physics class.
   * 
   * This class computes the residuals for the physics described by applying the weak Galerkin finite
   * element method to the PDE for Darcy flow.
   * 
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   (\mathbf{u},\mathbf{v})_T + (p_0, \nabla\cdot\mathbf{v})_T
   *       - \langle p_\partial, \mathbf{v} \cdot \mathbf{n} \rangle_{\mathcal{E}_T} &=& 0, \\
   *   (\mathbf{K}\mathbf{u}, \mathbf{s})_T + (\mathbf{t},\mathbf{s})_T &=& 0, \\
   *   (\nabla\cdot\mathbf{t}, q_0)_T &=& (f, q_0)_T, \\
   *   -\sum\limits_{T\in\mathcal{T}_h}
   *       \langle \mathbf{t}\cdot\mathbf{n}, q_\partial\rangle_{\mathcal{E}_T} &=& 0.
   * \f}
   * Where the unknowns \f$p_0\f$, \f$p_\partial\f$, \f$\mathbf{u}\f$, and \f$\mathbf{t}\f$ are the following:
   *   - \f$p_0\f$ is the interior pressure
   *   - \f$p_\partial\f$ is the boundary pressure
   *   - \f$\mathbf{u}\f$ is the weak Gradient
   *   - \f$\mathbf{t}\f$ is the Darcy velocity
   * The following functions may be specified in the input.yaml file
   *   - "source" is the source term, \f$f\f$
   *   - "perm" is the permeability, \f$\mathbf{K}\f$
   *   - "kxx" is the xx entry of the permeability tensor, \f$\mathbf{K}\f$
   *   - "kxy", "kyx", "kyy" are defined similarly, and similar terms involving z are used in 3d
   */
  class porousWeakGalerkin : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    porousWeakGalerkin() {} ;
    
    ~porousWeakGalerkin() {};
    
    porousWeakGalerkin(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    // The edge (2D) and face (3D) contributions to the residual
    // ========================================================================================
    
    void faceResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<Workset<AD> > & wkset_);

    //void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
    //void setAuxVars(std::vector<string> & auxvarlist);
    
    // ========================================================================================
    // ========================================================================================
    
    void updatePerm(View_AD2 perm);
    
  private:
    
    int pintnum=-1, pbndrynum=-1, unum=-1, tnum=-1;
    int auxpbndrynum=-1, auxunum=-1, auxtnum=-1;
    int dxnum=-1,dynum=-1,dznum=-1;
    
    bool usePermData;
    bool useAC; // use HDIV if this is false, otherwise use HDIV_AC
    
    vector<string> varlist;
    Kokkos::View<int****,AssemblyDevice> sideinfo;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousWeakGalerkin::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousWeakGalerkin::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousWeakGalerkin::computeFlux() - evaluation of interface flux");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousWeakGalerkin::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousWeakGalerkin::boundaryResidual() - evaluation of residual");
    
  };
  
}

#endif
