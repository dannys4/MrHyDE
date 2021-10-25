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

#ifndef PHYSBASE_H
#define PHYSBASE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "functionManager.hpp"

namespace MrHyDE {
  
  class physicsbase {
    
  public:
    
    physicsbase() {} ;
    
    virtual ~physicsbase() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    physicsbase(Teuchos::ParameterList & settings, const int & dimension_) {
      verbosity = settings.get<int>("verbosity",0);
    };
    
    
    // ========================================================================================
    // Define the functions for this module (not necessary, but probably need to be defined in all modules)
    // ========================================================================================
    
    virtual
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_) {
      functionManager = functionManager_;
        // GH: these print statements may be annoying when running on multiple MPI ranks
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::defineFunctions called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // The volumetric contributions to the residual
    // ========================================================================================
    
    virtual
    void volumeResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::volumeResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // The boundary contributions to the residual
    // ========================================================================================
    
    virtual
    void boundaryResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::boundaryResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // The edge (2D) and face (3D) contributions to the residual
    // ========================================================================================
    
    virtual
    void faceResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::faceResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    virtual
    void computeFlux() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::computeFlux called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // ========================================================================================
    
    virtual void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                                  const std::vector<string> & paramnames) {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::updateParameters called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // ========================================================================================
    
    virtual void setWorkset(Teuchos::RCP<workset> & wkset_) {
      wkset = wkset_;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    virtual std::vector<string> getDerivedNames() {
      std::vector<string> derived;
      return derived;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    virtual std::vector<View_AD2> getDerivedValues() {
      std::vector<View_AD2> derived;
      return derived;
    };
    
    // ========================================================================================
    // ========================================================================================

    /**
     * @brief Returns the integrand and its type (boundary/volume) for integrated quantities required
     * by the physics module. 
     *
     * In general, the user may also request integrated quantities in the input
     * file. The number of spatial dimensions is required explicitly here because the workset is 
     * not finalized before the postprocessing manager is set up.
     *
     * @param[in] spaceDim  The number of spatial dimensions.
     * @return integrandsNamesAndTypes  Integrands, names, and type (boundary/volume) (matrix of strings).
     */
    
    virtual std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim) {
      std::vector< std::vector<string> > integrandsNamesAndTypes;
      return integrandsNamesAndTypes;
    };

    /**
     * @brief Updates any values needed by the residual which depend on integrated quantities
     * required by the physics module.
     *
     * This must be called after the postprocessing routine.
     */

    virtual void updateIntegratedQuantitiesDependents() {
      if (verbosity > 10) {
        std::cout << "*** Warning: physicsBase::updateIntegratedQuantitiesDependents() called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // ========================================================================================

    string label;
    
    Teuchos::RCP<workset> wkset;
    Teuchos::RCP<FunctionManager> functionManager;
    vector<string> myvars, mybasistypes;
    bool include_face = false, isaux = false;
    string prefix = "";
    int verbosity;
    
    // Probably not used much
    View_AD2 adjrhs;
    
    // On host, so ok
    // Kokkos::View<int**,HostDevice> bcs;
    
    
  };
  
}

#endif
