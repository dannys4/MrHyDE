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

#ifndef MRHYDE_PHYSICS_IMPORTER_H
#define MRHYDE_PHYSICS_IMPORTER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief physicsImporter physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   */
  class physicsImporter {
    
  public:
    
    physicsImporter() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    ~physicsImporter() {};
    
    // ========================================================================================
    // Define the functions for this module (not necessary, but probably need to be defined in all modules)
    // ========================================================================================
    
    vector<Teuchos::RCP<physicsbase> > import(vector<string> & module_list,
                                              Teuchos::ParameterList & settings,
                                              const int & dimension,
                                              Teuchos::RCP<MpiComm> & Commptr);
  };
  
}

#endif
