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
//#include "data.hpp"
#include "klexpansion.hpp"
#include "workset.hpp"
#include "functionManager.hpp"

namespace MrHyDE {
  
  class physicsbase {
    
  public:
    
    physicsbase() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    virtual ~physicsbase() {};
    
    // ========================================================================================
    // Define the functions for this module (not necessary, but probably need to be defined in all modules)
    // ========================================================================================
    
    virtual
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_) {} ;
    
    // ========================================================================================
    // The volumetric contributions to the residual
    // ========================================================================================
    
    virtual
    void volumeResidual() = 0;
    
    // ========================================================================================
    // The boundary contributions to the residual
    // ========================================================================================
    
    virtual
    void boundaryResidual() = 0;
    
    // ========================================================================================
    // The edge (2D) and face (3D) contributions to the residual
    // ========================================================================================
    
    virtual
    void faceResidual() {};
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    virtual
    void computeFlux() = 0;
    
    // ========================================================================================
    // Set the global index for each variable
    // ========================================================================================
    
    virtual void setVars(vector<string> & varlist_) = 0;
    
    // ========================================================================================
    // Set the global index for each variable
    // ========================================================================================
    
    virtual void setAuxVars(vector<string> & auxvarlist) {} ;
    
    // ========================================================================================
    // ========================================================================================
    
    virtual void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params, const std::vector<string> & paramnames) {} ;
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<workset> & wkset_) {
      wkset = wkset_;
      
      
      if (isaux) {
        sol = wkset->local_aux;
        sol_dot = wkset->local_aux_dot;
        sol_grad = wkset->local_aux_grad;
        sol_div = wkset->local_aux_div;
        sol_curl = wkset->local_aux_curl;
        
        sol_side = wkset->local_aux_side;
        sol_grad_side = wkset->local_aux_grad_side;
        
        sol_face = wkset->local_aux_face;
        sol_grad_face = wkset->local_aux_grad_face;
        offsets = wkset->aux_offsets;
        bcs = wkset->aux_var_bcs;
      }
      else {
        sol = wkset->local_soln;
        sol_dot = wkset->local_soln_dot;
        sol_grad = wkset->local_soln_grad;
        sol_div = wkset->local_soln_div;
        sol_curl = wkset->local_soln_curl;
        
        sol_side = wkset->local_soln_side;
        sol_grad_side = wkset->local_soln_grad_side;
        
        sol_face = wkset->local_soln_face;
        sol_grad_face = wkset->local_soln_grad_face;
        offsets = wkset->offsets;
        bcs = wkset->var_bcs;
      }
      
      aux = wkset->local_aux;
      aux_side = wkset->local_aux_side;
      
      //res = wkset->res;
      adjrhs = wkset->adjrhs;
      flux = wkset->flux;
      
    }
    
    // ========================================================================================
    // ========================================================================================
    
    string label;
    
    Teuchos::RCP<workset> wkset;
    Teuchos::RCP<FunctionManager> functionManager;
    int spaceDim;
    vector<string> myvars, mybasistypes;
    bool include_face = false, isaux = false;
    string prefix = "";
    
    // All of these point to specific information in the workset - AND - 
    // We always take subviews, so these are ok on device
    View_AD4 sol, sol_dot, sol_grad, sol_side, sol_grad_side, aux_grad_side, sol_curl, sol_face, sol_grad_face, aux, aux_side;
    View_AD3 sol_div, flux;
    Kokkos::View<int**,AssemblyDevice> offsets;
    
    // Probably not used much
    View_AD2 adjrhs;
    
    // On host, so ok
    Kokkos::View<int**,HostDevice> bcs;
    
    
  };
  
}

#endif
