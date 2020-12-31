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

#ifndef PHYSICS_H
#define PHYSICS_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "physics_base.hpp"
#include "workset.hpp"

#include "Panzer_STK_Interface.hpp"
#include "Panzer_DOFManager.hpp"

namespace MrHyDE {
  
  /*
  static void physicsHelp(const string & details) {
    
    if (details == "none") {
      cout << "********** Help and Documentation for the Physics Interface **********" << endl;
    }
    
     else if (details == "porousHDIV") {
     porousHDIVHelp();
     }
     else if (details == "thermal") {
     thermalHelp();
     }
     else if (details == "thermal_enthalpy") {
     thermal_enthalpyHelp();
     }
     else if (details == "msphasefield") {
     msphasefieldHelp();
     }
     else if (details == "stokes") {
     stokesHelp();
     }
     else if (details == "navierstokes") {
     navierstokesHelp();
     }
     else if (details == "linearelasticity") {
     linearelasticityHelp();
     }
     else if (details == "helmholtz") {
     helmholtzHelp();
     }
     else if (details == "maxwells_fp") {
     maxwells_fpHelp();
     }
     else if (details == "shallowwater") {
     shallowwaterHelp();
     }
     else {
     cout << "Physics module help: unrecognized details: " << details << endl;
     }
     
  }
  */
  
  class physics {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    physics() {} ;
    
    physics(Teuchos::RCP<Teuchos::ParameterList> & settings, Teuchos::RCP<MpiComm> & Comm_,
            Teuchos::RCP<panzer_stk::STK_Interface> & mesh);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Add the requested physics modules, variables, discretization types 
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void importPhysics(const bool & isaux);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Add the functions to the function managers
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void defineFunctions(vector<Teuchos::RCP<FunctionManager> > & functionManagers_);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getvarOwner(const int & block, const string & var);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    AD getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                         const ScalarT & t, const string & var, const string & gside,
                         const bool & useadjoint, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    ScalarT getInitialValue(const int & block, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                            const string & var, const bool & useadjoint);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getNumResponses(const int & block, const string & var);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getNumResponses(const int & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_AD3 getPointResponse(const int & block, View_AD4 u_ip, View_AD4 ugrad_ip,
                              View_AD4 p_ip, View_AD4 pgrad_ip,
                              const View_Sc3 ip, const ScalarT & time,
                              Teuchos::RCP<workset> & wkset);
    
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_AD3 getResponse(const int & block, View_AD4 u_ip, View_AD4 ugrad_ip,
                         View_AD4 p_ip, View_AD4 pgrad_ip, const View_Sc3 ip,
                         const ScalarT & time, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    AD computeTopoResp(const size_t & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_AD3 target(const int & block, const View_Sc3 ip,
                    const ScalarT & current_time, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_AD3 weight(const int & block, const View_Sc3 ip,
                    const ScalarT & current_time, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc3 getInitial(const View_Sc3 ip, const int & block,
                        const bool & project, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc2 getDirichlet(const View_Sc3 ip, const int & var,
                          const int & block, const std::string & sidename,
                          Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void setVars();
    
    void setAuxVars(size_t & block, vector<string> & vars);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    std::vector<string> getResponseFieldNames(const int & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    std::vector<string> getExtraFieldNames(const int & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    vector<string> getExtraCellFieldNames(const int & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    vector<View_Sc3> getExtraFields(const int & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc2 getExtraFields(const int & block, const int & fnum,
                            const DRV & ip, const ScalarT & time,
                            Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc1 getExtraCellFields(const int & block, const int & fnum, View_Sc2 wts);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getUniqueIndex(const int & block, const std::string & var);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void volumeResidual(const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void boundaryResidual(const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void computeFlux(const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void setWorkset(vector<Teuchos::RCP<workset> > & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    bool checkFace(const size_t & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void faceResidual(const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Teuchos::ParameterList> settings;
    vector<vector<Teuchos::RCP<physicsbase> > > modules, aux_modules;
    vector<Teuchos::RCP<FunctionManager> > functionManagers;
    Teuchos::RCP<MpiComm> Commptr;
    vector<Teuchos::ParameterList> blockPhysSettings, blockDiscSettings, aux_blockPhysSettings, aux_blockDiscSettings;
    vector<string> blocknames, sideNames;
    int spaceDim, milo_debug_level;
    size_t numBlocks;
    
    bool have_aux = false;
    vector<int> numVars, aux_numVars;
    vector<vector<bool> > useSubgrid, aux_useSubgrid;
    vector<vector<bool> > useDG, aux_useDG;
    //bool haveDirichlet;
    
    vector<vector<string> > varlist, aux_varlist;
    vector<vector<int> > varowned, aux_varowned;
    vector<vector<int> > orders, aux_orders;
    vector<vector<string> > types, aux_types;
    vector<vector<int> > unique_orders, aux_unique_orders;
    vector<vector<string> > unique_types, aux_unique_types;
    vector<vector<int> > unique_index, aux_unique_index;
        
    string initial_type, cellfield_reduction;
    
    vector<vector<string> > extrafields_list, extracellfields_list, response_list, target_list, weight_list;
    
    Teuchos::RCP<Teuchos::Time> bctimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::setBCData()");
    Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::setDirichletData()");
    Teuchos::RCP<Teuchos::Time> sideinfotimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::getSideInfo()");
    Teuchos::RCP<Teuchos::Time> responsetimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics:computeResponse()");
    Teuchos::RCP<Teuchos::Time> pointreponsetimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::computePointResponse()");
    
  };
  
}

#endif
