/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MULTISCALE_MANAGER_H
#define MULTISCALE_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "cell.hpp"
#include "subgridModel.hpp"
#include "Amesos2.hpp"

void static multiscaleHelp(const string & details) {
  cout << "********** Help and Documentation for the Multiscale Interface **********" << endl;
}

class MultiScale {
  public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  MultiScale(const Teuchos::RCP<MpiComm> & MacroComm_,
             const Teuchos::RCP<MpiComm> & Comm_,
             Teuchos::RCP<Teuchos::ParameterList> & settings_,
             vector<vector<Teuchos::RCP<cell> > > & cells_,
             vector<Teuchos::RCP<SubGridModel> > subgridModels_,
             vector<Teuchos::RCP<FunctionManager> > macro_functionManagers_);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set the information from the macro-scale that does not depend on the specific cell
  ////////////////////////////////////////////////////////////////////////////////
  
  void setMacroInfo(vector<vector<basis_RCP> > & macro_basis_pointers,
                    vector<vector<string> > & macro_basis_types,
                    vector<vector<string> > & macro_varlist,
                    vector<vector<int> > macro_usebasis,
                    vector<vector<vector<int> > > & macro_offsets,
                    Kokkos::View<int*,AssemblyDevice> & macro_numDOF,
                    vector<string> & macro_paramnames,
                    vector<string> & macro_disc_paramnames);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initial assignment of subgrid models to cells
  ////////////////////////////////////////////////////////////////////////////////
  
  ScalarT initialize();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Re-assignment of subgrid models to cells
  ////////////////////////////////////////////////////////////////////////////////
  
  ScalarT update();
  
  void reset();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Post-processing
  ////////////////////////////////////////////////////////////////////////////////
  
  void writeSolution(const string & macrofilename, const vector<ScalarT> & solvetimes,
                     const int & globalPID);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update parameters
  ////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the mean subgrid cell fields
  ////////////////////////////////////////////////////////////////////////////////
  
  
  Kokkos::View<ScalarT**,HostDevice> getMeanCellFields(const size_t & block, const int & timeindex,
                                                      const ScalarT & time, const int & numfields);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update the mesh data (for UQ studies)
  ////////////////////////////////////////////////////////////////////////////////
  
  void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  bool subgrid_static;
  int milo_debug_level, macro_concurrency;
  vector<Teuchos::RCP<SubGridModel> > subgridModels;
  Teuchos::RCP<MpiComm> Comm, MacroComm;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  vector<vector<Teuchos::RCP<cell> > > cells;
  vector<Teuchos::RCP<workset> > macro_wkset;
  vector<vector<Teuchos::RCP<LA_CrsMatrix> > > subgrid_projection_maps;
  vector<Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > > subgrid_projection_solvers;
  vector<Teuchos::RCP<FunctionManager> > macro_functionManagers;
  
  Teuchos::RCP<Teuchos::Time> resettimer = Teuchos::TimeMonitor::getNewCounter("MILO::multiscale::reset()");
  Teuchos::RCP<Teuchos::Time> initializetimer = Teuchos::TimeMonitor::getNewCounter("MILO::multiscale::initialize()");
  Teuchos::RCP<Teuchos::Time> updatetimer = Teuchos::TimeMonitor::getNewCounter("MILO::multiscale::update()");
};

#endif
