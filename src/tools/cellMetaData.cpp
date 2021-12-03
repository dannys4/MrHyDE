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

#include "cellMetaData.hpp"
using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

CellMetaData::CellMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
                           const topo_RCP & cellTopo_,
                           const Teuchos::RCP<PhysicsInterface> & physics_RCP_,
                           const size_t & myBlock_,
                           const size_t & myLevel_, const int & numElem_,
                           const bool & build_face_terms_,
                           const vector<bool> & assemble_face_terms_,
                           const vector<string> & sidenames_,
                           const size_t & num_params) :
assemble_face_terms(assemble_face_terms_), build_face_terms(build_face_terms_),
myBlock(myBlock_), myLevel(myLevel_), numElem(numElem_),
physics_RCP(physics_RCP_), sidenames(sidenames_), numDiscParams(num_params),
cellTopo(cellTopo_) {

  Teuchos::TimeMonitor localtimer(*celltimer);
  
  compute_diff = settings->sublist("Postprocess").get<bool>("Compute Difference in Objective", true);
  useFineScale = settings->sublist("Postprocess").get<bool>("Use fine scale sensors",true);
  loadSensorFiles = settings->sublist("Analysis").get<bool>("Load Sensor Files",false);
  writeSensorFiles = settings->sublist("Analysis").get<bool>("Write Sensor Files",false);
  mortar_objective = settings->sublist("Solver").get<bool>("Use Mortar Objective",false);
  storeAll = settings->sublist("Solver").get<bool>("store all cell data",true);
  matrix_free = settings->sublist("Solver").get<bool>("matrix free",false);
  
  requiresTransient = true;
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "steady-state") {
    requiresTransient = false;
  }
  
  requiresAdjoint = true;
  if (settings->sublist("Analysis").get<string>("analysis type","forward") == "forward") {
    requiresAdjoint = false;
  }
  
  compute_sol_avg = true;
  if (!(settings->sublist("Postprocess").get<bool>("write solution", false))) {
    compute_sol_avg = false;
  }
  
  multiscale = false;
  numnodes = cellTopo->getNodeCount();
  dimension = cellTopo->getDimension();
  
  if (dimension == 1) {
    numSides = 2;
  }
  else if (dimension == 2) {
    numSides = cellTopo->getSideCount();
  }
  else if (dimension == 3) {
    numSides = cellTopo->getFaceCount();
  }
  //response_type = "global";
  response_type = settings->sublist("Postprocess").get("response type", "pointwise");
  have_cell_phi = false;
  have_cell_rotation = false;
  have_extra_data = false;
  
  numSets = physics_RCP->setnames.size();
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void CellMetaData::updatePhysicsSet(const size_t & set) {
  if (numSets> 1) {
    numDOF = set_numDOF[set];
    numDOF_host = set_numDOF_host[set];
  }
}

//===================================================
// Clear out the saved data if we are done with it
//===================================================

void CellMetaData::clearPhysicalData() {
  jacobian = DRV("empty view");
  jacobianDet = DRV("empty view");
  jacobianInv = DRV("empty view");
  physip = DRV("empty view");
  physwts = DRV("empty view");
  
  side_physwts = DRV("empty view");
  side_physip = DRV("empty view");
  side_physnormals = DRV("empty view");
  side_phystangents = DRV("empty view");
  side_jacobian = DRV("empty view");
  side_jacobianDet = DRV("empty view");
  side_jacobianInv = DRV("empty view");
  
  
  phys_basis1.clear();
  phys_basisgrad1.clear();
  phys_basisdiv1.clear();
  phys_basiscurl1.clear();
  phys_basis2.clear();
  phys_basisgrad2.clear();
  phys_basisdiv2.clear();
  phys_basiscurl2.clear();
  
  side_phys_basis1.clear();
  side_phys_basisgrad1.clear();
  side_phys_basisdiv1.clear();
  side_phys_basiscurl1.clear();
  side_phys_basis2.clear();
  side_phys_basisgrad2.clear();
  side_phys_basisdiv2.clear();
  side_phys_basiscurl2.clear();
  
}
