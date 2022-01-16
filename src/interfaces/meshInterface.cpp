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

#include "meshInterface.hpp"
#include "exodusII.h"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

MeshInterface::MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                             const Teuchos::RCP<MpiComm> & Commptr_) :
settings(settings_), Commptr(Commptr_) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::meshInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface constructor ..." << endl;
    }
  }
  
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  spaceDim = settings->sublist("Mesh").get<int>("dim",0);
  if (spaceDim == 0) {
    spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
  }
  verbosity = settings->get<int>("verbosity",0);
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "mesh_data_pts";
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",false);
    
  }
  
  meshmod_xvar = settings->sublist("Solver").get<int>("solution for x-mesh mod",-1);
  meshmod_yvar = settings->sublist("Solver").get<int>("solution for y-mesh mod",-1);
  meshmod_zvar = settings->sublist("Solver").get<int>("solution for z-mesh mod",-1);
  meshmod_TOL = settings->sublist("Solver").get<ScalarT>("solution based mesh mod TOL",1.0);
  meshmod_usesmoother = settings->sublist("Solver").get<bool>("solution based mesh mod smoother",false);
  meshmod_center = settings->sublist("Solver").get<ScalarT>("solution based mesh mod param",0.1);
  meshmod_layer_size = settings->sublist("Solver").get<ScalarT>("solution based mesh mod layer thickness",0.1);
  
  shards::CellTopology cTopo;
  shards::CellTopology sTopo;
  
  if (spaceDim == 1) {
    cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );// lin. cell topology on the interior
    sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Node>() );          // line cell topology on the boundary
  }
  if (spaceDim == 2) {
    if (shape == "quad") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );          // line cell topology on the boundary
    }
    if (shape == "tri") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );          // line cell topology on the boundary
    }
  }
  if (spaceDim == 3) {
    if (shape == "hex") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );          // line cell topology on the boundary
    }
    if (shape == "tet") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );          // line cell topology on the boundary
    }
    
  }
  // Get dimensions
  numNodesPerElem = cTopo.getNodeCount();
  settings->sublist("Mesh").set("numNodesPerElem",numNodesPerElem,"number of nodes per element");
  sideDim = 0;
  if (spaceDim > 1) {
    sTopo.getDimension();
  }
  settings->sublist("Mesh").set("sideDim",sideDim,"dimension of the sides of each element");
  numSides = cTopo.getSideCount();
  numFaces = cTopo.getFaceCount();
  if (spaceDim == 1)
    settings->sublist("Mesh").set("numSidesPerElem",2,"number of sides per element");
  if (spaceDim == 2)
    settings->sublist("Mesh").set("numSidesPerElem",numSides,"number of sides per element");
  if (spaceDim == 3)
    settings->sublist("Mesh").set("numSidesPerElem",numFaces,"number of sides per element");
  
  // Define a parameter list with the required fields for the panzer_stk mesh factory
  RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
  
  if (settings->sublist("Mesh").get<std::string>("source","Internal") ==  "Exodus") {
    mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
    pl->set("File Name",settings->sublist("Mesh").get<std::string>("mesh file","mesh.exo"));
  }
  else {
    pl->set("X Blocks",settings->sublist("Mesh").get("Xblocks",1));
    pl->set("X Elements",settings->sublist("Mesh").get("NX",20));
    pl->set("X0",settings->sublist("Mesh").get("xmin",0.0));
    pl->set("Xf",settings->sublist("Mesh").get("xmax",1.0));
    if (spaceDim > 1) {
      pl->set("X Procs", settings->sublist("Mesh").get("Xprocs",Commptr->getSize()));
      pl->set("Y Blocks",settings->sublist("Mesh").get("Yblocks",1));
      pl->set("Y Elements",settings->sublist("Mesh").get("NY",20));
      pl->set("Y0",settings->sublist("Mesh").get("ymin",0.0));
      pl->set("Yf",settings->sublist("Mesh").get("ymax",1.0));
      pl->set("Y Procs", settings->sublist("Mesh").get("Yprocs",1));
    }
    if (spaceDim > 2) {
      pl->set("Z Blocks",settings->sublist("Mesh").get("Zblocks",1));
      pl->set("Z Elements",settings->sublist("Mesh").get("NZ",20));
      pl->set("Z0",settings->sublist("Mesh").get("zmin",0.0));
      pl->set("Zf",settings->sublist("Mesh").get("zmax",1.0));
      pl->set("Z Procs", settings->sublist("Mesh").get("Zprocs",1));
    }
    if (spaceDim == 1) {
      mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
    }
    else if (spaceDim == 2) {
      if (shape == "quad") {
        mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
      }
      if (shape == "tri") {
        mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
      }
    }
    else if (spaceDim == 3) {
      if (shape == "hex") {
        mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());
      }
      if (shape == "tet") {
        mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory());
      }
    }
  }
  // Syntax for periodic BCs ... must be set in the mesh input file
  
  if (settings->sublist("Mesh").isSublist("Periodic BCs")) {
    pl->sublist("Periodic BCs").setParameters( settings->sublist("Mesh").sublist("Periodic BCs") );
  }
  
  mesh_factory->setParameterList(pl);
  
  // create the mesh
  stk_mesh = mesh_factory->buildUncommitedMesh(*(Commptr->getRawMpiComm()));
  
  // create a mesh for an optmization movie
  if (settings->sublist("Postprocess").get("create optimization movie",false)) {
    stk_optimization_mesh = mesh_factory->buildUncommitedMesh(*(Commptr->getRawMpiComm()));
  }
  
  stk_mesh->getElementBlockNames(block_names);

  for (size_t b=0; b<block_names.size(); b++) {
    cellTopo.push_back(stk_mesh->getCellTopology(block_names[b]));
  }
  
  for (size_t b=0; b<block_names.size(); b++) {
    topo_RCP cellTopo = stk_mesh->getCellTopology(block_names[b]);
    string shape = cellTopo->getName();
    if (spaceDim == 1) {
      // nothing to do here
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
      if (shape == "Triangle_3") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() )));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() )));
      }
    }
    
  }

  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

MeshInterface::MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                             const Teuchos::RCP<MpiComm> & Commptr_,
                             Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                             Teuchos::RCP<panzer_stk::STK_Interface> & stk_mesh_) :
settings(settings_), Commptr(Commptr_), mesh_factory(mesh_factory_), stk_mesh(stk_mesh_) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  debug_level = settings->get<int>("debug level",0);
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface constructor ..." << endl;
    }
  }
  
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  spaceDim = settings->sublist("Mesh").get<int>("dim",0);
  if (spaceDim == 0) {
    spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
  }
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "mesh_data_pts";
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",true);
    
  }
  
  meshmod_xvar = settings->sublist("Solver").get<int>("solution for x-mesh mod",-1);
  meshmod_yvar = settings->sublist("Solver").get<int>("solution for y-mesh mod",-1);
  meshmod_zvar = settings->sublist("Solver").get<int>("solution for z-mesh mod",-1);
  meshmod_TOL = settings->sublist("Solver").get<ScalarT>("solution based mesh mod TOL",1.0);
  meshmod_usesmoother = settings->sublist("Solver").get<bool>("solution based mesh mod smoother",false);
  meshmod_center = settings->sublist("Solver").get<ScalarT>("solution based mesh mod param",0.1);
  meshmod_layer_size = settings->sublist("Solver").get<ScalarT>("solution based mesh mod layer thickness",0.1);
  
  stk_mesh->getElementBlockNames(block_names);
  
  for (size_t b=0; b<block_names.size(); b++) {
    cellTopo.push_back(stk_mesh->getCellTopology(block_names[b]));
  }
  
  for (size_t b=0; b<block_names.size(); b++) {
    topo_RCP cellTopo = stk_mesh->getCellTopology(block_names[b]);
    string shape = cellTopo->getName();
    if (spaceDim == 1) {
      // nothing to do here?
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
      if (shape == "Triangle_3") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() )));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() )));
      }
    }
    
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::finalize(Teuchos::RCP<PhysicsInterface> & phys) {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface finalize ..." << endl;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Add fields to the mesh
  ////////////////////////////////////////////////////////////////////////////////
  
  if (settings->sublist("Postprocess").get("write solution",false)) {
    std::vector<std::string> appends;
    if (settings->sublist("Analysis").get<std::string>("analysis type","forward") == "UQ") {
      if (settings->sublist("Postprocess").get("write solution",false)) {
        int numsamples = settings->sublist("Analysis").sublist("UQ").get<int>("samples",100);
        for (int j=0; j<numsamples; ++j) {
          std::stringstream ss;
          ss << j;
          appends.push_back(ss.str());
        }
      }
      else {
        appends = {""};
      }
    }
    else {
      appends = {""};
    }
    
    for (size_t app=0; app<appends.size(); ++app) {
      
      std::string append = appends[app];
      for (std::size_t set=0; set<phys->setnames.size(); set++) {
        
        for (std::size_t i=0;i<block_names.size();i++) {
          
          std::vector<string> varlist = phys->varlist[set][i];
          std::vector<string> vartypes = phys->types[set][i];
          
          for (size_t j=0; j<varlist.size(); j++) {
            if (vartypes[j] == "HGRAD") {
              stk_mesh->addSolutionField(varlist[j]+append, block_names[i]);
            }
            else if (vartypes[j] == "HVOL") { // PW constant
              stk_mesh->addCellField(varlist[j]+append, block_names[i]);
            }
            else if (vartypes[j] == "HFACE") { // hybridized variable
              stk_mesh->addCellField(varlist[j]+append, block_names[i]);
            }
            else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL") { // HDIV or HCURL
              stk_mesh->addCellField(varlist[j]+append+"x", block_names[i]);
              if (spaceDim > 1) {
                stk_mesh->addCellField(varlist[j]+append+"y", block_names[i]);
              }
              if (spaceDim > 2) {
                stk_mesh->addCellField(varlist[j]+append+"z", block_names[i]);
              }
            }
          }
          
          //stk_mesh->addSolutionField("disp"+append+"x", block_names[i]);
          //stk_mesh->addSolutionField("disp"+append+"y", block_names[i]);
          //stk_mesh->addSolutionField("disp"+append+"z", block_names[i]);
          
          
          Teuchos::ParameterList efields;
          if (settings->sublist("Postprocess").isSublist(block_names[i])) {
            efields = settings->sublist("Postprocess").sublist(block_names[i]).sublist("Extra fields");
          }
          else {
            efields = settings->sublist("Postprocess").sublist("Extra fields");
          }
          Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
          while (ef_itr != efields.end()) {
            stk_mesh->addSolutionField(ef_itr->first+append, block_names[i]);
            ef_itr++;
          }
          
          Teuchos::ParameterList ecfields;
          if (settings->sublist("Postprocess").isSublist(block_names[i])) {
            ecfields = settings->sublist("Postprocess").sublist(block_names[i]).sublist("Extra cell fields");
          }
          else {
            ecfields = settings->sublist("Postprocess").sublist("Extra cell fields");
          }
          Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
          while (ecf_itr != ecfields.end()) {
            stk_mesh->addCellField(ecf_itr->first+append, block_names[i]);
            if (settings->isSublist("Subgrid")) {
              string sgfn = "subgrid_mean_" + ecf_itr->first;
              stk_mesh->addCellField(sgfn+append, block_names[i]);
            }
            ecf_itr++;
          }
          
          for (size_t j=0; j<phys->modules[set][i].size(); ++j) {
            std::vector<string> derivedlist = phys->modules[set][i][j]->getDerivedNames();
            for (size_t k=0; k<derivedlist.size(); ++k) {
              stk_mesh->addCellField(derivedlist[k]+append, block_names[i]);
            }
          }
          
          if (have_mesh_data || compute_mesh_data) {
            stk_mesh->addCellField("mesh_data_seed", block_names[i]);
            stk_mesh->addCellField("mesh_data", block_names[i]);
          }
          
          if (settings->isSublist("Subgrid")) {
            stk_mesh->addCellField("subgrid model", block_names[i]);
          }
          
          if (settings->sublist("Postprocess").get("write group number",false)) {
            stk_mesh->addCellField("group number", block_names[i]);
          }
          
          if (settings->isSublist("Parameters")) {
            Teuchos::ParameterList parameters = settings->sublist("Parameters");
            Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
            while (pl_itr != parameters.end()) {
              Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
              if (newparam.get<string>("usage") == "discretized") {
                if (newparam.get<string>("type") == "HGRAD") {
                  stk_mesh->addSolutionField(pl_itr->first+append, block_names[i]);
                }
                else if (newparam.get<string>("type") == "HVOL") {
                  stk_mesh->addCellField(pl_itr->first+append, block_names[i]);
                }
                else if (newparam.get<string>("type") == "HDIV" || newparam.get<string>("type") == "HCURL") {
                  stk_mesh->addCellField(pl_itr->first+append+"x", block_names[i]);
                  if (spaceDim > 1) {
                    stk_mesh->addCellField(pl_itr->first+append+"y", block_names[i]);
                  }
                  if (spaceDim > 2) {
                    stk_mesh->addCellField(pl_itr->first+append+"z", block_names[i]);
                  }
                }
              }
              pl_itr++;
            }
          }
        }
      }
      
    }
  }
  
  mesh_factory->completeMeshConstruction(*stk_mesh,*(Commptr->getRawMpiComm()));
  
  if (verbosity>1) {
    if (Commptr->getRank() == 0) {
      stk_mesh->printMetaData(std::cout);
    }
  }
  
  if (settings->sublist("Postprocess").get("create optimization movie",false)) {
    
    for(std::size_t i=0;i<block_names.size();i++) {
      
      if (settings->isSublist("Parameters")) {
        Teuchos::ParameterList parameters = settings->sublist("Parameters");
        Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
        while (pl_itr != parameters.end()) {
          Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
          if (newparam.get<string>("usage") == "discretized") {
            if (newparam.get<string>("type") == "HGRAD") {
              stk_optimization_mesh->addSolutionField(pl_itr->first, block_names[i]);
            }
            else if (newparam.get<string>("type") == "HVOL") {
              stk_optimization_mesh->addCellField(pl_itr->first, block_names[i]);
            }
            else if (newparam.get<string>("type") == "HDIV" || newparam.get<string>("type") == "HCURL") {
              stk_optimization_mesh->addCellField(pl_itr->first+"x", block_names[i]);
              if (spaceDim > 1) {
                stk_optimization_mesh->addCellField(pl_itr->first+"y", block_names[i]);
              }
              if (spaceDim > 2) {
                stk_optimization_mesh->addCellField(pl_itr->first+"z", block_names[i]);
              }
            }
          }
          pl_itr++;
        }
      }
    }
    
    mesh_factory->completeMeshConstruction(*stk_optimization_mesh,*(Commptr->getRawMpiComm()));
    if (verbosity>1) {
      stk_optimization_mesh->printMetaData(std::cout);
    }
  }
  
  if (settings->sublist("Mesh").get<bool>("have element data", false) ||
      settings->sublist("Mesh").get<bool>("have nodal data", false)) {
    this->readExodusData();
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface finalize" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV MeshInterface::perturbMesh(const int & b, DRV & blocknodes) {
  
  ////////////////////////////////////////////////////////////////////////////////
  // Perturb the mesh (if requested)
  ////////////////////////////////////////////////////////////////////////////////
  
  //for (size_t b=0; b<block_names.size(); b++) {
    //vector<size_t> localIds;
    //DRV blocknodes;
    //panzer_stk::workset_utils::getIdsAndVertices(*mesh, block_names[b], localIds, blocknodes);
    int numNodesPerElem = blocknodes.extent(1);
    DRV blocknodePert("blocknodePert",blocknodes.extent(0),numNodesPerElem,spaceDim);
    
    if (settings->sublist("Mesh").get("modify mesh height",false)) {
      vector<vector<ScalarT> > values;
      
      string ptsfile = settings->sublist("Mesh").get("mesh pert file","meshpert.dat");
      std::ifstream fin(ptsfile.c_str());
      
      for (string line; getline(fin, line); )
      {
        replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        values.push_back(vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                         std::istream_iterator<ScalarT>()));
      }
      
      DRV pertdata("pertdata",values.size(),3);
      for (size_t i=0; i<values.size(); i++) {
        for (size_t j=0; j<3; j++) {
          pertdata(i,j) = values[i][j];
        }
      }
      //int Nz = settings->sublist("Mesh").get<int>("NZ",1);
      ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("zmin",0.0);
      ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("zmax",1.0);
      for (size_type k=0; k<blocknodes.extent(0); k++) {
        for (int i=0; i<numNodesPerElem; i++){
          ScalarT x = blocknodes(k,i,0);
          ScalarT y = blocknodes(k,i,1);
          ScalarT z = blocknodes(k,i,2);
          int node = -1;
          ScalarT dist = (ScalarT)RAND_MAX;
          for( size_type j=0; j<pertdata.extent(0); j++ ) {
            ScalarT xhat = pertdata(j,0);
            ScalarT yhat = pertdata(j,1);
            ScalarT d = std::sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat));
            if( d<dist ) {
              node = j;
              dist = d;
            }
          }
          if (node > 0) {
            ScalarT ch = pertdata(node,2);
            blocknodePert(k,i,0) = 0.0;
            blocknodePert(k,i,1) = 0.0;
            blocknodePert(k,i,2) = (ch)*(z-zmin)/(zmax-zmin);
          }
        }
        //for (int k=0; k<blocknodeVert.extent(0); k++) {
        //  for (int i=0; i<numNodesPerElem; i++){
        //    for (int s=0; s<spaceDim; s++) {
        //      blocknodeVert(k,i,s) += blocknodePert(k,i,s);
        //    }
        //  }
        //}
      }
    }
    
    if (settings->sublist("Mesh").get("modify mesh",false)) {
      for (size_type k=0; k<blocknodes.extent(0); k++) {
        for (int i=0; i<numNodesPerElem; i++){
          blocknodePert(k,i,0) = 0.0;
          blocknodePert(k,i,1) = 0.0;
          blocknodePert(k,i,2) = 0.0 + 0.2*sin(2*3.14159*blocknodes(k,i,0))*sin(2*3.14159*blocknodes(k,i,1));
        }
      }
      //for (int k=0; k<blocknodeVert.extent(0); k++) {
      //  for (int i=0; i<numNodesPerElem; i++){
      //    for (int s=0; s<spaceDim; s++) {
      //      blocknodeVert(k,i,s) += blocknodePert(k,i,s);
      //    }
      //  }
      //}
    }
    //nodepert.push_back(blocknodePert);
  //}
  return blocknodePert;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::setMeshData(vector<vector<Teuchos::RCP<Group> > > & groups,
                                vector<vector<Teuchos::RCP<BoundaryGroup>>> & boundary_groups) {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface setMeshData" << endl;
    }
  }
  
  if (have_mesh_data) {
    this->importMeshData(groups, boundary_groups);
  }
  else if (compute_mesh_data) {
    int randSeed = settings->sublist("Mesh").get<int>("random seed", 1234);
    auto seeds = this->generateNewMicrostructure(randSeed);
    this->importNewMicrostructure(randSeed, seeds, groups, boundary_groups);
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface setMeshData" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

View_Sc2 MeshInterface::getElementCenters(DRV nodes, topo_RCP & reftopo) {
  
  typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;

  DRV refCenter("cell center", 1, spaceDim);
  CellTools::getReferenceCellCenter(refCenter, *reftopo);
  DRV tmp_centers("tmp physical cell centers", nodes.extent(0), 1, spaceDim);
  CellTools::mapToPhysicalFrame(tmp_centers, refCenter, nodes, *reftopo);
  View_Sc2 centers("physics cell centers", nodes.extent(0), spaceDim);
  auto tmp_centers_sv = subview(tmp_centers, ALL(), 0, ALL());
  deep_copy(centers, tmp_centers_sv);
  
  return centers;
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::importMeshData(vector<vector<Teuchos::RCP<Group> > > & groups,
                                   vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups) {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::importMeshData ..." << endl;
    }
  }
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  int numdata = 1;
  if (have_rotations) {
    numdata = 9;
  }
  else if (have_rotation_phi) {
    numdata = 3;
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      groups[block][grp]->data_seed = vector<size_t>(numElem);
      groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  Teuchos::RCP<Data> mesh_data;
  
  string mesh_data_pts_file = mesh_data_pts_tag + ".dat";
  string mesh_data_file = mesh_data_tag + ".dat";
  
  bool have_grid_data = settings->sublist("Mesh").get<bool>("data on grid",false);
  if (have_grid_data) {
    int Nx = settings->sublist("Mesh").get<int>("data grid Nx",0);
    int Ny = settings->sublist("Mesh").get<int>("data grid Ny",0);
    int Nz = settings->sublist("Mesh").get<int>("data grid Nz",0);
    mesh_data = Teuchos::rcp(new Data("mesh data", spaceDim, mesh_data_pts_file,
                                      mesh_data_file, false, Nx, Ny, Nz));
    
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        DRV nodes = groups[block][grp]->nodes;
        int numElem = groups[block][grp]->numElem;
        
        auto centers = this->getElementCenters(nodes, groups[block][grp]->groupData->cellTopo);
        auto centers_host = create_mirror_view(centers);
        deep_copy(centers_host,centers);
        
        for (int c=0; c<numElem; c++) {
          ScalarT distance = 0.0;
          
          // Doesn't use the Compadre interface
          int cnode = mesh_data->findClosestGridPoint(centers_host(c,0), centers_host(c,1),
                                                      centers_host(c,2), distance);
          
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode);
          for (size_type i=0; i<cdata.extent(1); i++) {
            groups[block][grp]->data(c,i) = cdata(0,i);
          }
          groups[block][grp]->groupData->have_extra_data = true;
          groups[block][grp]->groupData->have_rotation = have_rotations;
          groups[block][grp]->groupData->have_phi = have_rotation_phi;
          
          groups[block][grp]->data_seed[c] = cnode;
          groups[block][grp]->data_seedindex[c] = cnode % 100;
          groups[block][grp]->data_distance[c] = distance;
        }
      }
    }
    
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        auto centers = this->getElementCenters(nodes, boundary_groups[block][grp]->groupData->cellTopo);
        auto centers_host = create_mirror_view(centers);
        deep_copy(centers_host,centers);
        
        for (int c=0; c<numElem; c++) {
          ScalarT distance = 0.0;
          
          int cnode = mesh_data->findClosestGridPoint(centers_host(c,0), centers_host(c,1),
                                                      centers_host(c,2), distance);
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode);
          for (size_type i=0; i<cdata.extent(1); i++) {
            boundary_groups[block][grp]->data(c,i) = cdata(0,i);
          }
          boundary_groups[block][grp]->groupData->have_extra_data = true;
          boundary_groups[block][grp]->groupData->have_rotation = have_rotations;
          boundary_groups[block][grp]->groupData->have_phi = have_rotation_phi;
          
          boundary_groups[block][grp]->data_seed[c] = cnode;
          boundary_groups[block][grp]->data_seedindex[c] = cnode % 100;
          boundary_groups[block][grp]->data_distance[c] = distance;
        }
      }
    }
  }
  else {
    mesh_data = Teuchos::rcp(new Data("mesh data", spaceDim, mesh_data_pts_file,
                                      mesh_data_file, false));
    
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        DRV nodes = groups[block][grp]->nodes;
        int numElem = groups[block][grp]->numElem;
        
        auto centers = this->getElementCenters(nodes, groups[block][grp]->groupData->cellTopo);
        
        Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
        Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
        
        mesh_data->findClosestPoint(centers,cnode,distance);
        
        auto distance_mirror = Kokkos::create_mirror_view(distance);
        auto data_mirror = Kokkos::create_mirror_view(groups[block][grp]->data);

        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));

          for (size_t i=0; i<cdata.extent(1); i++) {
            data_mirror(c,i) = cdata(0,i);
          }

          groups[block][grp]->groupData->have_extra_data = true;
          groups[block][grp]->groupData->have_rotation = have_rotations;
          groups[block][grp]->groupData->have_phi = have_rotation_phi;
          
          groups[block][grp]->data_seed[c] = cnode(c);
          groups[block][grp]->data_seedindex[c] = cnode(c) % 100;
          groups[block][grp]->data_distance[c] = distance_mirror(c);

        }
        Kokkos::deep_copy(groups[block][grp]->data, data_mirror);
      }
    }
    
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        auto centers = this->getElementCenters(nodes, boundary_groups[block][grp]->groupData->cellTopo);
        
        Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
        Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
        
        mesh_data->findClosestPoint(centers,cnode,distance);

        auto distance_mirror = Kokkos::create_mirror_view(distance);
        auto data_mirror = Kokkos::create_mirror_view(boundary_groups[block][grp]->data);
        
        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));

          for (size_t i=0; i<cdata.extent(1); i++) {
            data_mirror(c,i) = cdata(0,i);
          }

          boundary_groups[block][grp]->groupData->have_extra_data = true;
          boundary_groups[block][grp]->groupData->have_rotation = have_rotations;
          boundary_groups[block][grp]->groupData->have_phi = have_rotation_phi;
          
          boundary_groups[block][grp]->data_seed[c] = cnode(c);
          boundary_groups[block][grp]->data_seedindex[c] = cnode(c) % 50;
          boundary_groups[block][grp]->data_distance[c] = distance_mirror(c);
        }
        Kokkos::deep_copy(boundary_groups[block][grp]->data, data_mirror);
      }
    }
  }
  
  
  meshimporttimer.stop();
  if (verbosity>5 && Commptr->getRank() == 0) {
    cout << "mesh data import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh::meshDataImport" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

View_Sc2 MeshInterface::generateNewMicrostructure(int & randSeed) {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::generateNewMicrostructure ..." << endl;
    }
  }
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  have_rotations = true;
  have_rotation_phi = false;
  
  View_Sc2 seeds;
  randomSeeds.push_back(randSeed);
  std::default_random_engine generator(randSeed);
  numSeeds = 0;
  
  ////////////////////////////////////////////////////////////////////////////////
  // Generate the micro-structure using seeds and nearest neighbors
  ////////////////////////////////////////////////////////////////////////////////
  
  bool fast_and_crude = settings->sublist("Mesh").get<bool>("fast and crude microstructure",false);
  
  if (fast_and_crude) {
    int numxSeeds = settings->sublist("Mesh").get<int>("number of xseeds",10);
    int numySeeds = settings->sublist("Mesh").get<int>("number of yseeds",10);
    int numzSeeds = settings->sublist("Mesh").get<int>("number of zseeds",10);
    
    ScalarT xmin = settings->sublist("Mesh").get<ScalarT>("x min",0.0);
    ScalarT ymin = settings->sublist("Mesh").get<ScalarT>("y min",0.0);
    ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("z min",0.0);
    ScalarT xmax = settings->sublist("Mesh").get<ScalarT>("x max",1.0);
    ScalarT ymax = settings->sublist("Mesh").get<ScalarT>("y max",1.0);
    ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("z max",1.0);
    
    ScalarT dx = (xmax-xmin)/(ScalarT)(numxSeeds+1);
    ScalarT dy = (ymax-ymin)/(ScalarT)(numySeeds+1);
    ScalarT dz = (zmax-zmin)/(ScalarT)(numzSeeds+1);
    
    ScalarT maxpert = 0.25;
    
    Kokkos::View<ScalarT*,HostDevice> xseeds("xseeds",numxSeeds);
    Kokkos::View<ScalarT*,HostDevice> yseeds("yseeds",numySeeds);
    Kokkos::View<ScalarT*,HostDevice> zseeds("zseeds",numzSeeds);
    
    for (int k=0; k<numxSeeds; k++) {
      xseeds(k) = xmin + (k+1)*dx;
    }
    for (int k=0; k<numySeeds; k++) {
      yseeds(k) = ymin + (k+1)*dy;
    }
    for (int k=0; k<numzSeeds; k++) {
      zseeds(k) = zmin + (k+1)*dz;
    }
    
    std::uniform_real_distribution<ScalarT> pdistribution(-maxpert,maxpert);
    numSeeds = numxSeeds*numySeeds*numzSeeds;
    seeds = View_Sc2("seeds",numSeeds,3);
    auto seeds_host = create_mirror_view(seeds);
    
    int prog = 0;
    for (int i=0; i<numxSeeds; i++) {
      for (int j=0; j<numySeeds; j++) {
        for (int k=0; k<numzSeeds; k++) {
          ScalarT xp = pdistribution(generator);
          ScalarT yp = pdistribution(generator);
          ScalarT zp = pdistribution(generator);
          seeds_host(prog,0) = xseeds(i) + xp*dx;
          seeds_host(prog,1) = yseeds(j) + yp*dy;
          seeds_host(prog,2) = zseeds(k) + zp*dz;
          prog += 1;
        }
      }
    }
    deep_copy(seeds,seeds_host);
    
  }
  else {
    numSeeds = settings->sublist("Mesh").get<int>("number of seeds",10);
    seeds = View_Sc2("seeds",numSeeds,3);
    auto seeds_host = create_mirror_view(seeds);
    
    ScalarT xwt = settings->sublist("Mesh").get<ScalarT>("x weight",1.0);
    ScalarT ywt = settings->sublist("Mesh").get<ScalarT>("y weight",1.0);
    ScalarT zwt = settings->sublist("Mesh").get<ScalarT>("z weight",1.0);
    ScalarT nwt = sqrt(xwt*xwt+ywt*ywt+zwt*zwt);
    xwt *= 3.0/nwt;
    ywt *= 3.0/nwt;
    zwt *= 3.0/nwt;
    
    ScalarT xmin = settings->sublist("Mesh").get<ScalarT>("x min",0.0);
    ScalarT ymin = settings->sublist("Mesh").get<ScalarT>("y min",0.0);
    ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("z min",0.0);
    ScalarT xmax = settings->sublist("Mesh").get<ScalarT>("x max",1.0);
    ScalarT ymax = settings->sublist("Mesh").get<ScalarT>("y max",1.0);
    ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("z max",1.0);
    
    std::uniform_real_distribution<ScalarT> xdistribution(xmin,xmax);
    std::uniform_real_distribution<ScalarT> ydistribution(ymin,ymax);
    std::uniform_real_distribution<ScalarT> zdistribution(zmin,zmax);
    
    bool wellspaced = settings->sublist("Mesh").get<bool>("well spaced seeds",true);
    if (wellspaced) {
      // we use a relatively crude algorithm to obtain well-spaced points
      int batch_size = 10;
      int prog = 0;
      Kokkos::View<ScalarT**,HostDevice> cseeds("cand seeds",batch_size,3);
      
      while (prog<numSeeds) {
        // fill in the candidate seeds
        for (int k=0; k<batch_size; k++) {
          ScalarT x = xdistribution(generator);
          cseeds(k,0) = x;
          ScalarT y = ydistribution(generator);
          cseeds(k,1) = y;
          ScalarT z = zdistribution(generator);
          cseeds(k,2) = z;
        }
        int bestpt = 0;
        if (prog > 0) { // for prog = 0, just take the first one
          ScalarT maxdist = 0.0;
          for (int k=0; k<batch_size; k++) {
            ScalarT cmindist = 1.0e200;
            for (int j=0; j<prog; j++) {
              ScalarT dx = cseeds(k,0)-seeds(j,0);
              ScalarT dy = cseeds(k,1)-seeds(j,1);
              ScalarT dz = cseeds(k,2)-seeds(j,2);
              ScalarT cval = xwt*dx*dx + ywt*dy*dy + zwt*dz*dz;
              if (cval < cmindist) {
                cmindist = cval;
              }
            }
            if (cmindist > maxdist) {
              maxdist = cmindist;
              bestpt = k;
            }
          }
        }
        for (int j=0; j<3; j++) {
          seeds_host(prog,j) = cseeds(bestpt,j);
        }
        prog += 1;
      }
    }
    else {
      for (int k=0; k<numSeeds; k++) {
        ScalarT x = xdistribution(generator);
        seeds_host(k,0) = x;
        ScalarT y = ydistribution(generator);
        seeds_host(k,1) = y;
        ScalarT z = zdistribution(generator);
        seeds_host(k,2) = z;
      }
    }
    deep_copy(seeds, seeds_host);
    
  }
  //KokkosTools::print(seeds);
  
  meshimporttimer.stop();
  if (verbosity>5 && Commptr->getRank() == 0) {
    cout << "microstructure regeneration time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh::generateNewMicrostructure ..." << endl;
    }
  }
  
  return seeds;
}



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::importNewMicrostructure(int & randSeed, View_Sc2 seeds,
                                            vector<vector<Teuchos::RCP<Group> > > & groups,
                                            vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups) {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::importNewMicrostructure ..." << endl;
    }
  }
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  std::default_random_engine generator(randSeed);
  
  int numSeeds = seeds.extent(0);
  std::uniform_int_distribution<int> idistribution(0,100);
  Kokkos::View<int*,HostDevice> seedIndex("seed index",numSeeds);
  for (int i=0; i<numSeeds; i++) {
    int ci = idistribution(generator);
    seedIndex(i) = ci;
  }
  
  //KokkosTools::print(seedIndex);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  std::normal_distribution<ScalarT> ndistribution(0.0,1.0);
  Kokkos::View<ScalarT**,HostDevice> rotation_data("cell_data",numSeeds,numdata);
  for (int k=0; k<numSeeds; k++) {
    ScalarT x = ndistribution(generator);
    ScalarT y = ndistribution(generator);
    ScalarT z = ndistribution(generator);
    ScalarT w = ndistribution(generator);
    
    ScalarT r = sqrt(x*x + y*y + z*z + w*w);
    x *= 1.0/r;
    y *= 1.0/r;
    z *= 1.0/r;
    w *= 1.0/r;
    
    rotation_data(k,0) = w*w + x*x - y*y - z*z;
    rotation_data(k,1) = 2.0*(x*y - w*z);
    rotation_data(k,2) = 2.0*(x*z + w*y);
    
    rotation_data(k,3) = 2.0*(x*y + w*z);
    rotation_data(k,4) = w*w - x*x + y*y - z*z;
    rotation_data(k,5) = 2.0*(y*z - w*x);
    
    rotation_data(k,6) = 2.0*(x*z - w*y);
    rotation_data(k,7) = 2.0*(y*z + w*x);
    rotation_data(k,8) = w*w - x*x - y*y + z*z;
    
  }
  
  //KokkosTools::print(rotation_data);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initialize cell data
  ////////////////////////////////////////////////////////////////////////////////
  
  int totalElem = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      totalElem += numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      groups[block][grp]->data_seed = vector<size_t>(numElem);
      groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell nodes
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV totalNodes("nodes from all groups",totalElem,
                 groups[0][0]->nodes.extent(1),
                 groups[0][0]->nodes.extent(2));
  int prog = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      auto nodes = groups[block][grp]->nodes;
      parallel_for("mesh data cell nodes",
                   RangePolicy<AssemblyExec>(0,nodes.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nodes.extent(1); ++pt) {
          for (size_type dim=0; dim<nodes.extent(2); ++dim) {
            totalNodes(prog+elem,pt,dim) = nodes(elem,pt,dim);
          }
        }
      });
      prog += groups[block][grp]->numElem;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell centers
  ////////////////////////////////////////////////////////////////////////////////
  
  auto centers = this->getElementCenters(totalNodes, groups[0][0]->groupData->cellTopo);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Find the closest seeds
  ////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",totalElem);
  Kokkos::View<int*, CompadreDevice> cnode("cnode",totalElem);
  
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighborlists = CompadreTools_constructNeighborLists(seeds, centers, distance);
  cnode = neighborlists.getNeighborLists();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set group data
  ////////////////////////////////////////////////////////////////////////////////
  
  prog = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      //auto centers = this->getElementCenters(nodes, groups[block][grp]->groupData->cellTopo);
      
      //Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
      //Kokkos::View<int*, AssemblyDevice> cnode("cnode",numElem);
      //Compadre::NeighborLists<Kokkos::View<int*> > neighborlists = CompadreTools_constructNeighborLists(seeds, centers, distance);
      //cnode = neighborlists.getNeighborLists();

      for (int c=0; c<numElem; c++) {
        
        int cpt = cnode(prog);
        prog++;
        
        for (int i=0; i<9; i++) {
          groups[block][grp]->data(c,i) = rotation_data(cpt,i);//rotation_data(cnode(c),i);
        }
        
        groups[block][grp]->groupData->have_rotation = true;
        groups[block][grp]->groupData->have_phi = false;
        
        groups[block][grp]->data_seed[c] = cpt % 100;//cnode(c) % 100;
        groups[block][grp]->data_seedindex[c] = seedIndex(cpt); //seedIndex(cnode(c));
        groups[block][grp]->data_distance[c] = distance(cpt);//distance(c);
        
      }
    }
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initialize boundary data
  ////////////////////////////////////////////////////////////////////////////////
  
  totalElem = 0;
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      totalElem += numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell nodes
  ////////////////////////////////////////////////////////////////////////////////
  
  if (totalElem > 0) {
    
    totalNodes = DRV("nodes from all groups",totalElem,
                     groups[0][0]->nodes.extent(1),
                     groups[0][0]->nodes.extent(2));
    prog = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        auto nodes = boundary_groups[block][grp]->nodes;
        parallel_for("mesh data cell nodes",
                     RangePolicy<AssemblyExec>(0,nodes.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<nodes.extent(1); ++pt) {
            for (size_type dim=0; dim<nodes.extent(2); ++dim) {
              totalNodes(prog+elem,pt,dim) = nodes(elem,pt,dim);
            }
          }
        });
        prog += boundary_groups[block][grp]->numElem;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create a list of all cell centers
    ////////////////////////////////////////////////////////////////////////////////
    
    centers = this->getElementCenters(totalNodes, groups[0][0]->groupData->cellTopo);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Find the closest seeds
    ////////////////////////////////////////////////////////////////////////////////
    
    distance = Kokkos::View<ScalarT*, AssemblyDevice>("distance",totalElem);
    cnode = Kokkos::View<int*, CompadreDevice>("cnode",totalElem);
    neighborlists = CompadreTools_constructNeighborLists(seeds, centers, distance);
    cnode = neighborlists.getNeighborLists();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set data
    ////////////////////////////////////////////////////////////////////////////////
    
    prog = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        for (int c=0; c<numElem; c++) {
          
          int cpt = cnode(prog);
          prog++;
          
          for (int i=0; i<9; i++) {
            boundary_groups[block][grp]->data(c,i) = rotation_data(cpt,i);
          }
          
          boundary_groups[block][grp]->groupData->have_rotation = true;
          boundary_groups[block][grp]->groupData->have_phi = false;
          
          boundary_groups[block][grp]->data_seed[c] = cpt % 100;
          boundary_groups[block][grp]->data_seedindex[c] = seedIndex(cpt);
          boundary_groups[block][grp]->data_distance[c] = distance(cpt);
          
        }
      }
    }
    
  }
  
  meshimporttimer.stop();
  if (verbosity>5 && Commptr->getRank() == 0) {
    cout << "microstructure import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh::importNewMicrostructure" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV MeshInterface::getElemNodes(const int & block, const int & elemID) {
  vector<size_t> localIds;
  DRV blocknodes;
  
  panzer_stk::workset_utils::getIdsAndVertices(*stk_mesh, block_names[block], localIds, blocknodes);
  int nnodes = blocknodes.extent(1);
  
  DRV cnodes("element nodes",1,nnodes,spaceDim);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<spaceDim; j++) {
      cnodes(0,i,j) = blocknodes(elemID,i,j);
    }
  }
  return cnodes;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<string> MeshInterface::breakupList(const string & list, const string & delimiter) {
  // Script to break delimited list into pieces
  string tmplist = list;
  vector<string> terms;
  size_t pos = 0;
  if (tmplist.find(delimiter) == string::npos) {
    terms.push_back(tmplist);
  }
  else {
    string token;
    while ((pos = tmplist.find(delimiter)) != string::npos) {
      token = tmplist.substr(0, pos);
      terms.push_back(token);
      tmplist.erase(0, pos + delimiter.length());
    }
    terms.push_back(tmplist);
  }
  return terms;
}

/////////////////////////////////////////////////////////////////////////////
// Read in discretized data from an exodus mesh
/////////////////////////////////////////////////////////////////////////////

void MeshInterface::readExodusData() {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::readExodusData ..." << endl;
    }
  }
  
  string exofile;
  string fname;
  
  exofile = settings->sublist("Mesh").get<std::string>("mesh file","mesh.exo");
  
  if (Commptr->getSize() > 1) {
    std::stringstream ssProc, ssPID;
    ssProc << Commptr->getSize();
    ssPID << Commptr->getRank();
    string strProc = ssProc.str();
    string strPID = ssPID.str();
    // this section may need tweaking if the input exodus mesh is
    // spread across 10's, 100's, or 1000's (etc) of processors
    //if (Comm->MyPID() < 10)
    if (false)
      fname = exofile + "." + strProc + ".0" + strPID;
    else
      fname = exofile + "." + strProc + "." + strPID;
  }
  else {
    fname = exofile;
  }
  
  // open exodus file
  int CPU_word_size, IO_word_size, exoid, exo_error;
  int num_dim, num_nods, num_el, num_el_blk, num_ns, num_ss;
  char title[MAX_STR_LENGTH+1];
  float exo_version;
  CPU_word_size = sizeof(ScalarT);
  IO_word_size = 0;
  exoid = ex_open(fname.c_str(), EX_READ, &CPU_word_size,&IO_word_size,
                  &exo_version);
  exo_error = ex_get_init(exoid, title, &num_dim, &num_nods, &num_el,
                          &num_el_blk, &num_ns, &num_ss);
  
  if (exo_error>0) {
    // need some debug statement
  }
  int id = 1; // only one blkid
  int step = 1; // only one time step (for now)
  ex_block eblock;
  eblock.id = id;
  eblock.type = EX_ELEM_BLOCK;
  
  exo_error = ex_get_block_param(exoid, &eblock);
  
  int num_el_in_blk = eblock.num_entry;
  //int num_node_per_el = eblock.num_nodes_per_entry;
  
  
  // get elem vars
  if (settings->sublist("Mesh").get<bool>("have element data", false)) {
    int num_elem_vars = 0;
    int var_ind;
    numResponses = 1;
    //exo_error = ex_get_var_param(exoid, "e", &num_elem_vars); // TMW: this is depracated
    exo_error = ex_get_variable_param(exoid, EX_ELEM_BLOCK, &num_elem_vars); // TMW: this is depracated
    // This turns off this feature
    for (int i=0; i<num_elem_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      ScalarT *var_vals = new ScalarT[num_el_in_blk];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_ELEM_BLOCK, var_ind, varname);
      string vname(varname);
      efield_names.push_back(vname);
      size_t found = vname.find("Val");
      if (found != std::string::npos) {
        vector<string> results;
        std::stringstream sns, snr;
        int nr;
        results = this->breakupList(vname,"_");
        //boost::split(results, vname, [](char u){return u == '_';});
        snr << results[3];
        snr >> nr;
        numResponses = std::max(numResponses,nr);
      }
      efield_vals.push_back(vector<ScalarT>(num_el_in_blk));
      exo_error = ex_get_var(exoid,step,EX_ELEM_BLOCK,var_ind,id,num_el_in_blk,var_vals);
      for (int j=0; j<num_el_in_blk; j++) {
        efield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
  }
  
  /*
  // assign nodal vars to meas multivector
  if (settings->sublist("Mesh").get<bool>("have nodal data", false)) {
    int *connect = new int[num_el_in_blk*num_node_per_el];
    int edgeconn, faceconn;
    //exo_error = ex_get_elem_conn(exoid, id, connect);
    exo_error = ex_get_conn(exoid, EX_ELEM_BLOCK, id, connect, &edgeconn, &faceconn);
    
    // get nodal vars
    int num_node_vars = 0;
    int var_ind;
    //exo_error = ex_get_variable_param(exoid, EX_NODAL, &num_node_vars);
    // This turns off this feature
    for (int i=0; i<num_node_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      ScalarT *var_vals = new ScalarT[num_nods];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_NODAL, var_ind, varname);
      string vname(varname);
      nfield_names.push_back(vname);
      nfield_vals.push_back(vector<ScalarT>(num_nods));
      exo_error = ex_get_var(exoid,step,EX_NODAL,var_ind,0,num_nods,var_vals);
      for (int j=0; j<num_nods; j++) {
        nfield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
    
    
    meas = Teuchos::rcp(new Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>(LA_overlapped_map,1)); // empty solution
    size_t b = 0;
    //meas->sync<HostDevice>();
    auto meas_kv = meas->getLocalView<HostDevice>();
    
    //meas.modify_host();
    int index, dindex;
    
    auto dev_offsets = groups[b][0]->wkset->offsets;
    auto offsets = Kokkos::create_mirror_view(dev_offsets);
    Kokkos::deep_copy(offsets,dev_offsets);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      //cindex = groups[block][grp]->index;
      auto LIDs = groups[block][grp]->LIDs_host;
      auto nDOF = groups[block][grp]->groupData->numDOF_host;
      
      for (int n=0; n<nDOF(0); n++) {
        //Kokkos::View<GO**,HostDevice> GIDs = assembler->groups[block][grp]->GIDs;
        for (size_t p=0; p<groups[block][grp]->numElem; p++) {
          for( int i=0; i<nDOF(n); i++ ) {
            index = LIDs(p,offsets(n,i));//cindex(p,n,i);//LA_overlapped_map->getLocalElement(GIDs(p,curroffsets[n][i]));
            dindex = connect[e*num_node_per_el + i] - 1;
            meas_kv(index,0) = nfield_vals[n][dindex];
            //(*meas)[0][index] = nfield_vals[n][dindex];
          }
        }
      }
    }
    //meas.sync<>();
    delete [] connect;
    
  }
   */
  exo_error = ex_close(exoid);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh::readExodusData" << endl;
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we might be able to get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

void MeshInterface::purgeMemory() {
  
  mesh_factory.reset();
  nfield_vals.clear();
  efield_vals.clear();
  meas.reset();
  
  bool write_solution = settings->sublist("Postprocess").get("write solution",false);
  bool create_optim_movie = settings->sublist("Postprocess").get("create optimization movie",false);
  if (!write_solution && !create_optim_movie) {
    stk_mesh = Teuchos::null;
  }
}
