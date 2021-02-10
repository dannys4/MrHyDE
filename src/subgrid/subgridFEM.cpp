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

#include "subgridFEM.hpp"
#include "cell.hpp"

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

SubGridFEM::SubGridFEM(const Teuchos::RCP<MpiComm> & LocalComm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                       topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
                       ScalarT & macro_deltat_) :
settings(settings_), macro_cellTopo(macro_cellTopo_),
num_macro_time_steps(num_macro_time_steps_), macro_deltat(macro_deltat_) {
  
  LocalComm = LocalComm_;
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
  dimension = settings->sublist("Mesh").get<int>("dimension",2);
  multiscale_method = settings->get<string>("multiscale method","mortar");
  mesh_type = settings->sublist("Mesh").get<string>("mesh type","inline"); // or "Exodus" or "panzer"
  mesh_file = settings->sublist("Mesh").get<string>("mesh file","mesh.exo"); // or "Exodus" or "panzer"
  numrefine = settings->sublist("Mesh").get<int>("refinements",0);
  shape = settings->sublist("Mesh").get<string>("element type","quad");
  macroshape = settings->sublist("Mesh").get<string>("macro element type","quad");
  time_steps = settings->sublist("Solver").get<int>("number of steps",1);
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  string solver = settings->sublist("Solver").get<string>("solver","steady-state");
  if (solver == "steady-state") {
    final_time = 0.0;
  }
  soln = Teuchos::rcp(new SolutionStorage<SubgridSolverNode>(settings));
  adjsoln = Teuchos::rcp(new SolutionStorage<SubgridSolverNode>(settings));
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  
  if (settings->isParameter("Functions input file")) {
    std::string filename = settings->get<std::string>("Functions input file");
    std::ifstream fn(filename.c_str());
    if (fn.good()) {
      Teuchos::RCP<Teuchos::ParameterList> functions_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
      Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
      settings->setParameters( *functions_parlist );
    }
    else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
      TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the functions input file: " + filename);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Read-in any mesh-dependent data (from file)
  ////////////////////////////////////////////////////////////////////////////////
  
  have_mesh_data = false;
  have_rotation_phi = false;
  have_rotations = false;
  have_multiple_data_files = false;
  mesh_data_pts_tag = "mesh_data_pts";
  number_mesh_data_files = 1;
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",false);
    have_multiple_data_files = settings->sublist("Mesh").get<bool>("have multiple mesh data files",false);
    number_mesh_data_files = settings->sublist("Mesh").get<int>("number mesh data files",1);
  }
  
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

int SubGridFEM::addMacro(DRV & macronodes_,
                         Kokkos::View<int****,HostDevice> & macrosideinfo_,
                         LIDView macroLIDs_,
                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & macroorientation_) {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemTotalAddMacroTimer);
  
  Teuchos::RCP<SubGridMacroData> newdata = Teuchos::rcp( new SubGridMacroData(macronodes_,
                                                                              macrosideinfo_,
                                                                              macroLIDs_,
                                                                              macroorientation_) );
  macroData.push_back(newdata);
  
  int mID = macroData.size()-1;
  return mID;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::setUpSubgridModels() {
  
  Teuchos::TimeMonitor subgridsetuptimer(*sgfemTotalSetUpTimer);
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Starting SubgridFEM::setupSubgridModels ..." << endl;
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid mesh
  /////////////////////////////////////////////////////////////////////////////////////
 
  string blockID = "eblock";
  
  Kokkos::View<ScalarT**,HostDevice> nodes;
  vector<vector<GO> > connectivity;
  Kokkos::View<int****,HostDevice> sideinfo;
  
  vector<string> eBlocks;
  
  DRV refnodes("nodes on reference element",macroData[0]->macronodes.extent(1), dimension);
  CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *macro_cellTopo);

  SubGridTools sgt(LocalComm, macroshape, shape, refnodes,
                   macroData[0]->macrosideinfo, mesh_type, mesh_file);
  
  {
    Teuchos::TimeMonitor localmeshtimer(*sgfemSubMeshTimer);
    
    sgt.createSubMesh(numrefine);
    
    nodes = sgt.getListOfPhysicalNodes(macroData[0]->macronodes, macro_cellTopo);
    
    int reps = macroData[0]->macronodes.extent(0);
    connectivity = sgt.getPhysicalConnectivity(reps);
    
    sideinfo = sgt.getPhysicalSideinfo(macroData[0]->macrosideinfo);
    
    for (size_t c=0; c<sideinfo.extent(0); c++) { // number of elem in cell
      for (size_t i=0; i<sideinfo.extent(1); i++) { // number of variables
        for (size_t j=0; j<sideinfo.extent(2); j++) { // number of sides per element
          if (sideinfo(c,i,j,0) == 1) {
            sideinfo(c,i,j,0) = 5;
            sideinfo(c,i,j,1) = -1;
          }
        }
      }
    }
    
    panzer_stk::SubGridMeshFactory meshFactory(shape, nodes, connectivity, blockID);
    
    Teuchos::RCP<panzer_stk::STK_Interface> mesh = meshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
    
    mesh->getElementBlockNames(eBlocks);
    
    meshFactory.completeMeshConstruction(*mesh,*(LocalComm->getRawMpiComm()));
    
    sub_mesh = Teuchos::rcp(new meshInterface(settings, LocalComm) );
    sub_mesh->mesh = mesh;
    if (debug_level > 1) {
      if (LocalComm->getRank() == 0) {
        mesh->printMetaData(std::cout);
      }
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_physics = Teuchos::rcp( new physics(settings, LocalComm, sub_mesh->mesh) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the subgrid discretizations
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_disc = Teuchos::rcp( new discretization(settings, LocalComm, sub_mesh->mesh, sub_physics) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the function managers
  /////////////////////////////////////////////////////////////////////////////////////
  
  // Note that the workset size is determined by the number of elements per macro-element
  // times te number of macro-elements
  size_t numSubElem = connectivity.size();
  
  settings->sublist("Solver").set<int>("workset size",(int)numSubElem);
  vector<Teuchos::RCP<FunctionManager> > functionManagers;
  functionManagers.push_back(Teuchos::rcp(new FunctionManager(blockID,
                                                              numSubElem,
                                                              sub_disc->numip[0],
                                                              sub_disc->numip_side[0])));
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // Define the functions on each block
  ////////////////////////////////////////////////////////////////////////////////
  
  sub_physics->defineFunctions(functionManagers);
  
  ////////////////////////////////////////////////////////////////////////////////
  // The DOF-manager needs to be aware of the physics and the discretization(s)
  ////////////////////////////////////////////////////////////////////////////////
  
  //Teuchos::RCP<panzer::DOFManager> DOF = sub_disc->buildDOF(sub_mesh->mesh,
  //                                                          sub_physics->varlist,
  //                                                          sub_physics->types,
  //                                                          sub_physics->orders,
  //                                                          sub_physics->useDG);
  
  //sub_physics->setBCData(settings, sub_mesh->mesh, DOF, sub_disc->cards);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the parameter manager, the assembler and the solver
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_params = Teuchos::rcp( new ParameterManager<SubgridSolverNode>(LocalComm, settings, sub_mesh->mesh,
                                                                     sub_physics, sub_disc));
  
  sub_assembler = Teuchos::rcp( new AssemblyManager<SubgridSolverNode>(LocalComm, settings, sub_mesh->mesh,
                                                                       sub_disc, sub_physics, sub_params, numSubElem));
  
  cells = sub_assembler->cells;
  
  Teuchos::RCP<CellMetaData> cellData = sub_assembler->cellData[0];
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Boundary cells are not set up properly due to the lack of side sets in the subgrid mesh
  // These just need to be defined once though
  /////////////////////////////////////////////////////////////////////////////////////
  
  int numNodesPerElem = sub_mesh->cellTopo[0]->getNodeCount();
  vector<Teuchos::RCP<BoundaryCell> > newbcells;
  
  //int numLocalBoundaries = macroData[0]->macrosideinfo.extent(2);
  
  vector<int> unique_sides;
  vector<int> unique_local_sides;
  vector<string> unique_names;
  vector<vector<size_t> > boundary_groups;
  
  sgt.getUniqueSides(sideinfo, unique_sides, unique_local_sides, unique_names,
                     macrosidenames, boundary_groups);
  
  vector<stk::mesh::Entity> stk_meshElems;
  sub_mesh->mesh->getMyElements(blockID, stk_meshElems);
  
  // Does need to be PHX::Device
  Kokkos::View<const LO**,Kokkos::LayoutRight, PHX::Device> LIDs = sub_disc->DOF->getLIDs();
  
  for (size_t s=0; s<unique_sides.size(); s++) {
    
    string sidename = unique_names[s];
    vector<size_t> group = boundary_groups[s];
    
    size_t prog = 0;
    while (prog < group.size()) {
      size_t currElem = numSubElem;  // Avoid faults in last iteration
      if (prog+currElem > group.size()){
        currElem = group.size()-prog;
      }
      Kokkos::View<int*,AssemblyDevice> eIndex("element indices",currElem);
      Kokkos::View<int*,AssemblyDevice> sideIndex("local side indices",currElem);
      DRV currnodes("currnodes", currElem, numNodesPerElem, dimension);
      auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
      Kokkos::View<int*,HostDevice> host_eIndex2("element indices",currElem);
      auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
      auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
      for (size_t e=0; e<currElem; e++) {
        host_eIndex(e) = group[e+prog];
        host_sideIndex(e) = unique_local_sides[s];
        for (int n=0; n<numNodesPerElem; n++) {
          for (int m=0; m<dimension; m++) {
            host_currnodes(e,n,m) = nodes(connectivity[host_eIndex(e)][n],m);
          }
        }
      }
      int sideID = s;
     
      Kokkos::deep_copy(currnodes,host_currnodes);
      Kokkos::deep_copy(eIndex,host_eIndex);
      Kokkos::deep_copy(host_eIndex2,host_eIndex);
      Kokkos::deep_copy(sideIndex,host_sideIndex); 
      
      // Build the Kokkos View of the cell GIDs ------
      
      LIDView cellLIDs("LIDs on device", currElem,LIDs.extent(1));
      parallel_for("assembly copy LIDs",RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), KOKKOS_LAMBDA (const int i ) {
        size_t elemID = eIndex(i);
        for (size_type j=0; j<LIDs.extent(1); j++) {
          cellLIDs(i,j) = LIDs(elemID,j);
        }
      });
      
      //-----------------------------------------------
      // Set the side information (soon to be removed)-
      Kokkos::View<int****,HostDevice> sideinfo = sub_disc->getSideInfo(0,host_eIndex2);
      
      //-----------------------------------------------
      // Set the cell orientation ---
      Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices",
                                                                      currElem, numNodesPerElem);
      auto host_currind = Kokkos::create_mirror_view(currind);
      for (size_t i=0; i<currElem; i++) {
        vector<stk::mesh::EntityId> stk_nodeids;
        size_t elemID = host_eIndex(i);
        sub_mesh->mesh->getNodeIdsForElement(stk_meshElems[elemID], stk_nodeids);
        for (int n=0; n<numNodesPerElem; n++) {
          host_currind(i,n) = stk_nodeids[n];
        }
      }
      Kokkos::deep_copy(currind, host_currind);
      
      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orient_drv("kv to orients",currElem);
      Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, currind, *(sub_mesh->cellTopo[0]));
      
      newbcells.push_back(Teuchos::rcp(new BoundaryCell(cellData,currnodes,eIndex,sideIndex,
                                                        sideID, sidename, newbcells.size(),
                                                        cellLIDs, sideinfo)));//, orient_drv)));
      
      prog += currElem;
    }
    
    
  }
  
  sub_disc->setPhysicalData(cellData, newbcells);
  boundaryCells.push_back(newbcells);
  
  sub_assembler->boundaryCells = boundaryCells;
  
  Kokkos::View<string**,HostDevice> currbcs("boundary conditions",
                                         sideinfo.extent(1),
                                         macroData[0]->macrosideinfo.extent(2));
  for (size_t i=0; i<sideinfo.extent(1); i++) { // number of variables
    for (size_t j=0; j<macroData[0]->macrosideinfo.extent(2); j++) { // number of sides per element
      currbcs(i,j) = 5;
    }
  }
  for (size_t c=0; c<sideinfo.extent(0); c++) {
    for (size_t i=0; i<sideinfo.extent(1); i++) { // number of variables
      for (size_t j=0; j<sideinfo.extent(2); j++) { // number of sides per element
        if (sideinfo(c,i,j,0) > 1) { // TMW: should != 5
          for (size_t p=0; p<unique_sides.size(); p++) {
            if (unique_sides[p] == sideinfo(c,i,j,1)) {
              if (sideinfo(c,i,j,0) == 1) {
                currbcs(i,p) = "Dirichlet";
              }
              else if (sideinfo(c,i,j,0) == 2) {
                currbcs(i,p) = "Neumann";
              }
              else if (sideinfo(c,i,j,0) == 4) {
                currbcs(i,p) = "weak Dirichlet";
              }
              else if (sideinfo(c,i,j,0) == 5) {
                currbcs(i,p) = "interface";
              }
              //currbcs(i,p) = sideinfo(c,i,j,0);
            }
          }
        }
      }
    }
  }
  macroData[0]->bcs = currbcs;
  
  size_t numMacroDOF = macroData[0]->macroLIDs.extent(1);
  sub_solver = Teuchos::rcp( new SubGridFEM_Solver(LocalComm, settings, sub_mesh, sub_disc, sub_physics,
                                                   sub_assembler, sub_params, macro_deltat,
                                                   numMacroDOF) );
  
  sub_postproc = Teuchos::rcp( new PostprocessManager<SubgridSolverNode>(LocalComm, settings, sub_mesh->mesh,
                                                      sub_mesh->optimization_mesh,
                                                      sub_disc, sub_physics,
                                                      functionManagers, sub_assembler) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Create a subgrid function mananger
  /////////////////////////////////////////////////////////////////////////////////////
  
  {
    Teuchos::TimeMonitor localtimer(*sgfemLinearAlgebraSetupTimer);
    
    varlist = sub_physics->varlist[0];
    functionManagers[0]->setupLists(sub_physics->varlist[0], sub_physics->aux_varlist[0],
                                    macro_paramnames, macro_disc_paramnames);
    sub_assembler->wkset[0]->params_AD = paramvals_KVAD;
    
    functionManagers[0]->wkset = sub_assembler->wkset[0];
    
    functionManagers[0]->validateFunctions();
    functionManagers[0]->decomposeFunctions();
  }
  
  wkset = sub_assembler->wkset;
  
  wkset[0]->addAux(macro_varlist);
  Kokkos::View<int*,HostDevice> macro_numDOF_host("aux DOF on host",macro_numDOF.extent(0));
  auto macro_numDOF_m = Kokkos::create_mirror_view(macro_numDOF);
  Kokkos::deep_copy(macro_numDOF_m, macro_numDOF);
  Kokkos::deep_copy(macro_numDOF_host,macro_numDOF_m);
  
  for(size_t e=0; e<boundaryCells[0].size(); e++) {
    boundaryCells[0][e]->addAuxVars(macro_varlist);
    boundaryCells[0][e]->cellData->numAuxDOF = macro_numDOF;
    boundaryCells[0][e]->cellData->numAuxDOF_host = macro_numDOF_host;
    //boundaryCells[0][e]->cellData->numAuxDOF = macro_numDOF;
    boundaryCells[0][e]->setAuxUseBasis(macro_usebasis);
    boundaryCells[0][e]->auxoffsets = macro_offsets;
    boundaryCells[0][e]->wkset = wkset[0];
  }
  
  // TMW: would like to remove these since most of this is stored by the
  //      parameter manager
  
  vector<GO> params;
  if (sub_params->paramOwnedAndShared.size() == 0) {
    params.push_back(0);
  }
  else {
    params = sub_params->paramOwnedAndShared;
  }
  
  num_active_params = sub_params->getNumParams(1);
  num_stochclassic_params = sub_params->getNumParams(2);
  stochclassic_param_names = sub_params->getParamsNames(2);
  
  stoch_param_types = sub_params->stochastic_distribution;
  stoch_param_means = sub_params->getStochasticParams("mean");
  stoch_param_vars = sub_params->getStochasticParams("variance");
  stoch_param_mins = sub_params->getStochasticParams("min");
  stoch_param_maxs = sub_params->getStochasticParams("max");
  discparamnames = sub_params->discretized_param_names;
  
  
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  // Go through all of the macro-elements using this subgrid model and store
  // all of the local information
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  
  for (size_t mindex = 0; mindex<macroData.size(); mindex++) {
 
    // Define the cells and boundary cells for mindex>0
    if (mindex > 0) {
    
      // Use the subgrid mesh interface to define new nodes
      DRV newnodes = sgt.getPhysicalNodes(macroData[mindex]->macronodes, macro_cellTopo);
      
      //DRV newnodes = sgt.getNewNodes(macroData[mindex]->macronodes);
      
      int sprog = 0;
      
      // Redefine the sideinfo for the subcells
      Kokkos::View<int****,HostDevice> newsideinfo = sgt.getPhysicalSideinfo(macroData[mindex]->macrosideinfo);
      
      Kokkos::View<int****,HostDevice> subsideinfo("subcell side info", newsideinfo.extent(0), newsideinfo.extent(1),
                                                   newsideinfo.extent(2), newsideinfo.extent(3));
      
      for (size_t c=0; c<newsideinfo.extent(0); c++) { // number of elem in cell
        for (size_t i=0; i<newsideinfo.extent(1); i++) { // number of variables
          for (size_t j=0; j<newsideinfo.extent(2); j++) { // number of sides per element
            for (size_t k=0; k<newsideinfo.extent(3); k++) { // boundary information
              subsideinfo(c,i,j,k) = newsideinfo(sprog,i,j,k);
            }
            if (subsideinfo(c,i,j,0) == 1) {
              subsideinfo(c,i,j,0) = 5;
              subsideinfo(c,i,j,1) = -1;
            }
          }
        }
        sprog += 1;
        
      }
      
      ///////////////////////////////////////////////////////////
      // New cells
      ///////////////////////////////////////////////////////////
      
      vector<Teuchos::RCP<cell> > newcells;
      int numElem = newnodes.extent(0);
      int maxElem = cells[0][0]->numElem;
      
      Kokkos::View<LO*,AssemblyDevice> localID;
      LIDView LIDs;
      
      if (numElem == maxElem) { // reuse if possible
        localID = cells[0][0]->localElemID;
        LIDs = cells[0][0]->LIDs;
      }
      else { // subviews do not work, so performing a deep copy (should only be on last cell)
        localID = Kokkos::View<LO*,AssemblyDevice>("local elem ids",numElem);
        LIDs = LIDView("LIDs",numElem,cells[0][0]->LIDs.extent(1));
        Kokkos::View<LO*,AssemblyDevice> localID_0 = cells[0][0]->localElemID;
        LIDView LIDs_0 = cells[0][0]->LIDs;
        parallel_for("subgrid LIDs",RangePolicy<AssemblyExec>(0,numElem), KOKKOS_LAMBDA (const int e ) {
          localID(e) = localID_0(e);
          for (size_t j=0; j<LIDs.extent(1); j++) {
            LIDs(e,j) = LIDs_0(e,j);
          }
        });
          
      }
      newcells.push_back(Teuchos::rcp(new cell(sub_assembler->cellData[0],
                                               newnodes, localID,
                                               LIDs, subsideinfo)));
      
      //////////////////////////////////////////////////////////////
      // New boundary cells (more complicated than interior cells)
      //////////////////////////////////////////////////////////////
      
      vector<int> unique_sides;
      vector<int> unique_local_sides;
      vector<string> unique_names;
      vector<vector<size_t> > boundary_groups;
      
      sgt.getUniqueSides(subsideinfo, unique_sides, unique_local_sides, unique_names,
                         macrosidenames, boundary_groups);
      
      
      vector<Teuchos::RCP<BoundaryCell> > newbcells;
      for (size_t s=0; s<unique_sides.size(); s++) {
        vector<size_t> group = boundary_groups[s];
        Kokkos::View<size_t*,AssemblyDevice> group_KV("group members on device",group.size());
        auto group_KV_host = Kokkos::create_mirror_view(group_KV);
        for (size_t e=0; e<group.size(); e++) {
          group_KV_host(e) = group[e];
        }
        Kokkos::deep_copy(group_KV, group_KV_host);
        
        DRV currnodes("currnodes", group.size(), numNodesPerElem, dimension);
        
        parallel_for("subgrid bcell group",RangePolicy<AssemblyExec>(0,currnodes.extent(0)), KOKKOS_LAMBDA (const int e ) {
          size_t eIndex = group_KV(e);
          for (size_type n=0; n<currnodes.extent(1); n++) {
            for (size_type m=0; m<currnodes.extent(2); m++) {
              currnodes(e,n,m) = newnodes(eIndex,n,m);
            }
          }
        });
        
        int numElem = currnodes.extent(0);
        int maxElem = boundaryCells[0][s]->numElem;
        Kokkos::View<LO*,AssemblyDevice> localID;
        LIDView LIDs;
        Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
        Kokkos::View<LO*,AssemblyDevice> sideID;
        int sidenum = boundaryCells[0][s]->sidenum;
        
        if (numElem == maxElem) { // reuse if possible
          localID = boundaryCells[0][s]->localElemID;
          LIDs = boundaryCells[0][s]->LIDs;
          orientation = boundaryCells[0][s]->orientation;
          sideID = boundaryCells[0][s]->localSideID;
        }
        else { // subviews do not work, so performing a deep copy (should only be on last cell)
          localID = Kokkos::View<LO*,AssemblyDevice>("local elem ids",numElem);
          LIDs = LIDView("LIDs",numElem,boundaryCells[0][s]->LIDs.extent(1));
          orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("orientation",numElem);
          sideID = Kokkos::View<LO*,AssemblyDevice>("side IDs",numElem);
          
          Kokkos::View<LO*,AssemblyDevice> localID_0 = boundaryCells[0][s]->localElemID;
          LIDView LIDs_0 = boundaryCells[0][s]->LIDs;
          Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation_0 = boundaryCells[0][s]->orientation;
          Kokkos::View<LO*,AssemblyDevice> sideID_0 = boundaryCells[0][s]->localSideID;
          
          parallel_for("subgrid LIDs",RangePolicy<AssemblyExec>(0,numElem), KOKKOS_LAMBDA (const int e ) {
            localID(e) = localID_0(e);
            orientation(e) = orientation_0(e);
            sideID(e) = sideID_0(e);
            for (size_type j=0; j<LIDs.extent(1); j++) {
              LIDs(e,j) = LIDs_0(e,j);
            }
          });
          
        }
        
        newbcells.push_back(Teuchos::rcp(new BoundaryCell(sub_assembler->cellData[0], currnodes,
                                                          localID, sideID, sidenum, unique_names[s],
                                                          newbcells.size(), LIDs, subsideinfo)));//, orientation)));
        
      
        newbcells[s]->addAuxVars(macro_varlist);
        newbcells[s]->cellData->numAuxDOF = macro_numDOF;
        newbcells[s]->setAuxUseBasis(macro_usebasis);
        newbcells[s]->auxoffsets = macro_offsets;
        newbcells[s]->wkset = wkset[0];
        
      }
      sub_disc->setPhysicalData(sub_assembler->cellData[0], newcells);
      sub_disc->setPhysicalData(sub_assembler->cellData[0], newbcells);
      
      cells.push_back(newcells);
      
      boundaryCells.push_back(newbcells);
      
      Kokkos::View<string**,HostDevice> currbcs("boundary conditions",
                                             subsideinfo.extent(1),
                                             macroData[mindex]->macrosideinfo.extent(2));
      for (size_t i=0; i<subsideinfo.extent(1); i++) { // number of variables
        for (size_t j=0; j<macroData[mindex]->macrosideinfo.extent(2); j++) { // number of sides per element
          currbcs(i,j) = 5;
        }
      }
      for (size_type c=0; c<subsideinfo.extent(0); c++) {
        for (size_type i=0; i<subsideinfo.extent(1); i++) { // number of variables
          for (size_type j=0; j<subsideinfo.extent(2); j++) { // number of sides per element
            if (subsideinfo(c,i,j,0) > 1) { // TMW: should != 5
              for (size_t p=0; p<unique_sides.size(); p++) {
                if (unique_sides[p] == subsideinfo(c,i,j,1)) {
                  if (subsideinfo(c,i,j,0) == 1) {
                    currbcs(i,p) = "Dirichlet";
                  }
                  else if (subsideinfo(c,i,j,0) == 2) {
                    currbcs(i,p) = "Neumann";
                  }
                  else if (subsideinfo(c,i,j,0) == 4) {
                    currbcs(i,p) = "weak Dirichlet";
                  }
                  else if (subsideinfo(c,i,j,0) == 5) {
                    currbcs(i,p) = "interface";
                  }
                }
              }
            }
          }
        }
      }
      macroData[mindex]->bcs = currbcs;
      
      int numDOF = cells[mindex][0]->LIDs.extent(1);
      for (size_t e=0; e<cells[mindex].size(); e++) {
        cells[mindex][e]->setWorkset(sub_assembler->wkset[0]);
        cells[mindex][e]->setUseBasis(sub_solver->milo_solver->useBasis[0],
                                      sub_solver->milo_solver->numsteps,
                                      sub_solver->milo_solver->numstages);
        cells[mindex][e]->setUpAdjointPrev(numDOF,
                                           sub_solver->milo_solver->numsteps,
                                           sub_solver->milo_solver->numstages);
        cells[mindex][e]->setUpSubGradient(sub_solver->milo_solver->params->num_active_params);
      }
      if (boundaryCells.size() > mindex) { // should always be true here
        for (size_t e=0; e<boundaryCells[mindex].size(); e++) {
          if (boundaryCells[mindex][e]->numElem > 0) {
            boundaryCells[mindex][e]->setWorkset(sub_assembler->wkset[0]);
            boundaryCells[mindex][e]->setUseBasis(sub_solver->milo_solver->useBasis[0],
                                                  sub_solver->milo_solver->numsteps,
                                                  sub_solver->milo_solver->numstages);
          }
        }
      }
      
    } // if mindex == 0
    
    macroData[mindex]->setMacroIDs(cells[mindex][0]->numElem);
    
    // For all cells, define the macro basis functions at subgrid ip
    for (size_t e=0; e<boundaryCells[mindex].size(); e++) {
      vector<size_t> mIDs;
      Kokkos::View<size_t*,AssemblyDevice> mID_dev("mID device",boundaryCells[mindex][e]->localElemID.extent(0));
      auto mID_host = Kokkos::create_mirror_view(mID_dev);
      auto localEID = boundaryCells[mindex][e]->localElemID;
      auto macroIDs = macroData[mindex]->macroIDs;
      parallel_for("subgrid bcell mIDs",RangePolicy<AssemblyExec>(0,mID_dev.extent(0)), KOKKOS_LAMBDA (const int e ) {
        mID_dev(e) = macroIDs(localEID(e));
      });
      Kokkos::deep_copy(mID_host,mID_dev);
      for (size_type c=0; c<mID_host.extent(0); c++) {
        mIDs.push_back(mID_host(c));
      }
      boundaryCells[mindex][e]->auxMIDs = mIDs;
      boundaryCells[mindex][e]->auxMIDs_dev = mID_dev;
      size_t numElem = boundaryCells[mindex][e]->numElem;
      // define the macro LIDs
      LIDView cLIDs("boundary macro LIDs",numElem,
                    macroData[mindex]->macroLIDs.extent(1));
      auto cLIDs_host = Kokkos::create_mirror_view(cLIDs);
      auto macroLIDs_host = Kokkos::create_mirror_view(macroData[mindex]->macroLIDs);
      Kokkos::deep_copy(macroLIDs_host,macroData[mindex]->macroLIDs);
      for (size_t c=0; c<numElem; c++) {
        size_t mid = mIDs[c];
        for (size_type i=0; i<cLIDs.extent(1); i++) {
          cLIDs_host(c,i) = macroLIDs_host(mid,i);
        }
      }
      Kokkos::deep_copy(cLIDs,cLIDs_host);
      boundaryCells[mindex][e]->auxLIDs = cLIDs;
      LIDView_host cLIDs_host2("LIDs on host",cLIDs_host.extent(0),cLIDs_host.extent(1));
      Kokkos::deep_copy(cLIDs_host2,cLIDs_host);
      boundaryCells[mindex][e]->auxLIDs_host = cLIDs_host2;
    }
    
    //////////////////////////////////////////////////////////////
    // Set the initial conditions
    //////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubICTimer);
      
      Teuchos::RCP<SG_MultiVector> init = sub_solver->milo_solver->linalg->getNewOverlappedVector();
      //Teuchos::rcp(new SG_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
      this->setInitial(init, mindex, false);
      soln->store(init,initial_time,mindex);
      
      Teuchos::RCP<SG_MultiVector> inita = sub_solver->milo_solver->linalg->getNewOverlappedVector();
      //Teuchos::rcp(new SG_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
      adjsoln->store(inita,final_time,mindex);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // The current macro-element will store the values of its own basis functions
    // at the sub-grid integration points
    // Used to map the macro-scale solution to the sub-grid evaluation/integration pts
    ////////////////////////////////////////////////////////////////////////////////
   
    {
      Teuchos::TimeMonitor auxbasistimer(*sgfemComputeAuxBasisTimer);
      nummacroVars = macro_varlist.size();
      if (mindex == 0) {
        if (multiscale_method != "mortar" ) {
          // nothin yet
        }
        else {
          for (size_t e=0; e<boundaryCells[mindex].size(); e++) {
            
            size_t numElem = boundaryCells[mindex][e]->numElem;
            
            auto sside_ip = boundaryCells[mindex][e]->ip;//wkset->ip_side_vec[BIDs[e]];
            vector<DRV> currside_basis;
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis = DRV("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip.extent(1));
              currside_basis.push_back(tmp_basis);
            }
            int mcount = 0;
            for (size_t c=0; c<numElem; c++) {
              size_t mID = boundaryCells[mindex][e]->auxMIDs[c];
              if (mID == 0) {
                mcount++;
              }
            }
            vector<DRV> refbasis;
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis = DRV("basis values",mcount,macro_basis_pointers[i]->getCardinality(),sside_ip.extent(1));
              refbasis.push_back(tmp_basis);
            }
            DRV sref_side_ip("sref_side_ip", sside_ip.extent(1), sside_ip.extent(2));
            DRV side_ip_e("side_ip_e",1, sside_ip.extent(1), sside_ip.extent(2));
            DRV sref_side_ip_tmp("sref_side_ip_tmp",1, sside_ip.extent(1), sside_ip.extent(2));
            DRV cnodes("tmp nodes",1,macroData[mindex]->macronodes.extent(1),
                       macroData[mindex]->macronodes.extent(2));
            
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV basisvals("basisvals", macro_basis_pointers[i]->getCardinality(), sref_side_ip.extent(0));
              DRV basisvals_Transformed("basisvals_Transformed", 1, macro_basis_pointers[i]->getCardinality(), sref_side_ip.extent(0));
              for (int c=0; c<mcount; c++) {
                auto cip = Kokkos::subview(sside_ip,c,Kokkos::ALL(),Kokkos::ALL());
                auto sip = Kokkos::subview(side_ip_e,0,Kokkos::ALL(),Kokkos::ALL());
                Kokkos::deep_copy(sip,cip);
                auto mnodes = Kokkos::subview(macroData[mindex]->macronodes,0,Kokkos::ALL(),Kokkos::ALL());
                auto cnodes0 = Kokkos::subview(cnodes,0,Kokkos::ALL(), Kokkos::ALL());
                Kokkos::deep_copy(cnodes0,mnodes);
              
                CellTools::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, cnodes, *macro_cellTopo);
                auto sip_tmp0 = Kokkos::subview(sref_side_ip_tmp,0,Kokkos::ALL(),Kokkos::ALL());
                Kokkos::deep_copy(sref_side_ip,sip_tmp0);
              
                macro_basis_pointers[i]->getValues(basisvals, sref_side_ip, Intrepid2::OPERATOR_VALUE);
                
                FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
                auto crefbasis = Kokkos::subview(refbasis[i],c,Kokkos::ALL(),Kokkos::ALL());
                auto bvt0 = Kokkos::subview(basisvals_Transformed,0,Kokkos::ALL(),Kokkos::ALL());
                Kokkos::deep_copy(crefbasis,bvt0);
              }
            }
            int numIDs = numElem / mcount;
      
            Kokkos::View<int[1],PHX::Device> mcount_kv("view of mcount");
            Kokkos::deep_copy(mcount_kv,mcount);
            Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("tmp orientation",numElem);
            auto morient = macroData[mindex]->macroorientation;
            parallel_for("subgrid macro basis",RangePolicy<PHX::Device::execution_space>(0,numIDs), KOKKOS_LAMBDA (const int m ) {
              int mcount = mcount_kv(0);
              for (int n=0; n<mcount; n++) {
                int index= m*mcount+n;
                corientation(index) = morient(m);
              }
            });
            
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip.extent(1));
              auto rbasis = refbasis[i];
              parallel_for("subgrid macro basis",RangePolicy<PHX::Device::execution_space>(0,numIDs), KOKKOS_LAMBDA (const int m ) {
                int mcount = mcount_kv(0);
                for (int n=0; n<mcount; n++) {
                  int index= m*mcount+n;
                  for (size_type dof=0; dof<rbasis.extent(1); dof++) {
                    for (size_type pt=0; pt<rbasis.extent(2); pt++) {
                       tmp_basis(index,dof,pt) = rbasis(n,dof,pt);
                    }
                  }
                }
              });
     
              OrientTools::modifyBasisByOrientation(currside_basis[i], tmp_basis,
                                                    corientation, macro_basis_pointers[i].get());
                
            }

            boundaryCells[mindex][e]->auxside_basis = currside_basis;
          }
        }
      }
      else {
        if (multiscale_method != "mortar" ) {
          // nothin yet
        }
        else {
          for (size_t e=0; e<boundaryCells[mindex].size(); e++) {
            boundaryCells[mindex][e]->auxside_basis = boundaryCells[0][e]->auxside_basis;
          }
        }
      }
    }
    
  }
  
  sub_assembler->cells = cells;
  sub_assembler->boundaryCells = boundaryCells;
  sub_physics->setWorkset(wkset);
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Finished SubgridFEM::setupSubgridModels ..." << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::finalize(const int & globalSize, const int & globalPID) {
  
  // globalRank and globalPID are associated with the global MPI communicator
  // only needed to define a unique output file
  
  if (macroData.size() > 0) {
    this->setUpSubgridModels();
    
    size_t defblock = 0;
    if (cells.size() > 0) {
      sub_physics->setAuxVars(defblock, macro_varlist);
    }
  }
  
  std::stringstream ss;
  ss << globalSize << "." << globalPID;
  combined_mesh_filename = "subgrid_data/subgrid_combined_output." + ss.str() + ".exo";
  
  this->setupCombinedExodus();
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::addMeshData() {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemMeshDataTimer);
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Starting SubgridFEM::addMeshData ..." << endl;
    }
  }
  
  if (have_mesh_data) {
    
    int numdata = 1;
    if (have_rotations) {
      numdata = 9;
    }
    else if (have_rotation_phi) {
      numdata = 3;
    }
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        int numElem = cells[b][e]->numElem;
        Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
        cells[b][e]->cell_data = cell_data;
        cells[b][e]->cell_data_distance = vector<ScalarT>(numElem);
        cells[b][e]->cell_data_seed = vector<size_t>(numElem);
        cells[b][e]->cell_data_seedindex = vector<size_t>(numElem);
      }
    }
    
    for (int p=0; p<number_mesh_data_files; p++) {
      
      Teuchos::RCP<data> mesh_data;
      
      string mesh_data_pts_file;
      string mesh_data_file;
      
      if (have_multiple_data_files) {
        std::stringstream ss;
        ss << p+1;
        mesh_data_pts_file = mesh_data_pts_tag + "." + ss.str() + ".dat";
        mesh_data_file = mesh_data_tag + "." + ss.str() + ".dat";
      }
      else {
        mesh_data_pts_file = mesh_data_pts_tag + ".dat";
        mesh_data_file = mesh_data_tag + ".dat";
      }
      
      bool have_grid_data = settings->sublist("Mesh").get<bool>("data on grid",false);
      if (have_grid_data) {
        int Nx = settings->sublist("Mesh").get<int>("data grid Nx",0);
        int Ny = settings->sublist("Mesh").get<int>("data grid Ny",0);
        int Nz = settings->sublist("Mesh").get<int>("data grid Nz",0);
        mesh_data = Teuchos::rcp(new data("mesh data", dimension, mesh_data_pts_file,
                                          mesh_data_file, false, Nx, Ny, Nz));
        
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            size_t numElem = cells[b][e]->numElem;
            DRV nodes = cells[b][e]->nodes;
            for (size_t c=0; c<numElem; c++) {
              Kokkos::View<ScalarT**,AssemblyDevice> center("center",1,3);
              size_type numnodes = nodes.extent(1);
              for (size_type i=0; i<numnodes; i++) {
                for (int j=0; j<dimension; j++) {
                  center(0,j) += nodes(c,i,j)/(ScalarT)numnodes;
                }
              }
              ScalarT distance = 0.0;
              
              int cnode = mesh_data->findClosestGridNode(center(0,0), center(0,1), center(0,2), distance);
              
              bool iscloser = true;
              if (p>0){
                if (cells[b][e]->cell_data_distance[c] < distance) {
                  iscloser = false;
                }
              }
              if (iscloser) {
                Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode);
                
                for (size_type i=0; i<cdata.extent(1); i++) {
                  cells[b][e]->cell_data(c,i) = cdata(0,i);
                }
                cells[b][e]->cellData->have_extra_data = true;
                if (have_rotations)
                  cells[b][e]->cellData->have_cell_rotation = true;
                if (have_rotation_phi)
                  cells[b][e]->cellData->have_cell_phi = true;
                
                cells[b][e]->cell_data_seed[c] = cnode % 50;
                cells[b][e]->cell_data_distance[c] = distance;
              }
            }
          }
        }
      }
      else {
      
        mesh_data = Teuchos::rcp(new data("mesh data", dimension, mesh_data_pts_file,
                                          mesh_data_file, false));
        
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            size_t numElem = cells[b][e]->numElem;
            DRV nodes = cells[b][e]->nodes;
            Kokkos::View<ScalarT**, AssemblyDevice> center("center",numElem,3);
            for (size_t c=0; c<numElem; c++) {
              size_type numnodes = nodes.extent(1);
              for (size_type i=0; i<numnodes; i++) {
                for (int j=0; j<dimension; j++) {
                  center(c,j) += nodes(c,i,j)/(ScalarT)numnodes;
                }
              }
	    }
            
	    Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
	    Kokkos::View<int*, AssemblyDevice> cnode("cnode",numElem);  
            mesh_data->findClosestNode(center,cnode,distance);
              
            for(size_t c=0; c<numElem; ++c) {
              bool iscloser = true;
              if (p>0){
                if (cells[b][e]->cell_data_distance[c] < distance(c)) {
                  iscloser = false;
                }
              }
              if (iscloser) {
                Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode(c));
                
                for (size_type i=0; i<cdata.extent(1); i++) {
                  cells[b][e]->cell_data(c,i) = cdata(0,i);
                }
                cells[b][e]->cellData->have_extra_data = true;
                if (have_rotations)
                  cells[b][e]->cellData->have_cell_rotation = true;
                if (have_rotation_phi)
                  cells[b][e]->cellData->have_cell_phi = true;
                
                cells[b][e]->cell_data_seed[c] = cnode(c) % 50;
                cells[b][e]->cell_data_distance[c] = distance(c);
              }
            }
          }
        }
      }
    }
  }
  
  if (compute_mesh_data) {
    have_rotations = true;
    have_rotation_phi = false;
    
    Kokkos::View<ScalarT**,HostDevice> seeds;
    int randSeed = settings->sublist("Mesh").get<int>("random seed",1234);
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
      
      ScalarT maxpert = 0.2;
      
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
      seeds = Kokkos::View<ScalarT**,HostDevice>("seeds",numSeeds,3);
      int prog = 0;
      for (int i=0; i<numxSeeds; i++) {
        for (int j=0; j<numySeeds; j++) {
          for (int k=0; k<numzSeeds; k++) {
            ScalarT xp = pdistribution(generator);
            ScalarT yp = pdistribution(generator);
            ScalarT zp = pdistribution(generator);
            seeds(prog,0) = xseeds(i) + xp*dx;
            seeds(prog,1) = yseeds(j) + yp*dy;
            seeds(prog,2) = zseeds(k) + zp*dz;
            prog += 1;
          }
        }
      }
    }
    else {
      numSeeds = settings->sublist("Mesh").get<int>("number of seeds",1000);
      seeds = Kokkos::View<ScalarT**,HostDevice>("seeds",numSeeds,3);
      
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
          ScalarT mindist = 1.0e6;
          for (int k=0; k<batch_size; k++) {
            ScalarT cmindist = 1.0e6;
            for (int j=0; j<prog; j++) {
              ScalarT dx = cseeds(k,0)-seeds(j,0);
              ScalarT dy = cseeds(k,1)-seeds(j,1);
              ScalarT dz = cseeds(k,2)-seeds(j,2);
              ScalarT cval = sqrt(xwt*dx*dx + ywt*dy*dy + zwt*dz*dz);
              if (cval < cmindist) {
                cmindist = cval;
              }
            }
            if (cmindist<mindist) {
              mindist = cmindist;
              bestpt = k;
            }
          }
        }
        for (int j=0; j<3; j++) {
          seeds(prog,j) = cseeds(bestpt,j);
        }
        prog += 1;
      }
    }
    //KokkosTools::print(seeds);
    
    std::uniform_int_distribution<int> idistribution(0,50);
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
    
    
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        int numElem = cells[b][e]->numElem;
        Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
        cells[b][e]->cell_data = cell_data;
        cells[b][e]->cell_data_distance = vector<ScalarT>(numElem);
        cells[b][e]->cell_data_seed = vector<size_t>(numElem);
        cells[b][e]->cell_data_seedindex = vector<size_t>(numElem);
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set cell data
    ////////////////////////////////////////////////////////////////////////////////
    
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        DRV nodes = cells[b][e]->nodes;
        
        int numElem = cells[b][e]->numElem;
        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT[1][3],HostDevice> center("center");
          for (size_t i=0; i<nodes.extent(1); i++) {
            for (size_t j=0; j<nodes.extent(2); j++) {
              center(0,j) += nodes(c,i,j)/(ScalarT)nodes.extent(1);
            }
          }
          ScalarT distance = 1.0e6;
          int cnode = 0;
          for (int k=0; k<numSeeds; k++) {
            ScalarT dx = center(0,0)-seeds(k,0);
            ScalarT dy = center(0,1)-seeds(k,1);
            ScalarT dz = center(0,2)-seeds(k,2);
            ScalarT cdist = sqrt(dx*dx + dy*dy + dz*dz);
            if (cdist<distance) {
              cnode = k;
              distance = cdist;
            }
          }
          
          for (int i=0; i<9; i++) {
            cells[b][e]->cell_data(c,i) = rotation_data(cnode,i);
          }
          
          cells[b][e]->cell_data_seed[c] = cnode;
          cells[b][e]->cell_data_seedindex[c] = seedIndex(cnode);
          cells[b][e]->cell_data_distance[c] = distance;
          
        }
      }
    }
  }
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Finished SubgridFEM::addMeshData ..." << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::subgridSolver(Kokkos::View<ScalarT***,AssemblyDevice> coarse_fwdsoln,
                               Kokkos::View<ScalarT***,AssemblyDevice> coarse_adjsoln,
                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                               const bool & compute_jacobian, const bool & compute_sens,
                               const int & num_active_params,
                               const bool & compute_disc_sens, const bool & compute_aux_sens,
                               workset & macrowkset,
                               const int & usernum, const int & macroelemindex,
                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
  
  Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Starting SubgridFEM::subgridSolver ..." << endl;
    }
  }
  
  // Update the cells for this macro-element (or set of macro-elements)
  this->updateLocalData(usernum);
  
  // Copy the locak data (look into using subviews for this)
  // Solver does not know about localData
  Kokkos::View<ScalarT***,AssemblyDevice> coarse_u("local u",cells[usernum][0]->numElem,
                                                   coarse_fwdsoln.extent(1),
                                                   coarse_fwdsoln.extent(2));
  Kokkos::View<ScalarT***,AssemblyDevice> coarse_phi("local phi",cells[usernum][0]->numElem,
                                                     coarse_adjsoln.extent(1),
                                                     coarse_adjsoln.extent(2));
  
  // TMW: update for device (subgrid or assembly?)
  // Need to move localData[]->macroIDs to a Kokkos::View on the appropriate device
  auto macroIDs = macroData[usernum]->macroIDs;
  parallel_for("subgrid set coarse sol",
               RangePolicy<AssemblyExec>(0,coarse_u.extent(0)),
               KOKKOS_LAMBDA (const size_type e ) {
    for (size_type i=0; i<coarse_u.extent(1); i++) {
      for (size_type j=0; j<coarse_u.extent(2); j++) {
        coarse_u(e,i,j) = coarse_fwdsoln(macroIDs(e),i,j);
      }
    }
  });
  if (isAdjoint) {
    parallel_for("subgrid set coarse adj",
                 RangePolicy<AssemblyExec>(0,coarse_phi.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type i=0; i<coarse_phi.extent(1); i++) {
        for (size_type j=0; j<coarse_phi.extent(2); j++) {
          coarse_phi(e,i,j) = coarse_adjsoln(macroIDs(e),i,j);
        }
      }
    });
  }
  
  // Extract the previous solution as the initial guess/condition for subgrid problems
  Teuchos::RCP<SG_MultiVector> prev_fwdsoln, prev_adjsoln;
  {
    Teuchos::TimeMonitor localtimer(*sgfemInitialTimer);
    
    ScalarT prev_time = 0.0; // TMW: needed?
    //size_t numtimes = soln->times[usernum].size();
    if (isAdjoint) {
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_fwdsoln, usernum, time, prev_time);
        bool foundadj = adjsoln->extract(prev_adjsoln, usernum, time);
        if (!foundfwd || !foundadj) {
          // throw error
        }
      }
      else {
        bool foundfwd = soln->extract(prev_fwdsoln, usernum, time);
        bool foundadj = adjsoln->extract(prev_adjsoln, usernum, time);
        if (!foundfwd || !foundadj) {
          // throw error
        }
      }
    }
    else { // forward or compute sens
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_fwdsoln, usernum, time, prev_time);
        if (!foundfwd) { // this subgrid has not been solved at this time yet
          foundfwd = soln->extractLast(prev_fwdsoln, usernum, prev_time);
        }
      }
      else {
        bool foundfwd = soln->extractLast(prev_fwdsoln,usernum,prev_time);
        if (!foundfwd) {
          // throw error
        }
      }
      if (compute_sens) {
        double nexttime = 0.0;
        bool foundadj = adjsoln->extractNext(prev_adjsoln,usernum,time,nexttime);
        if (!foundadj) {
          // throw error
        }
      }
    }
  }
  
  // Containers for current forward/adjoint solutions
  //Teuchos::RCP<LA_MultiVector> curr_fwdsoln = Teuchos::rcp(new LA_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
  //Teuchos::RCP<LA_MultiVector> curr_adjsoln = Teuchos::rcp(new LA_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
  
  // Solve the local subgrid problem and fill in the coarse macrowkset->res;
  sub_solver->solve(coarse_u, coarse_phi,
                    prev_fwdsoln, prev_adjsoln, //curr_fwdsoln, curr_adjsoln,
                    Psol[0],
                    macroData[usernum], time, isTransient, isAdjoint, compute_jacobian,
                    compute_sens, num_active_params, compute_disc_sens, compute_aux_sens,
                    macrowkset, usernum, macroelemindex, subgradient, store_adjPrev);
  
  // Store the subgrid fwd or adj solution
  if (isAdjoint) {
    adjsoln->store(sub_solver->phi,time,usernum);
  }
  else if (!compute_sens) {
    soln->store(sub_solver->u,time,usernum);
  }
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Finished SubgridFEM::subgridSolver ..." << endl;
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux) {
  
  //int num_dof_lambda = lambda.extent(1)*lambda.extent(2);
  
  std::ofstream ofs;
  
  // Input data - macro DOFs
  ofs.open ("input_data.txt", std::ofstream::out | std::ofstream::app);
  ofs.precision(10);
  for (size_t e=0; e<lambda.extent(0); e++) {
    for (size_t i=0; i<lambda.extent(1); i++) {
      for (size_t j=0; j<lambda.extent(2); j++) {
        ofs << lambda(e,i,j) << "  ";
      }
    }
    ofs << endl;
  }
  ofs.close();
  
  // Output data - upscaled flux
  ofs.open ("output_data.txt", std::ofstream::out | std::ofstream::app);
  ofs.precision(10);
  for (size_t e=0; e<flux.extent(0); e++) {
    //for (size_t i=0; i<flux.extent(1); i++) {
    //for (size_t j=0; j<flux.extent(2); j++) {
    ofs << flux(e,0).val() << "  ";
    //}
    //}
    ofs << endl;
  }
  ofs.close();
  
}
 

//////////////////////////////////////////////////////////////
// Compute the initial values for the subgrid solution
//////////////////////////////////////////////////////////////

void SubGridFEM::setInitial(Teuchos::RCP<SG_MultiVector> & initial,
                            const int & usernum, const bool & useadjoint) {
  
  initial->putScalar(0.0);
  // TMW: uncomment if you need a nonzero initial condition
  //      right now, it slows everything down ... especially if using an L2-projection
  
  /*
   bool useL2proj = true;//settings->sublist("Solver").get<bool>("Project initial",true);
   
   if (useL2proj) {
   
   // Compute the L2 projection of the initial data into the discrete space
   Teuchos::RCP<LA_MultiVector> rhs = Teuchos::rcp(new LA_MultiVector(*overlapped_map,1)); // reset residual
   Teuchos::RCP<LA_CrsMatrix>  mass = Teuchos::rcp(new LA_CrsMatrix(Copy, *overlapped_map, -1)); // reset Jacobian
   Teuchos::RCP<LA_MultiVector> glrhs = Teuchos::rcp(new LA_MultiVector(*owned_map,1)); // reset residual
   Teuchos::RCP<LA_CrsMatrix>  glmass = Teuchos::rcp(new LA_CrsMatrix(Copy, *owned_map, -1)); // reset Jacobian
   
   
   //for (size_t b=0; b<cells.size(); b++) {
   for (size_t e=0; e<cells[usernum].size(); e++) {
   int numElem = cells[usernum][e]->numElem;
   vector<vector<int> > GIDs = cells[usernum][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[usernum][e]->getInitial(true, useadjoint);
   Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[usernum][e]->getMass();
   
   // assemble into global matrix
   for (int c=0; c<numElem; c++) {
   for( size_t row=0; row<GIDs[c].size(); row++ ) {
   int rowIndex = GIDs[c][row];
   ScalarT val = localrhs(c,row);
   rhs->SumIntoGlobalValue(rowIndex,0, val);
   for( size_t col=0; col<GIDs[c].size(); col++ ) {
   int colIndex = GIDs[c][col];
   ScalarT val = localmass(c,row,col);
   mass->InsertGlobalValues(rowIndex, 1, &val, &colIndex);
   }
   }
   }
   }
   //}
   
   
   mass->FillComplete();
   glmass->PutScalar(0.0);
   glmass->Export(*mass, *exporter, Add);
   
   glrhs->PutScalar(0.0);
   glrhs->Export(*rhs, *exporter, Add);
   
   glmass->FillComplete();
   
   Teuchos::RCP<LA_MultiVector> glinitial = Teuchos::rcp(new LA_MultiVector(*overlapped_map,1)); // reset residual
   
   this->linearSolver(glmass, glrhs, glinitial);
   
   initial->Import(*glinitial, *importer, Add);
   
   }
   else {
   
   for (size_t e=0; e<cells[usernum].size(); e++) {
   int numElem = cells[usernum][e]->numElem;
   vector<vector<int> > GIDs = cells[usernum][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localinit = cells[usernum][e]->getInitial(false, useadjoint);
   for (int c=0; c<numElem; c++) {
   for( size_t row=0; row<GIDs[c].size(); row++ ) {
   int rowIndex = GIDs[c][row];
   ScalarT val = localinit(c,row);
   initial->SumIntoGlobalValue(rowIndex,0, val);
   }
   }
   }
   
   }
  */
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the error for verification
///////////////////////////////////////////////////////////////////////////////////////

vector<std::pair<size_t, string> > SubGridFEM::getErrorList() {
  return sub_postproc->error_list[0];
}

///////////////////////////////////////////////////////////////////////////////////////
// These views are on the Host since we are using the postproc mananger
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT*,HostDevice> SubGridFEM::computeError(const ScalarT & time) {
  Kokkos::View<ScalarT*,HostDevice> errors;
  
  if (macroData.size() > 0) {
    
    errors = Kokkos::View<ScalarT*,HostDevice>("error", sub_postproc->error_list[0].size());
    
    bool compute = false;
    if (subgrid_static) {
      compute = true;
    }
    if (compute) {
      sub_postproc->computeError(time);
      for (size_t b=0; b<sub_postproc->errors[0].size(); b++) {
        Kokkos::View<ScalarT*,HostDevice> cerr = sub_postproc->errors[0][b];
        for (size_t etype=0; etype<cerr.extent(0); etype++) {
          errors(etype) += cerr(etype);
        }
      }
      sub_postproc->errors.clear();
    }
  }
  
  return errors;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> SubGridFEM::computeError(vector<std::pair<size_t, string> > & sub_error_list,
                                                            const vector<ScalarT> & times) {
  
  Kokkos::View<ScalarT**,HostDevice> errors;
  if (macroData.size() > 0) {
    
    errors = Kokkos::View<ScalarT**,HostDevice>("error", times.size(), sub_postproc->error_list[0].size());
    
    
    for (size_t t=0; t<times.size(); t++) {
      bool compute = false;
      if (subgrid_static) {
        compute = true;
      }
      if (compute) {
        sub_postproc->computeError(times[t]);
        for (size_t b=0; b<sub_postproc->errors[0].size(); b++) {
          Kokkos::View<ScalarT*,HostDevice> cerr = sub_postproc->errors[0][b];
          for (size_t etype=0; etype<cerr.extent(0); etype++) {
            errors(t,etype) += cerr(etype);
          }
        }
        sub_postproc->errors.clear();
      }
    }
  }
  
  return errors;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the objective function
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD*,AssemblyDevice> SubGridFEM::computeObjective(const string & response_type, const int & seedwhat,
                                                              const ScalarT & time, const int & usernum) {
  
  int tindex = -1;
  //for (int tt=0; tt<soln[usernum].size(); tt++) {
  //  if (abs(soln[usernum][tt].first - time)<1.0e-10) {
  //    tindex = tt;
  //  }
  //}
  
  Teuchos::RCP<SG_MultiVector> currsol;
  bool found = soln->extract(currsol,usernum,time,tindex);
  
  Kokkos::View<AD*,AssemblyDevice> objective;
  if (found) {
    this->updateLocalData(usernum);
    bool beensized = false;
    sub_solver->performGather(0, currsol, 0,0);
    //this->performGather(usernum, Psol[0], 4, 0);
    
    for (size_t e=0; e<cells[0].size(); e++) {
      auto curr_obj = cells[0][e]->computeObjective(time, tindex, seedwhat);
      if (!beensized && curr_obj.extent(1)>0) {
        objective = Kokkos::View<AD*,AssemblyDevice>("objective", curr_obj.extent(1));
        beensized = true;
      }
      for (size_t c=0; c<cells[0][e]->numElem; c++) {
        for (size_type i=0; i<curr_obj.extent(1); i++) {
          objective(i) += curr_obj(c,i);
        }
      }
    }
  }
  
  return objective;
}

///////////////////////////////////////////////////////////////////////////////////////
// Write the solution to a file
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::setupCombinedExodus() {
  
  Teuchos::TimeMonitor meshsetuptimer(*sgfemCombinedMeshSetupTimer);
  
  if (macroData.size() > 0) {
    bool isTD = false;
    string solver = settings->sublist("Solver").get<string>("solver","steady-state");
    if (solver == "transient") {
      isTD = true;
    }
    
    string blockID = "eblock";
    
    //////////////////////////////////////////////////////////////
    // Create a combined subgrid mesh
    //////////////////////////////////////////////////////////////
    
    // Create an initial mesh using the first macroelem
    DRV refnodes("nodes on reference element",macroData[0]->macronodes.extent(1), dimension);
    CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *macro_cellTopo);
    
    SubGridTools sgt(LocalComm, macroshape, shape, refnodes,//macroData[0]->macronodes,
                     macroData[0]->macrosideinfo, mesh_type, mesh_file);
    sgt.createSubMesh(numrefine);
    
    size_t numRefNodes = sgt.subnodes_list.extent(0);
    size_t numTotalNodes = 0;
    for (size_t usernum=0; usernum<macroData.size(); usernum++) {
      for (size_t e=0; e<macroData[usernum]->macronodes.extent(0); e++) {
        numTotalNodes += numRefNodes;
      }
    }
    //vector<vector<ScalarT> > comb_nodes;
    Kokkos::View<ScalarT**,HostDevice> comb_nodes("combined nodes",numTotalNodes,dimension);
    vector<vector<GO> > comb_connectivity;
    size_t nprog = 0;
    for (size_t usernum=0; usernum<macroData.size(); usernum++) {
      //vector<vector<ScalarT> > nodes = sgt.getNodes(macroData[usernum]->macronodes);
      Kokkos::View<ScalarT**,HostDevice> nodes = sgt.getListOfPhysicalNodes(macroData[usernum]->macronodes, macro_cellTopo);
      for (size_type n=0; n<nodes.extent(0); n++) {
        for (int s=0; s<dimension; s++) {
          comb_nodes(nprog+n,s) = nodes(n,s);
        }
      }
      GO num_prev_nodes = nprog;
      
      nprog += nodes.extent(0);
      //GO num_prev_nodes = static_cast<GO>(comb_nodes.size());
      //for (size_t n=0; n<nodes.size(); n++) {
      //  comb_nodes.push_back(nodes[n]);
      //}
      
      int reps = macroData[usernum]->macronodes.extent(0);
      vector<vector<GO> > connectivity = sgt.getPhysicalConnectivity(reps);
      for (size_t c=0; c<connectivity.size(); c++) {
        vector<GO> mod_elem;
        for (size_t n=0; n<connectivity[c].size(); n++) {
          mod_elem.push_back(connectivity[c][n]+num_prev_nodes);
        }
        comb_connectivity.push_back(mod_elem);
      }
    }
    //Kokkos::View<int****,HostDevice> sideinfo = sgt.getNewSideinfo(localData[usernum]->macrosideinfo);
    
    //size_t numNodesPerElem = comb_connectivity[0].size();
    
    panzer_stk::SubGridMeshFactory submeshFactory(shape, comb_nodes, comb_connectivity, blockID);
    combined_mesh = submeshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
    
    //////////////////////////////////////////////////////////////
    // Add in the necessary fields for plotting
    //////////////////////////////////////////////////////////////
    
    vector<string> vartypes = sub_physics->types[0];
    
    vector<string> subeBlocks;
    combined_mesh->getElementBlockNames(subeBlocks);
    for (size_t j=0; j<sub_physics->varlist[0].size(); j++) {
      if (vartypes[j] == "HGRAD") {
        combined_mesh->addSolutionField(sub_physics->varlist[0][j], subeBlocks[0]);
      }
      else if (vartypes[j] == "HVOL"){
        combined_mesh->addCellField(sub_physics->varlist[0][j], subeBlocks[0]);
      }
      else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL"){
        combined_mesh->addCellField(sub_physics->varlist[0][j]+"x", subeBlocks[0]);
        combined_mesh->addCellField(sub_physics->varlist[0][j]+"y", subeBlocks[0]);
        combined_mesh->addCellField(sub_physics->varlist[0][j]+"z", subeBlocks[0]);
        
        combined_mesh->addSolutionField(sub_physics->varlist[0][j]+"x", subeBlocks[0]);
        combined_mesh->addSolutionField(sub_physics->varlist[0][j]+"y", subeBlocks[0]);
        combined_mesh->addSolutionField(sub_physics->varlist[0][j]+"z", subeBlocks[0]);
      }
    }
    vector<string> subextrafieldnames = sub_physics->getExtraFieldNames(0);
    for (size_t j=0; j<subextrafieldnames.size(); j++) {
      combined_mesh->addSolutionField(subextrafieldnames[j], subeBlocks[0]);
    }
    vector<string> subextracellfields = sub_physics->getExtraCellFieldNames(0);
    for (size_t j=0; j<subextracellfields.size(); j++) {
      combined_mesh->addCellField(subextracellfields[j], subeBlocks[0]);
    }
    combined_mesh->addCellField("mesh_data_seed", subeBlocks[0]);
    combined_mesh->addCellField("mesh_data", subeBlocks[0]);
    
    if (discparamnames.size() > 0) {
      for (size_t n=0; n<discparamnames.size(); n++) {
        int paramnumbasis = cells[0][0]->cellData->numParamDOF.extent(0);
        if (paramnumbasis==1) {
          combined_mesh->addCellField(discparamnames[n], subeBlocks[0]);
        }
        else {
          combined_mesh->addSolutionField(discparamnames[n], subeBlocks[0]);
        }
      }
    }
    
    //////////////////////////////////////////////////////////////
    // Finalize the mesh
    //////////////////////////////////////////////////////////////
    
    combined_mesh->initialize(*(LocalComm->getRawMpiComm()));
    submeshFactory.modifyMesh(*combined_mesh);
    combined_mesh->buildLocalElementIDs();
    
    //////////////////////////////////////////////////////////////
    // Set up the output for transient data
    //////////////////////////////////////////////////////////////
    
    if (isTD) {
      combined_mesh->setupExodusFile(combined_mesh_filename);
    }
    
  }
}

//////////////////////////////////////////////////////////////
// Write the current states to the combined output file
//////////////////////////////////////////////////////////////

void SubGridFEM::writeSolution(const ScalarT & time) {

  Teuchos::TimeMonitor outputtimer(*sgfemCombinedMeshOutputTimer);
  
  if (macroData.size()>0) {
    
    bool isTD = false;
    string solver = settings->sublist("Solver").get<string>("solver","steady-state");
    if (solver == "transient") {
      isTD = true;
    }
    
    vector<size_t> myElements;
    size_t eprog = 0;
    for (size_t usernum=0; usernum<cells.size(); usernum++) {
      for (size_t e=0; e<cells[usernum].size(); e++) {
        for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
          myElements.push_back(eprog);
          eprog++;
        }
      }
    }
    
    string blockID = "eblock";
    topo_RCP cellTopo = combined_mesh->getCellTopology(blockID);
    size_t numNodesPerElem = cellTopo->getNodeCount();
    
    Kokkos::View<int**,AssemblyDevice> offsets = wkset[0]->offsets;
    Kokkos::View<int*,AssemblyDevice> numDOF = sub_assembler->cellData[0]->numDOF;
    vector<string> vartypes = sub_physics->types[0];
    vector<string> varlist = sub_physics->varlist[0];
    
    // Collect the subgrid solution
    for (size_t n = 0; n<varlist.size(); n++) {
      
      if (vartypes[n] == "HGRAD") {
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",myElements.size(), numNodesPerElem);
        
        size_t pprog = 0;
        for (size_t usernum=0; usernum<cells.size(); usernum++) {
          for( size_t e=0; e<cells[usernum].size(); e++ ) {
            Kokkos::View<ScalarT***,AssemblyDevice> sol = cells[usernum][e]->u;
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
              for( size_t i=0; i<numNodesPerElem; i++ ) {
                soln_computed(pprog,i) = host_sol(p,n,i);
              }
              pprog += 1;
            }
          }
        }
        combined_mesh->setSolutionFieldData(varlist[n], blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HVOL") {
        
        Kokkos::View<ScalarT*,HostDevice> soln_computed("soln",myElements.size());
        size_t pprog = 0;
        for( size_t usernum=0; usernum<cells.size(); usernum++ ) {
          for( size_t e=0; e<cells[usernum].size(); e++ ) {
            Kokkos::View<ScalarT***,AssemblyDevice> sol = cells[usernum][e]->u;
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
              soln_computed(pprog) = host_sol(p,n,0);
              pprog++;
            }
          }
        }
        combined_mesh->setCellFieldData(varlist[n], blockID, myElements, soln_computed);
        
      }
      else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") {
        
        Kokkos::View<ScalarT*,HostDevice> soln_x("soln",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> soln_y("soln",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> soln_z("soln",myElements.size());
        size_t pprog = 0;
        
        for( size_t usernum=0; usernum<cells.size(); usernum++ ) {
          for( size_t e=0; e<cells[usernum].size(); e++ ) {
            Kokkos::View<ScalarT***,AssemblyDevice> sol = cells[usernum][e]->u_avg;
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
              soln_x(pprog) = host_sol(p,n,0);
              if (dimension > 1) {
                soln_y(pprog) = host_sol(p,n,1);
              }
              if (dimension > 2) {
                soln_z(pprog) = host_sol(p,n,2);
              }
              pprog++;
            }
          }
        }
        combined_mesh->setCellFieldData(varlist[n]+"x", blockID, myElements, soln_x);
        combined_mesh->setCellFieldData(varlist[n]+"y", blockID, myElements, soln_y);
        combined_mesh->setCellFieldData(varlist[n]+"z", blockID, myElements, soln_z);
        
        if (sub_assembler->cellData[0]->requireBasisAtNodes) {
          Kokkos::View<ScalarT**,HostDevice> soln_nx("soln",myElements.size(), numNodesPerElem);
          Kokkos::View<ScalarT**,HostDevice> soln_ny("soln",myElements.size(), numNodesPerElem);
          Kokkos::View<ScalarT**,HostDevice> soln_nz("soln",myElements.size(), numNodesPerElem);
          
          pprog = 0;
          for (size_t usernum=0; usernum<cells.size(); usernum++) {
            for( size_t e=0; e<cells[usernum].size(); e++ ) {
              Kokkos::View<ScalarT***,AssemblyDevice> sol = cells[usernum][e]->getSolutionAtNodes(n);
              auto host_sol = Kokkos::create_mirror_view(sol);
              Kokkos::deep_copy(host_sol,sol);
              for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
                for( size_t i=0; i<numNodesPerElem; i++ ) {
                  soln_nx(pprog,i) = host_sol(p,i,0);
                  if (dimension > 1) {
                    soln_ny(pprog,i) = host_sol(p,i,1);
                  }
                  if (dimension > 2) {
                    soln_nz(pprog,i) = host_sol(p,i,2);
                  }
                }
                pprog += 1;
              }
            }
          }
          combined_mesh->setSolutionFieldData(varlist[n]+"x", blockID, myElements, soln_nx);
          combined_mesh->setSolutionFieldData(varlist[n]+"y", blockID, myElements, soln_ny);
          combined_mesh->setSolutionFieldData(varlist[n]+"z", blockID, myElements, soln_nz);
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////
    // Mesh data
    ////////////////////////////////////////////////////////////////
    
    
    Kokkos::View<ScalarT*,HostDevice> cseeds("cell data seeds",myElements.size());
    Kokkos::View<ScalarT*,HostDevice> cdata("cell data",myElements.size());
    
    if (cells[0][0]->cellData->have_cell_phi || cells[0][0]->cellData->have_cell_rotation || cells[0][0]->cellData->have_extra_data) {
      int eprog = 0;
      // TMW: need to use a mirror view here
      for (size_t usernum=0; usernum<cells.size(); usernum++) {
        for (size_t e=0; e<cells[usernum].size(); e++) {
          vector<size_t> cell_data_seed = cells[usernum][e]->cell_data_seed;
          vector<size_t> cell_data_seedindex = cells[usernum][e]->cell_data_seedindex;
          Kokkos::View<ScalarT**,AssemblyDevice> cell_data = cells[usernum][e]->cell_data;
          for (size_t p=0; p<cells[usernum][0]->numElem; p++) {
            cseeds(eprog) = cell_data_seedindex[p];
            cdata(eprog) = cell_data(p,0);
            eprog++;
          }
        }
      }
    }
    combined_mesh->setCellFieldData("mesh_data_seed", blockID, myElements, cseeds);
    combined_mesh->setCellFieldData("mesh_data", blockID, myElements, cdata);
    
    
    ////////////////////////////////////////////////////////////////
    // Extra nodal fields
    ////////////////////////////////////////////////////////////////
    
    vector<string> extrafieldnames = sub_physics->getExtraFieldNames(0);
    for (size_t j=0; j<extrafieldnames.size(); j++) {
      Kokkos::View<ScalarT**,HostDevice> efdata("field data",myElements.size(), numNodesPerElem);
      size_t eprog = 0;
      for (size_t usernum=0; usernum<cells.size(); usernum++) {
        for (size_t e=0; e<cells[usernum].size(); e++) {
          DRV nodes = cells[usernum][e]->nodes;
          Kokkos::View<ScalarT**,AssemblyDevice> cfields = sub_physics->getExtraFields(0, 0, nodes, time, wkset[0]);
          auto host_cfields = Kokkos::create_mirror_view(cfields);
          Kokkos::deep_copy(host_cfields,cfields);
          for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
            for (size_t i=0; i<host_cfields.extent(1); i++) {
              efdata(eprog,i) = host_cfields(p,i);
            }
            eprog++;
          }
        }
      }
      combined_mesh->setSolutionFieldData(extrafieldnames[j], blockID, myElements, efdata);
    }
    
    ////////////////////////////////////////////////////////////////
    // Extra cell fields
    ////////////////////////////////////////////////////////////////
    
    vector<string> extracellfieldnames = sub_physics->getExtraCellFieldNames(0);
    
    for (size_t j=0; j<extracellfieldnames.size(); j++) {
      Kokkos::View<ScalarT*,HostDevice> efdata("cell data",myElements.size());
      
      int eprog = 0;
      for (size_t usernum=0; usernum<cells.size(); usernum++) {
        for (size_t e=0; e<cells[usernum].size(); e++) {
        
          cells[usernum][e]->updateData();
          cells[usernum][e]->updateWorksetBasis();
          wkset[0]->time = time;
          wkset[0]->computeSolnSteadySeeded(cells[usernum][e]->u, 0);
          wkset[0]->computeSolnVolIP();
          wkset[0]->computeParamVolIP(cells[usernum][e]->param, 0);
          
          Kokkos::View<ScalarT*,AssemblyDevice> cfields = sub_physics->getExtraCellFields(0, j, cells[usernum][e]->wts);
          
          auto host_cfields = Kokkos::create_mirror_view(cfields);
          Kokkos::deep_copy(host_cfields, cfields);
          for (size_type p=0; p<host_cfields.extent(0); p++) {
            efdata(eprog) = host_cfields(p);
            eprog++;
          }
        }
      }
      combined_mesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, efdata);
    }
    
    
    if (isTD) {
      combined_mesh->writeToExodus(time);
    }
    else {
      combined_mesh->writeToExodus(combined_mesh_filename);
    }
    
  }
  
}


////////////////////////////////////////////////////////////////////////////////
// Add in the sensor data
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                            const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                            const vector<basis_RCP> & basisTypes, const int & usernum) {
  for (size_t e=0; e<cells[usernum].size(); e++) {
    //cells[usernum][e]->addSensors(sensor_points,sensor_loc_tol,sensor_data,
    //                              have_sensor_data, sub_disc, basisTypes, basisTypes);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridFEM::getProjectionMatrix() {
  
  return sub_solver->getProjectionMatrix();
  
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> > SubGridFEM::getProjectionMatrix(DRV & ip, DRV & wts,
                                                           std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  return sub_solver->getProjectionMatrix(ip, wts, other_basisinfo);
  
}

////////////////////////////////////////////////////////////////////////////////
// Get an empty vector
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> > SubGridFEM::getVector() {
  return sub_solver->getVector();
}

////////////////////////////////////////////////////////////////////////////////
// Get the integration points
////////////////////////////////////////////////////////////////////////////////

DRV SubGridFEM::getIP() {
  int numip_per_cell = wkset[0]->numip;
  int usernum = 0; // doesn't really matter
  int totalip = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    totalip += numip_per_cell*cells[usernum][e]->numElem;
  }
  
  DRV refip = DRV("refip",1,totalip,dimension);
  int prog = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    size_t numElem = cells[usernum][e]->numElem;
    auto ip = cells[usernum][e]->ip;
    for (size_t c=0; c<numElem; c++) {
      for (size_type i=0; i<ip.extent(1); i++) {
        for (size_type j=0; j<ip.extent(2); j++) {
          refip(0,prog,j) = ip(c,i,j);
        }
        prog++;
      }
    }
  }
  return refip;
  
}

////////////////////////////////////////////////////////////////////////////////
// Get the integration weights
////////////////////////////////////////////////////////////////////////////////

DRV SubGridFEM::getIPWts() {
  int numip_per_cell = wkset[0]->numip;
  int usernum = 0; // doesn't really matter
  int totalip = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    totalip += numip_per_cell*cells[usernum][e]->numElem;
  }
  DRV refwts = DRV("refwts",1,totalip);
  int prog = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    DRV wts = cells[0][e]->cellData->ref_wts;//wkset[0]->ref_wts;//cells[usernum][e]->ijac;
    size_t numElem = cells[usernum][e]->numElem;
    for (size_t c=0; c<numElem; c++) {
      for (size_type i=0; i<wts.extent(0); i++) {
        refwts(0,prog) = wts(i);//sref_ip_tmp(0,i,j);
        prog++;
      }
    }
  }
  return refwts;
  
}


////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
////////////////////////////////////////////////////////////////////////////////

std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numLIDs = cells[0][0]->LIDs.extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numLIDs);
  
  for (size_t e=0; e<cells[0].size(); e++) {
    int numElem = cells[0][e]->numElem;
    DRV nodes = cells[0][e]->nodes;
    for (int c=0; c<numElem;c++) {
      DRV refpts("refpts",1, numpts, dimpts);
      Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell",1,numpts);
      DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
      for (unsigned int i=0; i<nodes.extent(1); i++) {
        for (unsigned int j=0; j<nodes.extent(2); j++) {
          cnodes(0,i,j) = nodes(c,i,j);
        }
      }
      
      CellTools::mapToReferenceFrame(refpts, pts, cnodes, *(sub_mesh->cellTopo[0]));
      CellTools::checkPointwiseInclusion(inRefCell, refpts, *(sub_mesh->cellTopo[0]), 1.0e-12);
      for (size_t i=0; i<numpts; i++) {
        if (inRefCell(0,i) == 1) {
          owners(i,0) = e;//cells[0][e]->localElemID[c];
          owners(i,1) = c;
          LIDView LIDs = cells[0][e]->LIDs;
          for (size_t j=0; j<numLIDs; j++) {
            owners(i,j+2) = LIDs(c,j);
          }
        }
      }
    }
  }
  
  vector<DRV> ptsBasis;
  for (size_t i=0; i<numpts; i++) {
    vector<DRV> currBasis;
    DRV refpt_buffer("refpt_buffer",1,1,dimpts);
    DRV cpt("cpt",1,1,dimpts);
    for (size_t s=0; s<dimpts; s++) {
      cpt(0,0,s) = pts(0,i,s);
    }
    DRV nodes = cells[0][owners(i,0)]->nodes;
    DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
    for (unsigned int k=0; k<nodes.extent(1); k++) {
      for (unsigned int j=0; j<nodes.extent(2); j++) {
        cnodes(0,k,j) = nodes(owners(i,1),k,j);
      }
    }
    CellTools::mapToReferenceFrame(refpt_buffer, cpt, cnodes, *(sub_mesh->cellTopo[0]));
    DRV refpt("refpt",1,dimpts);
    Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
    Kokkos::View<int**,AssemblyDevice> offsets = wkset[0]->offsets;
    vector<int> usebasis = wkset[0]->usebasis;
    DRV basisvals("basisvals",offsets.extent(0),numLIDs);
    for (size_t n=0; n<offsets.extent(0); n++) {
      DRV bvals = sub_disc->evaluateBasis(sub_disc->basis_pointers[0][usebasis[n]], refpt);
      for (size_t m=0; m<offsets.extent(1); m++) {
        basisvals(n,offsets(n,m)) = bvals(0,m,0);
      }
    }
    ptsBasis.push_back(basisvals);
    
  }
  std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}

////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
// TMW: what is this function for???
////////////////////////////////////////////////////////////////////////////////

//std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM::evaluateBasis(const DRV & pts) {
  // this function is deprecated
//}


////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridFEM::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map) {
  return sub_solver->getEvaluationMatrix(newip, ip_map);
  /*
  matrix_RCP map_over = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph) );
  matrix_RCP map;
  if (LocalComm->getSize() > 1) {
    size_t maxEntries = 256;
    map = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_owned_map, maxEntries) );
    
    map->setAllToScalar(0.0);
    map->doExport(*map_over, *(sub_solver->exporter), Tpetra::ADD);
    map->fillComplete();
  }
  else {
    map = map_over;
  }
  return map;
   */
}

////////////////////////////////////////////////////////////////////////////////
// Get the subgrid cell GIDs
////////////////////////////////////////////////////////////////////////////////

LIDView SubGridFEM::getCellLIDs(const int & cellnum) {
  return cells[0][cellnum]->LIDs;
}

////////////////////////////////////////////////////////////////////////////////
// Update the subgrid parameters (will be depracated)
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  for (size_t b=0; b<wkset.size(); b++) {
    wkset[b]->params = params;
    wkset[b]->paramnames = paramnames;
  }
  sub_physics->updateParameters(params, paramnames);
  
}

////////////////////////////////////////////////////////////////////////////////
// TMW: Is the following functions used/required ???
////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> SubGridFEM::getCellFields(const int & usernum, const ScalarT & time) {
  
  Kokkos::View<ScalarT**,AssemblyDevice> nothing("empty view",1,1);
  return nothing;
  
  /*
   vector<string> extracellfieldnames = sub_physics->getExtraCellFieldNames(0);
   FC extracellfields(cells[usernum].size(),extracellfieldnames.size());
   
   int timeindex = 0;
   for (size_t k=0; k<soln[usernum].size(); k++) {
   if (abs(time-soln[usernum][k].first)<1.0e-10) {
   timeindex = k;
   }
   }
   
   for (size_t k=0; k<cells[usernum].size(); k++) {
   cells[usernum][k]->updateSolnWorkset(soln[usernum][timeindex].second, 0); // also updates ip, ijac
   wkset[0]->time = soln[usernum][timeindex].first;
   cells[usernum][k]->updateData();
   vector<FC> cfields = sub_physics->getExtraCellFields(0);
   size_t j = 0;
   for (size_t g=0; g<cfields.size(); g++) {
   for (size_t h=0; h<cfields[g].extent(0); h++) {
   extracellfields(k,j) = cfields[g](h,0);
   ++j;
   }
   }
   }
   
   return extracellfields;
   */
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
  /*
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      int numElem = cells[b][e]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = loc[b]->cell_data_seed[c];
        for (int i=0; i<9; i++) {
          cells[0][e]->cell_data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }*/
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::updateLocalData(const int & usernum) {
  
  wkset[0]->var_bcs = macroData[usernum]->bcs;
  
  /*
  for (size_t e=0; e<cells[0].size(); e++) {
    cells[0][e]->nodes = localData[usernum]->nodes;
    cells[0][e]->ip = localData[usernum]->ip;
    cells[0][e]->wts = localData[usernum]->wts;
    cells[0][e]->hsize = localData[usernum]->hsize;
    cells[0][e]->basis = localData[usernum]->basis;
    cells[0][e]->basis_grad = localData[usernum]->basis_grad;
    cells[0][e]->basis_div = localData[usernum]->basis_div;
    cells[0][e]->basis_curl = localData[usernum]->basis_curl;
    cells[0][e]->sideinfo = localData[usernum]->sideinfo;
    cells[0][e]->cell_data = localData[usernum]->cell_data;
  }
  
  for (size_t e=0; e<boundaryCells[0].size(); e++) {
    boundaryCells[0][e]->nodes = localData[usernum]->boundaryNodes[e];
    boundaryCells[0][e]->sidename = localData[usernum]->boundaryNames[e];
    boundaryCells[0][e]->ip = localData[usernum]->boundaryIP[e];
    boundaryCells[0][e]->wts = localData[usernum]->boundaryWts[e];
    boundaryCells[0][e]->normals = localData[usernum]->boundaryNormals[e];
    boundaryCells[0][e]->hsize = localData[usernum]->boundaryHsize[e];
    boundaryCells[0][e]->basis = localData[usernum]->boundaryBasis[e];
    boundaryCells[0][e]->basis_grad = localData[usernum]->boundaryBasisGrad[e];
    
    boundaryCells[0][e]->addAuxDiscretization(macro_basis_pointers,
                                              localData[usernum]->aux_side_basis[e],
                                              localData[usernum]->aux_side_basis_grad[e]);
    
    boundaryCells[0][e]->auxLIDs = localData[usernum]->boundaryMacroLIDs[e];
    boundaryCells[0][e]->auxMIDs = localData[usernum]->boundaryMIDs[e];
  }
  */
}
