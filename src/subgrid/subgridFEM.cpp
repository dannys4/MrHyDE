/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "subgridFEM.hpp"
#include "cell.hpp"

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

SubGridFEM::SubGridFEM(const Teuchos::RCP<MpiComm> & LocalComm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                       topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
                       ScalarT & macro_deltat_) :
settings(settings_), macro_cellTopo(macro_cellTopo_),
num_macro_time_steps(num_macro_time_steps_), macro_deltat(macro_deltat_) {
  
  LocalComm = LocalComm_;
  dimension = settings->sublist("Mesh").get<int>("dim",2);
  subgridverbose = settings->sublist("Solver").get<int>("verbosity",0);
  multiscale_method = settings->get<string>("multiscale method","mortar");
  numrefine = settings->sublist("Mesh").get<int>("refinements",0);
  shape = settings->sublist("Mesh").get<string>("shape","quad");
  macroshape = settings->sublist("Mesh").get<string>("macro-shape","quad");
  time_steps = settings->sublist("Solver").get<int>("number of steps",1);
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  write_subgrid_state = settings->sublist("Solver").get<bool>("write subgrid state",true);
  error_type = settings->sublist("Postprocess").get<string>("error type","L2"); // or "H1"
  store_aux_and_flux = settings->sublist("Postprocess").get<bool>("store aux and flux",false);
  string solver = settings->sublist("Solver").get<string>("solver","steady-state");
  if (solver == "steady-state") {
    final_time = 0.0;
  }
  
  soln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  adjsoln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  solndot = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  
  if (settings->isParameter("Functions input file")) {
    std::string filename = settings->get<std::string>("Functions input file");
    ifstream fn(filename.c_str());
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
                         Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation_) {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemTotalAddMacroTimer);
  
  Teuchos::RCP<SubGridLocalData> newdata = Teuchos::rcp( new SubGridLocalData(macronodes_,
                                                                              macrosideinfo_,
                                                                              macroLIDs_,
                                                                              macroorientation_) );
  localData.push_back(newdata);
  int bnum = localData.size()-1;
  return bnum;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::setUpSubgridModels() {
  
  Teuchos::TimeMonitor subgridsetuptimer(*sgfemTotalSetUpTimer);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid mesh
  /////////////////////////////////////////////////////////////////////////////////////
  
  string blockID = "eblock";
  
  vector<vector<ScalarT> > nodes;
  vector<vector<GO> > connectivity;
  Kokkos::View<int****,HostDevice> sideinfo;
  
  vector<string> eBlocks;
  
  SubGridTools sgt(LocalComm, macroshape, shape, localData[0]->macronodes,
                   localData[0]->macrosideinfo);
  
  {
    Teuchos::TimeMonitor localmeshtimer(*sgfemSubMeshTimer);
    
    sgt.createSubMesh(numrefine);
    
    nodes = sgt.getNodes(localData[0]->macronodes);
    int reps = localData[0]->macronodes.extent(0);
    connectivity = sgt.getSubConnectivity(reps);
    sideinfo = sgt.getNewSideinfo(localData[0]->macrosideinfo);
    
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
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  sub_physics = Teuchos::rcp( new physics(settings, LocalComm, sub_mesh->cellTopo,
                                          sub_mesh->sideTopo, sub_mesh->mesh) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the subgrid discretizations
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_disc = Teuchos::rcp( new discretization(settings, LocalComm, sub_mesh->mesh, sub_physics->unique_orders,
                                              sub_physics->unique_types) );
  
  
  int numSubElem = connectivity.size();
  
  settings->sublist("Solver").set<int>("workset size",numSubElem);
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
  
  Teuchos::RCP<panzer::DOFManager> DOF = sub_disc->buildDOF(sub_mesh->mesh,
                                                            sub_physics->varlist,
                                                            sub_physics->types,
                                                            sub_physics->orders,
                                                            sub_physics->useDG);
  
  sub_physics->setBCData(settings, sub_mesh->mesh, DOF, sub_disc->cards);
  //sub_disc->setIntegrationInfo(cells, boundaryCells, DOF, sub_physics);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the parameter manager, the assembler and the solver
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_params = Teuchos::rcp( new ParameterManager(LocalComm, settings, sub_mesh->mesh,
                                                  sub_physics, sub_disc));
  
  sub_assembler = Teuchos::rcp( new AssemblyManager(LocalComm, settings, sub_mesh->mesh,
                                                    sub_disc, sub_physics, DOF,
                                                    sub_params, numSubElem));
  
  cells = sub_assembler->cells;
  
  Teuchos::RCP<CellMetaData> cellData = sub_assembler->cellData[0];
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Boundary cells are not set up properly due to the lack of side sets in the subgrid mesh
  // These just need to be defined once though
  /////////////////////////////////////////////////////////////////////////////////////
  
  int numNodesPerElem = sub_mesh->cellTopo[0]->getNodeCount();
  vector<Teuchos::RCP<BoundaryCell> > newbcells;
  
  int numLocalBoundaries = localData[0]->macrosideinfo.extent(2);
  
  vector<int> unique_sides;
  vector<int> unique_local_sides;
  vector<string> unique_names;
  vector<vector<size_t> > boundary_groups;
  
  sgt.getUniqueSides(sideinfo, unique_sides, unique_local_sides, unique_names,
                     macrosidenames, boundary_groups);
  
  vector<stk::mesh::Entity> stk_meshElems;
  sub_mesh->mesh->getMyElements(blockID, stk_meshElems);
  
  // May need to be PHX::Device
  Kokkos::View<const LO**,Kokkos::LayoutRight,HostDevice> LIDs = DOF->getLIDs();
  
  for (size_t s=0; s<unique_sides.size(); s++) {
    
    string sidename = unique_names[s];
    vector<size_t> group = boundary_groups[s];
    
    int prog = 0;
    while (prog < group.size()) {
      int currElem = numSubElem;  // Avoid faults in last iteration
      if (prog+currElem > group.size()){
        currElem = group.size()-prog;
      }
      Kokkos::View<int*,AssemblyDevice> eIndex("element indices",currElem);
      Kokkos::View<int*,AssemblyDevice> sideIndex("local side indices",currElem);
      DRV currnodes("currnodes", currElem, numNodesPerElem, dimension);

      auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
      auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
      auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
      for (int e=0; e<currElem; e++) {
        host_eIndex(e) = group[e+prog];
        host_sideIndex(e) = unique_local_sides[s];
        for (int n=0; n<numNodesPerElem; n++) {
          for (int m=0; m<dimension; m++) {
            host_currnodes(e,n,m) = nodes[connectivity[eIndex(e)][n]][m];
          }
        }
      }
      int sideID = s;
     
      Kokkos::deep_copy(currnodes,host_currnodes);
      Kokkos::deep_copy(eIndex,host_eIndex);
      Kokkos::deep_copy(sideIndex,host_sideIndex); 
      
      // Build the Kokkos View of the cell GIDs ------
      
      LIDView hostLIDs("LIDs on host device",
                                             currElem,LIDs.extent(1));
      for (int i=0; i<currElem; i++) {
        size_t elemID = eIndex(i);
        for (int j=0; j<LIDs.extent(1); j++) {
          hostLIDs(i,j) = LIDs(elemID,j);
        }
      }
      
      //-----------------------------------------------
      // Set the side information (soon to be removed)-
      Kokkos::View<int****,HostDevice> sideinfo = sub_physics->getSideInfo(0,host_eIndex);
      
      //-----------------------------------------------
      // Set the cell orientation ---
      Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices",
                                                                      currElem, numNodesPerElem);
      for (int i=0; i<currElem; i++) {
        vector<stk::mesh::EntityId> stk_nodeids;
        size_t elemID = eIndex(i);
        sub_mesh->mesh->getNodeIdsForElement(stk_meshElems[elemID], stk_nodeids);
        for (int n=0; n<numNodesPerElem; n++) {
          currind(i,n) = stk_nodeids[n];
        }
      }
      
      Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orient_drv("kv to orients",currElem);
      Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, currind, *(sub_mesh->cellTopo[0]));
      
      newbcells.push_back(Teuchos::rcp(new BoundaryCell(cellData,currnodes,eIndex,sideIndex,
                                                        sideID,sidename, newbcells.size(),
                                                        hostLIDs, sideinfo, orient_drv)));
      
      prog += currElem;
    }
    
    
  }
  
  boundaryCells.push_back(newbcells);
  
  sub_assembler->boundaryCells = boundaryCells;
  
  size_t numMacroDOF = localData[0]->macroLIDs.extent(1);
  sub_solver = Teuchos::rcp( new SubGridFEM_Solver(LocalComm, settings, sub_mesh, sub_disc, sub_physics,
                                                   sub_assembler, sub_params, DOF, macro_deltat,
                                                   numMacroDOF) );
  
  sub_postproc = Teuchos::rcp( new PostprocessManager(LocalComm, settings, sub_mesh->mesh, sub_disc, sub_physics,
                                                      functionManagers, sub_assembler) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Create a subgrid function mananger
  /////////////////////////////////////////////////////////////////////////////////////
  
  {
    Teuchos::TimeMonitor localtimer(*sgfemLinearAlgebraSetupTimer);
    
    varlist = sub_physics->varlist[0];
    functionManagers[0]->setupLists(sub_physics->varlist[0], macro_paramnames,
                                macro_disc_paramnames);
    sub_assembler->wkset[0]->params_AD = paramvals_KVAD;
    
    functionManagers[0]->wkset = sub_assembler->wkset[0];
    
    functionManagers[0]->validateFunctions();
    functionManagers[0]->decomposeFunctions();
  }
  
  wkset = sub_assembler->wkset;
  
  wkset[0]->addAux(macro_varlist.size());
  for(size_t e=0; e<boundaryCells[0].size(); e++) {
    boundaryCells[0][e]->addAuxVars(macro_varlist);
    boundaryCells[0][e]->cellData->numAuxDOF = macro_numDOF;
    boundaryCells[0][e]->numAuxDOF = macro_numDOF;
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
  
  for (size_t mindex = 0; mindex<localData.size(); mindex++) {
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Define the local nodes
    /////////////////////////////////////////////////////////////////////////////////////
    
    localData[mindex]->nodes = sgt.getNewNodes(localData[mindex]->macronodes);
    
    localData[mindex]->setIP(cells[0][0]->cellData, cells[0][0]->orientation);
    
    vector<size_t> gids;
    for (size_t e=0; e<cells[0][0]->numElem; e++){
      size_t id = localData[mindex]->getMacroID(e);
      gids.push_back(id);
    }
    localData[mindex]->macroIDs = gids;
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Define the local sideinfo
    /////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<int****,HostDevice> newsideinfo = sgt.getNewSideinfo(localData[mindex]->macrosideinfo);
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubSideinfoTimer);
      
      int sprog = 0;
      // Redefine the sideinfo for the subcells
      Kokkos::View<int****,HostDevice> subsideinfo("subcell side info", cells[0][0]->numElem, newsideinfo.extent(1),
                                                   newsideinfo.extent(2), newsideinfo.extent(3));
      
      for (size_t c=0; c<cells[0][0]->numElem; c++) { // number of elem in cell
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
      localData[mindex]->sideinfo = subsideinfo;
      
      vector<int> unique_sides;
      vector<int> unique_local_sides;
      vector<string> unique_names;
      vector<vector<size_t> > boundary_groups;
      
      sgt.getUniqueSides(subsideinfo, unique_sides, unique_local_sides, unique_names,
                         macrosidenames, boundary_groups);
      
      
      vector<string> bnames;
      vector<DRV> boundaryNodes;
      vector<vector<size_t> > boundaryMIDs;
      // Number of cells in each group is less than workset size, so just add the groups
      // without breaking into subgroups
      for (size_t s=0; s<unique_sides.size(); s++) {
        vector<size_t> group = boundary_groups[s];
        DRV currnodes("currnodes", group.size(), numNodesPerElem, dimension);
        vector<size_t> mIDs;
        for (int e=0; e<group.size(); e++) {
          size_t eIndex = group[e];
          size_t mID = localData[mindex]->getMacroID(eIndex);
          
          mIDs.push_back(mID);
          for (int n=0; n<numNodesPerElem; n++) {
            for (int m=0; m<dimension; m++) {
              currnodes(e,n,m) = localData[mindex]->nodes(eIndex,n,m);//newnodes[connectivity[eIndex][n]][m];
            }
          }
        }
        boundaryNodes.push_back(currnodes);
        bnames.push_back(unique_names[s]);
        boundaryMIDs.push_back(mIDs);
        
      }
      localData[mindex]->boundaryNodes = boundaryNodes;
      localData[mindex]->boundaryNames = bnames;
      localData[mindex]->boundaryMIDs = boundaryMIDs;
      localData[mindex]->setBoundaryIndexLIDs(); // must be done after boundaryMIDs are set
      
      Kokkos::View<int**,UnifiedDevice> currbcs("boundary conditions",subsideinfo.extent(1),
                                                 localData[mindex]->macrosideinfo.extent(2));
      for (size_t i=0; i<subsideinfo.extent(1); i++) { // number of variables
        for (size_t j=0; j<localData[mindex]->macrosideinfo.extent(2); j++) { // number of sides per element
          currbcs(i,j) = 5;
        }
      }
      for (size_t c=0; c<subsideinfo.extent(0); c++) {
        for (size_t i=0; i<subsideinfo.extent(1); i++) { // number of variables
          for (size_t j=0; j<subsideinfo.extent(2); j++) { // number of sides per element
            if (subsideinfo(c,i,j,0) > 1) { // TMW: should != 5
              for (size_t p=0; p<unique_sides.size(); p++) {
                if (unique_sides[p] == subsideinfo(c,i,j,1)) {
                  currbcs(i,p) = subsideinfo(c,i,j,0);
                }
              }
            }
          }
        }
      }
      localData[mindex]->bcs = currbcs;
     
    }
    
    // This can only be done after the boundary nodes have been set up
    vector<Kokkos::View<LO*,AssemblyDevice> > localSideIDs;
    vector<Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> > borientation;
    for (size_t bcell=0; bcell<sub_assembler->boundaryCells[0].size(); bcell++) {
      localSideIDs.push_back(sub_assembler->boundaryCells[0][bcell]->localSideID);
      borientation.push_back(sub_assembler->boundaryCells[0][bcell]->orientation);
    }
    localData[mindex]->setBoundaryIP(cells[0][0]->cellData, localSideIDs, borientation);
    
    //////////////////////////////////////////////////////////////
    // Set the initial conditions
    //////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubICTimer);
      
      Teuchos::RCP<LA_MultiVector> init = Teuchos::rcp(new LA_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
      this->setInitial(init, mindex, false);
      soln->store(init,initial_time,mindex);
      
      Teuchos::RCP<LA_MultiVector> inita = Teuchos::rcp(new LA_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
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
          localData[mindex]->computeMacroBasisVolIP(macro_cellTopo, macro_basis_pointers, sub_disc);
        }
        else {
          localData[mindex]->computeMacroBasisBoundaryIP(macro_cellTopo, macro_basis_pointers, sub_disc);//, wkset[0]);
        }
      }
      else {
        localData[mindex]->aux_side_basis = localData[0]->aux_side_basis;
        localData[mindex]->aux_side_basis_grad = localData[0]->aux_side_basis_grad;
      }
    }
  }
  
  sub_physics->setWorkset(wkset);
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::finalize() {
  if (localData.size() > 0) {
    this->setUpSubgridModels();
    
    size_t defblock = 0;
    if (cells.size() > 0) {
      sub_physics->setAuxVars(defblock, macro_varlist);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::addMeshData() {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemMeshDataTimer);
  
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
    
    for (size_t b=0; b<localData.size(); b++) {
      int numElem = cells[0][0]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      localData[b]->cell_data = cell_data;
      localData[b]->cell_data_distance = vector<ScalarT>(numElem);
      localData[b]->cell_data_seed = vector<size_t>(numElem);
      localData[b]->cell_data_seedindex = vector<size_t>(numElem);
    }
    
    for (int p=0; p<number_mesh_data_files; p++) {
      
      Teuchos::RCP<data> mesh_data;
      
      string mesh_data_pts_file;
      string mesh_data_file;
      
      if (have_multiple_data_files) {
        stringstream ss;
        ss << p+1;
        mesh_data_pts_file = mesh_data_pts_tag + "." + ss.str() + ".dat";
        mesh_data_file = mesh_data_tag + "." + ss.str() + ".dat";
      }
      else {
        mesh_data_pts_file = mesh_data_pts_tag + ".dat";
        mesh_data_file = mesh_data_tag + ".dat";
      }
      
      mesh_data = Teuchos::rcp(new data("mesh data", dimension, mesh_data_pts_file,
                                        mesh_data_file, false));
      
      for (size_t b=0; b<localData.size(); b++) {
        for (size_t e=0; e<cells[0].size(); e++) {
          int numElem = cells[0][e]->numElem;
          DRV nodes = localData[b]->nodes;
          for (int c=0; c<numElem; c++) {
            Kokkos::View<ScalarT**,AssemblyDevice> center("center",1,3);
            int numnodes = nodes.extent(1);
            for (size_t i=0; i<numnodes; i++) {
              for (size_t j=0; j<dimension; j++) {
                center(0,j) += nodes(c,i,j)/(ScalarT)numnodes;
              }
            }
            ScalarT distance = 0.0;
            
            int cnode = mesh_data->findClosestNode(center(0,0), center(0,1), center(0,2), distance);
            
            bool iscloser = true;
            if (p>0){
              if (localData[b]->cell_data_distance[c] < distance) {
                iscloser = false;
              }
            }
            if (iscloser) {
              Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode);
              
              for (unsigned int i=0; i<cdata.extent(1); i++) {
                localData[b]->cell_data(c,i) = cdata(0,i);
              }
              cells[0][0]->cellData->have_extra_data = true;
              if (have_rotations)
                cells[0][0]->cellData->have_cell_rotation = true;
              if (have_rotation_phi)
                cells[0][0]->cellData->have_cell_phi = true;
              
              localData[b]->cell_data_seed[c] = cnode % 50;
              localData[b]->cell_data_distance[c] = distance;
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
      size_t prog = 0;
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
    
    for (size_t b=0; b<localData.size(); b++) {
      int numElem = cells[0][0]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      localData[b]->cell_data = cell_data;
      localData[b]->cell_data_distance = vector<ScalarT>(numElem);
      localData[b]->cell_data_seed = vector<size_t>(numElem);
      localData[b]->cell_data_seedindex = vector<size_t>(numElem);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set cell data
    ////////////////////////////////////////////////////////////////////////////////
    
    for (size_t b=0; b<localData.size(); b++) {
      for (size_t e=0; e<cells[0].size(); e++) {
        DRV nodes = localData[b]->nodes;
        
        int numElem = cells[0][e]->numElem;
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
            localData[b]->cell_data(c,i) = rotation_data(cnode,i);
          }
          
          localData[b]->cell_data_seed[c] = cnode;
          localData[b]->cell_data_seedindex[c] = seedIndex(cnode);
          localData[b]->cell_data_distance[c] = distance;
          
        }
      }
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
  
  // Update the cells for this macro-element (or set of macro-elements)
  this->updateLocalData(usernum);
  
  // Copy the locak data (look into using subviews for this)
  // Solver does not know about localData
  Kokkos::View<ScalarT***,AssemblyDevice> coarse_u("local u",cells[0][0]->numElem,
                                                   coarse_fwdsoln.extent(1),
                                                   coarse_fwdsoln.extent(2));
  Kokkos::View<ScalarT***,AssemblyDevice> coarse_phi("local phi",cells[0][0]->numElem,
                                                     coarse_adjsoln.extent(1),
                                                     coarse_adjsoln.extent(2));
  
  // TMW: update for device (subgrid or assembly?)
  // Need to move localData[]->macroIDs to a Kokkos::View on the appropriate device
  for (int e=0; e<coarse_u.extent(0); e++) {
    for (unsigned int i=0; i<coarse_u.extent(1); i++) {
      for (unsigned int j=0; j<coarse_u.extent(2); j++) {
        coarse_u(e,i,j) = coarse_fwdsoln(localData[usernum]->macroIDs[e],i,j);
      }
    }
  }
  for (int e=0; e<coarse_phi.extent(0); e++) {
    for (unsigned int i=0; i<coarse_phi.extent(1); i++) {
      for (unsigned int j=0; j<coarse_phi.extent(2); j++) {
        coarse_phi(e,i,j) = coarse_adjsoln(localData[usernum]->macroIDs[e],i,j);
      }
    }
  }
  
  // Extract the previous solution as the initial guess/condition for subgrid problems
  Teuchos::RCP<LA_MultiVector> prev_fwdsoln, prev_adjsoln;
  {
    Teuchos::TimeMonitor localtimer(*sgfemInitialTimer);
    
    ScalarT prev_time = 0.0; // TMW: needed?
    size_t numtimes = soln->times[usernum].size();
    if (isAdjoint) {
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_fwdsoln, usernum, time, prev_time);
        bool foundadj = adjsoln->extract(prev_adjsoln, usernum, time);
      }
      else {
        bool foundfwd = soln->extract(prev_fwdsoln, usernum, time);
        bool foundadj = adjsoln->extract(prev_adjsoln, usernum, time);
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
      }
      if (compute_sens) {
        double nexttime = 0.0;
        bool foundadj = adjsoln->extractNext(prev_adjsoln,usernum,time,nexttime);
      }
    }
  }
  
  // Containers for current forward/adjoint solutions
  Teuchos::RCP<LA_MultiVector> curr_fwdsoln = Teuchos::rcp(new LA_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
  Teuchos::RCP<LA_MultiVector> curr_adjsoln = Teuchos::rcp(new LA_MultiVector(sub_solver->milo_solver->LA_overlapped_map,1));
  
  // Solve the local subgrid problem and fill in the coarse macrowkset->res;
  sub_solver->solve(coarse_u, coarse_phi,
                    prev_fwdsoln, prev_adjsoln, curr_fwdsoln, curr_adjsoln, Psol[0],
                    localData[usernum], time, isTransient, isAdjoint, compute_jacobian,
                    compute_sens, num_active_params, compute_disc_sens, compute_aux_sens,
                    macrowkset, usernum, macroelemindex, subgradient, store_adjPrev);
  
  // Store the subgrid fwd or adj solution
  
  if (isAdjoint) {
    adjsoln->store(curr_adjsoln,time,usernum);
  }
  else if (!compute_sens) {
    soln->store(curr_fwdsoln,time,usernum);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux) {
  
  int num_dof_lambda = lambda.extent(1)*lambda.extent(2);
  
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

void SubGridFEM::setInitial(Teuchos::RCP<LA_MultiVector> & initial,
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

//Kokkos::View<ScalarT**,AssemblyDevice> SubGridFEM::computeError(const ScalarT & time, const int & usernum) {
Kokkos::View<ScalarT**,AssemblyDevice> SubGridFEM::computeError(vector<pair<size_t, string> > & sub_error_list,
                                                                const vector<ScalarT> & times) {
  
  //Kokkos::View<ScalarT***,AssemblyDevice> errors("error",solvetimes.size(), sub_physics->varlist[0].size(), error_types.size());
  Kokkos::View<ScalarT**,AssemblyDevice> errors;
  if (localData.size() > 0) {
    
    errors = Kokkos::View<ScalarT**,AssemblyDevice>("error", times.size(), sub_postproc->error_list[0].size());
    sub_error_list = sub_postproc->error_list[0];
  
    for (size_t t=0; t<times.size(); t++) {
      for (size_t b=0; b<localData.size(); b++) {// loop over coarse scale elements
        bool compute = false;
        if (subgrid_static) {
          compute = true;
        }
        else if (active[t][b]) {
          compute = true;
        }
        if (compute) {
          size_t usernum = b;
          Teuchos::RCP<LA_MultiVector> currsol;
          
          //size_t tindex = t;
          //bool found = soln->extract(currsol, usernum, solvetimes[t]);//, tindex);
          bool found = soln->extract(currsol, usernum, times[t]);//, tindex);
          //bool found = soln->extract(currsol, tindex, usernum);//, tindex);
          
          if (found) {
            this->updateLocalData(usernum);
            
            //Kokkos::View<ScalarT***,AssemblyDevice> localerror("error",solvetimes.size(),numVars[b],error_types.size());
            //bool fnd = solve->soln->extract(u,t);
            sub_solver->performGather(0,currsol,0,0);
            sub_postproc->computeError(times[t]);
            
            size_t numerrs = sub_postproc->errors.size();
            
            Kokkos::View<ScalarT*,AssemblyDevice> cerr = sub_postproc->errors[0][0];//sub_postproc->errors[0][numerrs-1];
            for (size_t etype=0; etype<cerr.extent(0); etype++) {
              errors(t,etype) += cerr(etype);
            }
            sub_postproc->errors.clear();
            
          }
        }
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
  
  Teuchos::RCP<LA_MultiVector> currsol;
  bool found = soln->extract(currsol,usernum,time,tindex);
  
  Kokkos::View<AD*,AssemblyDevice> objective;
  if (found) {
    this->updateLocalData(usernum);
    bool beensized = false;
    sub_solver->performGather(0, currsol, 0,0);
    //this->performGather(usernum, Psol[0], 4, 0);
    
    for (size_t e=0; e<cells[0].size(); e++) {
      Kokkos::View<AD**,AssemblyDevice> curr_obj = cells[0][e]->computeObjective(time, tindex, seedwhat);
      if (!beensized && curr_obj.extent(1)>0) {
        objective = Kokkos::View<AD*,AssemblyDevice>("objective", curr_obj.extent(1));
        beensized = true;
      }
      for (int c=0; c<cells[0][e]->numElem; c++) {
        for (size_t i=0; i<curr_obj.extent(1); i++) {
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

void SubGridFEM::writeSolution(const string & filename, const int & usernum) {
  
  bool isTD = false;
  if (soln->times[usernum].size() > 1) {
    isTD = true;
  }
  
  string blockID = "eblock";
  
  //////////////////////////////////////////////////////////////
  // Re-create the subgrid mesh
  //////////////////////////////////////////////////////////////
  
  SubGridTools sgt(LocalComm, macroshape, shape, localData[usernum]->macronodes,
                   localData[usernum]->macrosideinfo);
  sgt.createSubMesh(numrefine);
  vector<vector<ScalarT> > nodes = sgt.getSubNodes();
  vector<vector<GO> > connectivity = sgt.getSubConnectivity();
  Kokkos::View<int****,HostDevice> sideinfo = sgt.getSubSideinfo();
  
  size_t numNodesPerElem = connectivity[0].size();
  
  panzer_stk::SubGridMeshFactory submeshFactory(shape, nodes, connectivity, blockID);
  Teuchos::RCP<panzer_stk::STK_Interface> submesh = submeshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
  
  //////////////////////////////////////////////////////////////
  // Add in the necessary fields for plotting
  //////////////////////////////////////////////////////////////
  
  vector<string> vartypes = sub_physics->types[0];
  
  vector<string> subeBlocks;
  submesh->getElementBlockNames(subeBlocks);
  for (size_t j=0; j<sub_physics->varlist[0].size(); j++) {
    if (vartypes[j] == "HGRAD") {
      submesh->addSolutionField(sub_physics->varlist[0][j], subeBlocks[0]);
    }
    else if (vartypes[j] == "HVOL"){
      submesh->addCellField(sub_physics->varlist[0][j], subeBlocks[0]);
    }
    else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL"){
      submesh->addCellField(sub_physics->varlist[0][j]+"x", subeBlocks[0]);
      submesh->addCellField(sub_physics->varlist[0][j]+"y", subeBlocks[0]);
      submesh->addCellField(sub_physics->varlist[0][j]+"z", subeBlocks[0]);
    }
  }
  vector<string> subextrafieldnames = sub_physics->getExtraFieldNames(0);
  for (size_t j=0; j<subextrafieldnames.size(); j++) {
    submesh->addSolutionField(subextrafieldnames[j], subeBlocks[0]);
  }
  vector<string> subextracellfields = sub_physics->getExtraCellFieldNames(0);
  for (size_t j=0; j<subextracellfields.size(); j++) {
    submesh->addCellField(subextracellfields[j], subeBlocks[0]);
  }
  submesh->addCellField("mesh_data_seed", subeBlocks[0]);
  submesh->addCellField("mesh_data", subeBlocks[0]);
  
  if (discparamnames.size() > 0) {
    for (size_t n=0; n<discparamnames.size(); n++) {
      int paramnumbasis = cells[0][0]->numParamDOF.extent(0);
      if (paramnumbasis==1) {
        submesh->addCellField(discparamnames[n], subeBlocks[0]);
      }
      else {
        submesh->addSolutionField(discparamnames[n], subeBlocks[0]);
      }
    }
  }
  
  submeshFactory.completeMeshConstruction(*submesh,*(LocalComm->getRawMpiComm()));
  
  //////////////////////////////////////////////////////////////
  // Add fields to mesh
  //////////////////////////////////////////////////////////////
  
  if(isTD) {
    submesh->setupExodusFile(filename);
  }
  int numSteps = soln->times[usernum].size();
  
  for (int m=0; m<numSteps; m++) {
    
    vector<size_t> myElements;
    size_t eprog = 0;
    for (size_t e=0; e<cells[0].size(); e++) {
      for (size_t p=0; p<cells[0][e]->numElem; p++) {
        myElements.push_back(eprog);
        eprog++;
      }
    }
    
    vector_RCP u;
    bool fnd = soln->extract(u,usernum,soln->times[usernum][m],m);
    auto u_kv = u->getLocalView<HostDevice>();
    
    vector<vector<int> > suboffsets = sub_physics->offsets[0];
    // Collect the subgrid solution
    for (int n = 0; n<sub_physics->varlist[0].size(); n++) {
      if (vartypes[n] == "HGRAD") {
        //size_t numsb = cells[usernum][0]->numDOF(n);//index[0][n].size();
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",cells[0][0]->numElem, numNodesPerElem); // TMW temp. fix
        string var = sub_physics->varlist[0][n];
        size_t pprog = 0;
        
        for( size_t e=0; e<cells[0].size(); e++ ) {
          int numElem = cells[0][e]->numElem;
          LIDView LIDs = cells[0][e]->LIDs;
          for (int p=0; p<numElem; p++) {
            
            for( int i=0; i<numNodesPerElem; i++ ) {
              int pindex = LIDs(p,suboffsets[n][i]);
              soln_computed(pprog,i) = u_kv(pindex,0);
            }
            pprog += 1;
          }
        }
        
        submesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HVOL") {
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",cells[0][0]->numElem, 1);
        string var = sub_physics->varlist[0][n];
        size_t pprog = 0;
        
        for( size_t e=0; e<cells[0].size(); e++ ) {
          int numElem = cells[0][e]->numElem;
          LIDView LIDs = cells[0][e]->LIDs;
          for (int p=0; p<numElem; p++) {
            LO pindex = LIDs(p,suboffsets[n][0]);
            soln_computed(pprog,0) = u_kv(pindex,0);
            pprog += 1;
          }
        }
        submesh->setCellFieldData(var, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") {
        Kokkos::View<ScalarT**,HostDevice> soln_x("soln",cells[0][0]->numElem, 1);
        Kokkos::View<ScalarT**,HostDevice> soln_y("soln",cells[0][0]->numElem, 1);
        Kokkos::View<ScalarT**,HostDevice> soln_z("soln",cells[0][0]->numElem, 1);
        string var = sub_physics->varlist[0][n];
        size_t pprog = 0;
        
        this->updateLocalData(usernum);
        sub_solver->performGather(0,u,0,0);
        for( size_t e=0; e<cells[0].size(); e++ ) {
          cells[0][e]->updateWorksetBasis();
          wkset[0]->computeSolnSteadySeeded(cells[0][e]->u, 0);
          cells[0][e]->computeSolnVolIP();
          
          int numElem = cells[0][e]->numElem;
          LIDView LIDs = cells[0][e]->LIDs;
          
          for (int p=0; p<numElem; p++) {
            ScalarT avgxval = 0.0;
            ScalarT avgyval = 0.0;
            ScalarT avgzval = 0.0;
            ScalarT avgwt = 0.0;
            for (int j=0; j<suboffsets[n].size(); j++) {
              ScalarT xval = wkset[0]->local_soln(p,n,j,0).val();
              avgxval += xval*wkset[0]->wts(p,j);
              if (dimension > 1) {
                ScalarT yval = wkset[0]->local_soln(p,n,j,1).val();
                avgyval += yval*wkset[0]->wts(p,j);
              }
              if (dimension > 2) {
                ScalarT zval = wkset[0]->local_soln(p,n,j,2).val();
                avgzval += zval*wkset[0]->wts(p,j);
              }
              avgwt += wkset[0]->wts(p,j);
            }
            soln_x(pprog,0) = avgxval/avgwt;
            soln_y(pprog,0) = avgyval/avgwt;
            soln_z(pprog,0) = avgzval/avgwt;
            pprog += 1;
          }
        }
        submesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
        submesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
        submesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
      }
    }
    
    
    Kokkos::View<ScalarT**,HostDevice> cseeds("cell data seeds",cells[0][0]->numElem, 1);
    Kokkos::View<ScalarT**,HostDevice> cdata("cell data",cells[0][0]->numElem, 1);
    
    if (cells[0][0]->cellData->have_cell_phi || cells[0][0]->cellData->have_cell_rotation || cells[0][0]->cellData->have_extra_data) {
      int eprog = 0;
      vector<size_t> cell_data_seed = localData[usernum]->cell_data_seed;
      vector<size_t> cell_data_seedindex = localData[usernum]->cell_data_seedindex;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data = localData[usernum]->cell_data;
      // TMW: need to use a mirror view here
      for (int p=0; p<cells[0][0]->numElem; p++) {
        cseeds(eprog,0) = cell_data_seedindex[p];
        cdata(eprog,0) = cell_data(p,0);
        eprog++;
      }
    }
    submesh->setCellFieldData("mesh_data_seed", blockID, myElements, cseeds);
    submesh->setCellFieldData("mesh_data", blockID, myElements, cdata);
    if (isTD) {
      submesh->writeToExodus(soln->times[usernum][m]);
    }
    else {
      submesh->writeToExodus(filename);
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
    cells[usernum][e]->addSensors(sensor_points,sensor_loc_tol,sensor_data,
                                  have_sensor_data, sub_disc, basisTypes, basisTypes);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix>  SubGridFEM::getProjectionMatrix() {
  
  return sub_solver->getProjectionMatrix();
  
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix> SubGridFEM::getProjectionMatrix(DRV & ip, DRV & wts,
                                                           pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  return sub_solver->getProjectionMatrix(ip, wts, other_basisinfo);
  
}

////////////////////////////////////////////////////////////////////////////////
// Get an empty vector
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

vector_RCP SubGridFEM::getVector() {
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
    int numElem = cells[usernum][e]->numElem;
    DRV ip = cells[usernum][e]->ip;
    for (size_t c=0; c<numElem; c++) {
      for (size_t i=0; i<ip.extent(1); i++) {
        for (size_t j=0; j<ip.extent(2); j++) {
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
    int numElem = cells[usernum][e]->numElem;
    for (size_t c=0; c<numElem; c++) {
      for (size_t i=0; i<wts.extent(0); i++) {
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

pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numLIDs = cells[0][0]->LIDs.extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numLIDs);
  
  for (size_t e=0; e<cells[0].size(); e++) {
    int numElem = cells[0][e]->numElem;
    DRV nodes = cells[0][e]->nodes;
    for (int c=0; c<numElem;c++) {
      DRV refpts("refpts",1, numpts, dimpts);
      DRVint inRefCell("inRefCell",1,numpts);
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
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}

////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
// TMW: what is this function for???
////////////////////////////////////////////////////////////////////////////////

pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM::evaluateBasis(const DRV & pts) {
  // this function is deprecated
}


////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix>  SubGridFEM::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<LA_Map> & ip_map) {
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
  for (size_t b=0; b<localData.size(); b++) {
    for (size_t e=0; e<cells[0].size(); e++) {
      int numElem = cells[0][e]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = localData[b]->cell_data_seed[c];
        for (int i=0; i<9; i++) {
          cells[0][e]->cell_data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::updateLocalData(const int & usernum) {
  
  wkset[0]->var_bcs = localData[usernum]->bcs;
  
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
  
}
