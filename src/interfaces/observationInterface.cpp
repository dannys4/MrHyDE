#include "observationInterface.hpp"

using namespace MrHyDE;

ObservationInterface::ObservationInterface(
        std::string filename,
        Teuchos::RCP<MpiComm> & Comm_,
        std::vector<std::string>& which_params) : Comm(Comm_) {

    settings = UserInterfaceFactory::UserInterface(filename);
    // TODO: check that the postprocess is an observation with adjoint
    std::cout << "num params: " << which_params.size() << std::endl;
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the mesh
    ////////////////////////////////////////////////////////////////////////////////
    
    mesh = Teuchos::rcp(new MeshInterface(settings, Comm_) );
  
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the physics
    ////////////////////////////////////////////////////////////////////////////////
    
    physics = Teuchos::rcp( new PhysicsInterface(settings, Comm_, 
                                                 mesh->getBlockNames(),
                                                 mesh->getSideNames(),
                                                 mesh->getDimension()) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Mesh only needs the variable names and types to finalize
    ////////////////////////////////////////////////////////////////////////////////
    
    mesh->finalize(physics->getVarList(), physics->getVarTypes(), physics->getDerivedList());
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the discretization(s)
    ////////////////////////////////////////////////////////////////////////////////
        
    disc = Teuchos::rcp( new DiscretizationInterface(settings, Comm_, mesh, physics) );
    params = Teuchos::rcp( new ParameterManager<SolverNode>(Comm_, settings, mesh, physics, disc));
    param_indices = params->getParameterIndices(which_params);
    std::cout << "param indices: length" << param_indices.size() << ", (";
    for (auto i : param_indices) std::cout << i << ", ";
    std::cout << ")" << std::endl;
}

void ObservationInterface::ResetParameters(const std::vector<double> &parameters) {

    ////////////////////////////////////////////////////////////////////////////////
    // Set the parameters
    ////////////////////////////////////////////////////////////////////////////////
    
    params->setParameters(parameters, param_indices);
}

Teuchos::RCP<AnalysisManager> ObservationInterface::SetupAnalysis() {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the solver object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<AssemblyManager<SolverNode> > assembler = Teuchos::rcp( new AssemblyManager<SolverNode>(Comm, settings, mesh,
                                                                                                         disc, physics, params));
    
    assembler->setMeshData();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the subgrid discretizations/models if using multiscale method
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MultiscaleManager> multiscale_manager = Teuchos::rcp( new MultiscaleManager(Comm, mesh, settings,
                                                                                             assembler->groups,
                                                                                             #ifndef MrHyDE_NO_AD
                                                                                             assembler->function_managers_AD) );
                                                                                             #else
                                                                                             assembler->function_managers) );
                                                                                             #endif
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the solver and finalize some objects
    ////////////////////////////////////////////////////////////////////////////////
      
    Teuchos::RCP<SolverManager<SolverNode> > solve = Teuchos::rcp( new SolverManager<SolverNode>(Comm, settings, mesh,
                                                                                                 disc, physics, assembler, params) );
    

    solve->multiscale_manager = multiscale_manager;
    assembler->multiscale_manager = multiscale_manager;


    ///////////////////////////////////////////////////////////////////////////////
    // Create the postprocessing object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<PostprocessManager<SolverNode> >
    postproc = Teuchos::rcp( new PostprocessManager<SolverNode>(Comm, settings, mesh,
                                                                disc, physics, //assembler->function_managers_AD, 
                                                                multiscale_manager,
                                                                assembler, params) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Allocate most of the per-element memory usage
    ////////////////////////////////////////////////////////////////////////////////
    
    mesh->allocateMeshDataStructures();
    assembler->allocateGroupStorage();

    ////////////////////////////////////////////////////////////////////////////////
    // Purge Panzer memory before solving
    ////////////////////////////////////////////////////////////////////////////////
    int debug_level = settings->get<int>("debug level",0);
    if (settings->get<bool>("enable memory purge",false)) {
      if (debug_level > 0 && Comm->getRank() == 0) {
        std::cout << "******** Starting driver memory purge ..." << std::endl;
      }
      if (!settings->sublist("Postprocess").get("write solution",false) && 
          !settings->sublist("Postprocess").get("create optimization movie",false)) {
        mesh->purgeMesh();
        disc->mesh = Teuchos::null;
        params->mesh = Teuchos::null;
      }
      disc->purgeOrientations();
      disc->purgeLIDs();
      mesh->purgeMemory();
      assembler->purgeMemory();
      params->purgeMemory();
      physics->purgeMemory();
      if (debug_level > 0 && Comm->getRank() == 0) {
        std::cout << "******** Finished driver memory purge ..." << std::endl;
      } 
    }
        
    solve->completeSetup();
    
    if (settings->get<bool>("enable memory purge",false)) {
      disc->purgeMemory();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Finalize the function and multiscale managers
    ////////////////////////////////////////////////////////////////////////////////
    
    assembler->finalizeFunctions();

    solve->finalizeMultiscale();

    ////////////////////////////////////////////////////////////////////////////////
    // Perform the requested analysis (fwd solve, adj solve, dakota run, etc.)
    ////////////////////////////////////////////////////////////////////////////////
    
    solve->postproc = postproc;
    postproc->linalg = solve->linalg;
    
    Teuchos::RCP<AnalysisManager> analysis = Teuchos::rcp( new AnalysisManager(Comm, settings,
                                                                               solve, postproc, params) );
    
    // Make sure all processes are caught up at this point
    Kokkos::fence();
    Comm->barrier();
    return analysis;
}

double ObservationInterface::observe(const std::vector<double> &parameters) {
    this->ResetParameters(parameters);
    Teuchos::RCP<AnalysisManager> analysis = this->SetupAnalysis();
    DFAD objfun = analysis->forwardSolve();
    return objfun.val();
}

double ObservationInterface::observeDerivative(const std::vector<double> &parameters, std::vector<double> &gradient_out) {
    this->ResetParameters(parameters);
    Teuchos::RCP<AnalysisManager> analysis = this->SetupAnalysis();

    DFAD objfun = analysis->forwardSolve();
    
    MrHyDE_OptVector sens = analysis->adjointSolve();
    auto derivatives = sens.getParameter();
    for(int i = 0; i < param_indices.size(); i++) {
        int idx = param_indices[i];
        gradient_out[i] = (*derivatives)[idx];
    }
    return objfun.val();
}