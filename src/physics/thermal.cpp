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

#include "thermal.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

thermal::thermal(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  // Standard data
  isaux = isaux_;
  label = "thermal";
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("e")) {
      myvars.push_back("e");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("e","HGRAD"));
    }
  }
  else {
    myvars.push_back("e");
    mybasistypes.push_back("HGRAD");
  }
  // Extra data
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  have_nsvel = false;
  // Solely for testing purposes
  test_IQs = settings->sublist("Physics").get<bool>("test integrated quantities",false);

}

// ========================================================================================
// ========================================================================================

void thermal::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  functionManager->addFunction("thermal source",fs.get<string>("thermal source","0.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"ip");
  functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"side ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip");
  
}

// ========================================================================================
// ========================================================================================

void thermal::volumeResidual() {
  
 
  int spaceDim = wkset->dimension;
  auto basis = wkset->basis[e_basis_num];
  auto basis_grad = wkset->basis_grad[e_basis_num];
  
  /*
  Vista source, diff, cp, rho;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("thermal source","ip",true);
    diff = functionManager->evaluate("thermal diffusion","ip",true);
    cp = functionManager->evaluate("specific heat","ip",true);
    rho = functionManager->evaluate("density","ip",true);
  
  }
  */
  
  
  View_AD2 source, diff, cp, rho;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("thermal source","ip");
    diff = functionManager->evaluate("thermal diffusion","ip");
    cp = functionManager->evaluate("specific heat","ip");
    rho = functionManager->evaluate("density","ip");
  }
  
  
  // Contributes:
  // (f(u),v) + (DF(u),nabla v)
  // f(u) = rho*cp*de/dt - source
  // DF(u) = diff*grad(e)
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
 
  auto wts = wkset->wts;
  auto res = wkset->res;
    
  auto T = e_vol;
  auto dTdt = dedt_vol;
  
  auto off = subview( wkset->offsets, e_num, ALL());
  bool have_nsvel_ = have_nsvel;
  
  auto dTdx = dedx_vol;
  auto dTdy = dedy_vol;
  auto dTdz = dedz_vol;
  auto Ux = ux_vol;
  auto Uy = uy_vol;
  auto Uz = uz_vol;
  
  size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
  
  parallel_for("Thermal volume resid 3D part 1",
               TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
      for (size_type pt=0; pt<basis.extent(2); ++pt ) {
        res(elem,off(dof)) += (rho(elem,pt)*cp(elem,pt)*dTdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
        res(elem,off(dof)) += diff(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0);
        if (spaceDim > 1) {
          res(elem,off(dof)) += diff(elem,pt)*dTdy(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
        }
        if (spaceDim > 2) {
          res(elem,off(dof)) += diff(elem,pt)*dTdz(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,2);
        }
        if (have_nsvel_) {
          if (spaceDim == 1) {
            res(elem,off(dof)) += Ux(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
          }
          else if (spaceDim == 2) {
            res(elem,off(dof)) += (Ux(elem,pt)*dTdx(elem,pt) + Uy(elem,pt)*dTdy(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
          else {
            res(elem,off(dof)) += (Ux(elem,pt)*dTdx(elem,pt) + Uy(elem,pt)*dTdy(elem,pt) + Uz(elem,pt)*dTdz(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      }
    }
  });
  
}


// ========================================================================================
// ========================================================================================

void thermal::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(e_num,cside);

  auto basis = wkset->basis_side[e_basis_num];
  auto basis_grad = wkset->basis_grad_side[e_basis_num];
  
  View_AD2 nsource, diff_side, robin_alpha;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "weak Dirichlet" ) {
      nsource = functionManager->evaluate("Dirichlet e " + wkset->sidename,"side ip");
    }
    else if (bctype == "Neumann") {
      nsource = functionManager->evaluate("Neumann e " + wkset->sidename,"side ip");
    }
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
    robin_alpha = functionManager->evaluate("robin alpha","side ip");
    
  }
  
  ScalarT sf = formparam;
  if (wkset->isAdjoint) {
    sf = 1.0;
    adjrhs = wkset->adjrhs;
  }
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;
  auto off = subview( wkset->offsets, e_num, ALL());
  int dim = wkset->dimension;
  
  // Contributes
  // <g(u),v> + <p(u),grad(v)\cdot n>
  
  if (bcs(e_num,cside) == "Neumann") { // Neumann BCs
    parallel_for("Thermal bndry resid part 1",
                 TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          res(elem,off(dof)) += -nsource(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
        }
      }
    });
  }
  else if (bcs(e_num,cside) == "weak Dirichlet" || bcs(e_num,cside) == "interface") {
    auto T = e_side;
    auto dTdx = dedx_side;
    auto dTdy = dedy_side;
    auto dTdz = dedz_side;
    auto nx = wkset->getDataSc("nx side");
    auto ny = wkset->getDataSc("ny side");
    auto nz = wkset->getDataSc("nz side");
    View_AD2 bdata;
    if (bcs(e_num,cside) == "weak Dirichlet") {
      bdata = nsource;
    }
    else if (bcs(e_num,cside) == "interface") {
      bdata = wkset->getData("aux e side");
    }
    parallel_for("Thermal bndry resid wD",
                 TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      /*
      int myscratch = elem % scratch.extent(0);
      for (size_type pt=team.team_rank(); pt<wts.extent(1); pt+=team.team_size() ) {
        scratch(myscratch,pt,0) = 10.0/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt));
        scratch(myscratch,pt,0) += -diff_side(elem,pt)*dTdx(elem,pt)*nx(elem,pt);
        if (dim>1) {
          scratch(myscratch,pt,0) += -diff_side(elem,pt)*dTdy(elem,pt)*ny(elem,pt);
          if (dim>2) {
            scratch(myscratch,pt,0) += -diff_side(elem,pt)*dTdz(elem,pt)*nz(elem,pt);
          }
        }
        scratch(myscratch,pt,1) = -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt));
        scratch(myscratch,pt,0) *= wts(elem,pt);
        scratch(myscratch,pt,1) *= wts(elem,pt);
      }
       */
      if (dim == 1) {
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += 10.0/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
            res(elem,off(dof)) += -diff_side(elem,pt)*dTdx(elem,pt)*nx(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            res(elem,off(dof)) += -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt))*wts(elem,pt)*basis_grad(elem,dof,pt,0)*nx(elem,pt);
          }
        }
      }
      else if (dim == 2) {
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += 10.0/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
            res(elem,off(dof)) += -diff_side(elem,pt)*(dTdx(elem,pt)*nx(elem,pt)+dTdy(elem,pt)*ny(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
            res(elem,off(dof)) += -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt))*wts(elem,pt)*(basis_grad(elem,dof,pt,0)*nx(elem,pt) + basis_grad(elem,dof,pt,1)*ny(elem,pt));
          }
        }
      }
      else {
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += 10.0/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
            res(elem,off(dof)) += -diff_side(elem,pt)*(dTdx(elem,pt)*nx(elem,pt)+dTdy(elem,pt)*ny(elem,pt)+dTdz(elem,pt)*nz(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
            res(elem,off(dof)) += -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt))*wts(elem,pt)*(basis_grad(elem,dof,pt,0)*nx(elem,pt) + basis_grad(elem,dof,pt,1)*ny(elem,pt) + + basis_grad(elem,dof,pt,2)*nz(elem,pt));
          }
        }
      }
      /*
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          ScalarT gradv_dot_n = basis_grad(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            gradv_dot_n += basis_grad(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            gradv_dot_n += basis_grad(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) += scratch(myscratch,pt,0)*basis(elem,dof,pt,0) + scratch(myscratch,pt,1)*gradv_dot_n;
        }
      }*/
    });
    //if (wkset->isAdjoint) {
    //  adjrhs(e,resindex) += sf*diff_side(e,k)*gradv_dot_n*lambda - weakDiriScale*lambda*basis(e,i,k);
    //}
  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void thermal::computeFlux() {
  
  int spaceDim = wkset->dimension;
  // TMW: sf is still an issue for GPUs
  ScalarT sf = 1.0;
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  View_AD2 diff_side;
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
  }
  
  View_Sc2 nx, ny, nz;
  View_AD2 T, dTdx, dTdy, dTdz;
  wkset->get("nx side",nx);
  T = e_side;
  dTdx = dedx_side;
  if (spaceDim > 1) {
    wkset->get("ny side",ny);
    dTdy = dedy_side;
  }
  if (spaceDim > 2) {
    wkset->get("nz side",nz);
    dTdz = dedz_side;
  }
  
  auto h = wkset->h;
  int dim = wkset->dimension;
  
  {
    //Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto fluxT = subview(wkset->flux, ALL(), e_num, ALL());
    auto lambda = wkset->getData("aux e side");
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      parallel_for("Thermal bndry resid wD",
                   TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<nx.extent(1); pt+=team.team_size() ) {
          fluxT(elem,pt) = 10.0/h(elem)*diff_side(elem,pt)*(lambda(elem,pt)-T(elem,pt));
          fluxT(elem,pt) += sf*diff_side(elem,pt)*dTdx(elem,pt)*nx(elem,pt);
          if (dim > 1) {
            fluxT(elem,pt) += sf*diff_side(elem,pt)*dTdy(elem,pt)*ny(elem,pt);
          }
          if (dim > 2) {
            fluxT(elem,pt) += sf*diff_side(elem,pt)*dTdz(elem,pt)*nz(elem,pt);
          }
        }
      });
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

void thermal::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
  
  ux_num = -1;
  uy_num = -1;
  uz_num = -1;
  
  vector<string> varlist = wkset->varlist;
  //if (!isaux) {
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "e")
        e_num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
    }
  if (wkset->isInitialized) { // safeguard against proc having no elem on block
    e_basis_num = wkset->usebasis[e_num];
  }
    if (ux_num >=0)
      have_nsvel = true;
  //}
  
  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "e")
      auxe_num = i;
  }
  
  // Set these views so we don't need to search repeatedly
  
  if (mybasistypes[0] == "HGRAD") {
    wkset->get("e",e_vol);
    wkset->get("e side",e_side);
    wkset->get("e_t",dedt_vol);
    
    wkset->get("grad(e)[x]",dedx_vol);
    wkset->get("grad(e)[y]",dedy_vol);
    wkset->get("grad(e)[z]",dedz_vol);
    
    wkset->get("grad(e)[x] side",dedx_side);
    wkset->get("grad(e)[y] side",dedy_side);
    wkset->get("grad(e)[z] side",dedz_side);
    
    if (have_nsvel) {
      wkset->get("ux",ux_vol);
      wkset->get("uy",uy_vol);
      wkset->get("uz",uz_vol);
    }
  }

  // testing purposes only
  if (test_IQs) IQ_start = wkset->addIntegratedQuantities(3);

}

// ========================================================================================
// return the integrands for the integrated quantities (testing only for now)
// ========================================================================================

std::vector< std::vector<string> > thermal::setupIntegratedQuantities(const int & spaceDim) {

  std::vector< std::vector<string> > integrandsNamesAndTypes;

  // if not requested, be sure to return an empty vector
  if ( !(test_IQs) ) return integrandsNamesAndTypes;

  std::vector<string> IQ = {"e","thermal vol total e","volume"};
  integrandsNamesAndTypes.push_back(IQ);

  IQ = {"e","thermal bnd total e","boundary"};
  integrandsNamesAndTypes.push_back(IQ);

  // TODO -- BWR assumes the diffusion coefficient is 1.
  // I was getting all zeroes if I used "diff"
  string integrand = "(nx*grad(e)[x])";
  if (spaceDim == 2) integrand = "(nx*grad(e)[x] + ny*grad(e)[y])";
  if (spaceDim == 3) integrand = "(nx*grad(e)[x] + ny*grad(e)[y] + nz*grad(e)[z])";

  IQ = {integrand,"thermal bnd heat flux","boundary"};
  integrandsNamesAndTypes.push_back(IQ);

  return integrandsNamesAndTypes;

}
