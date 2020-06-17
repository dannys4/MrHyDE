/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "shallowwater.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

shallowwater::shallowwater(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "shallowwater";
  spaceDim = 2; // Just 2D
  
  myvars.push_back("H");
  myvars.push_back("Hu");
  myvars.push_back("Hv");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  
  gravity = settings->sublist("Physics").get<ScalarT>("gravity",9.8);
  
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                                   Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  Teuchos::ParameterList fs = settings->sublist("Functions");
  functionManager->addFunction("bathymetry",fs.get<string>("bathymetry","1.0"),"ip");
  functionManager->addFunction("bathymetry_x",fs.get<string>("bathymetry_x","0.0"),"ip");
  functionManager->addFunction("bathymetry_y",fs.get<string>("bathymetry_y","0.0"),"ip");
  functionManager->addFunction("bottom friction",fs.get<string>("bottom friction","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","0.0"),"ip");
  functionManager->addFunction("Coriolis",fs.get<string>("Coriolis","0.0"),"ip");
  functionManager->addFunction("source Hu",fs.get<string>("source Hu","0.0"),"ip");
  functionManager->addFunction("source Hv",fs.get<string>("source Hv","0.0"),"ip");
  functionManager->addFunction("flux left",fs.get<string>("flux left","0.0"),"side ip");
  functionManager->addFunction("flux right",fs.get<string>("flux right","0.0"),"side ip");
  functionManager->addFunction("flux top",fs.get<string>("flux top","0.0"),"side ip");
  functionManager->addFunction("flux bottom",fs.get<string>("flux bottom","0.0"),"side ip");
  functionManager->addFunction("Neumann source Hu",fs.get<string>("Neumann source Hu","0.0"),"side ip");
  functionManager->addFunction("Neumann source Hv",fs.get<string>("Neumann source Hv","0.0"),"side ip");
  functionManager->addFunction("bathymetry side",fs.get<string>("bathymetry","1.0"),"side_ip");
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    bath = functionManager->evaluate("bathymetry","ip");
    bath_x = functionManager->evaluate("bathymetry_x","ip");
    bath_y = functionManager->evaluate("bathymetry_y","ip");
    visc = functionManager->evaluate("viscosity","ip");
    cor = functionManager->evaluate("Coriolis","ip");
    source_Hu = functionManager->evaluate("source Hu","ip");
    source_Hv = functionManager->evaluate("source Hv","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int H_basis_num = wkset->usebasis[H_num];
  Hbasis = wkset->basis[H_basis_num];
  Hbasis_grad = wkset->basis_grad[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  Hubasis = wkset->basis[Hu_basis_num];
  Hubasis_grad = wkset->basis_grad[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  Hvbasis = wkset->basis[Hv_basis_num];
  Hvbasis_grad = wkset->basis_grad[Hv_basis_num];
  
  wts = wkset->wts;
  //KokkosTools::print(bath);
  
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    for (int k=0; k<sol.extent(2); k++ ) {
      
      AD xi = sol(e,H_num,k,0);
      AD xi_dot = sol_dot(e,H_num,k,0);
      AD H = xi + bath(e,k);
      AD Hu = sol(e,Hu_num,k,0);
      AD Hu_dot = sol_dot(e,Hu_num,k,0);
      
      AD Hv = sol(e,Hv_num,k,0);
      AD Hv_dot = sol_dot(e,Hv_num,k,0);
      
      
      for (int i=0; i<Hbasis.extent(1); i++ ) {
        
        int resindex = offsets(H_num,i);
        v = Hbasis(e,i,k);
        
        dvdx = Hbasis_grad(e,i,k,0);
        dvdy = Hbasis_grad(e,i,k,1);
        
        res(e,resindex) += (xi_dot*v - Hu*dvdx - Hv*dvdy)*wts(e,k);
        
      }
      
      for (int i=0; i<Hubasis.extent(1); i++ ) {
        
        int resindex = offsets(Hu_num,i);
        v = Hubasis(e,i,k);
        
        dvdx = Hubasis_grad(e,i,k,0);
        dvdy = Hubasis_grad(e,i,k,1);
        
        res(e,resindex) += (Hu_dot*v - (Hu*Hu/H + 0.5*gravity*(H*H-bath(e,k)*bath(e,k)))*dvdx - Hv*Hu/H*dvdy + gravity*xi*bath_x(e,k)*v)*wts(e,k);
        
      }
      
      for (int i=0; i<Hvbasis.extent(1); i++ ) {
        
        int resindex = offsets(Hv_num,i);
        v = Hvbasis(e,i,k);
        
        dvdx = Hvbasis_grad(e,i,k,0);
        dvdy = Hvbasis_grad(e,i,k,1);
        
        res(e,resindex) += (Hv_dot*v - (Hu*Hu/H)*dvdx - (Hv*Hu/H + 0.5*gravity*(H*H - bath(e,k)*bath(e,k)))*dvdy + gravity*xi*bath_y(e,k)*v)*wts(e,k);
        
      }
      
    }
    
  });
  
}


// ========================================================================================
// ========================================================================================

void shallowwater::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  string sidename = wkset->sidename;
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    if (sidename == "left") {
      nsource = functionManager->evaluate("flux left","side ip");
    }
    else if (sidename == "right") {
      nsource = functionManager->evaluate("flux right","side ip");
    }
    else if (sidename == "top") {
      nsource = functionManager->evaluate("flux top","side ip");
    }
    else if (sidename == "bottom") {
      nsource = functionManager->evaluate("flux bottom","side ip");
    }
    
  }
  
  sideinfo = wkset->sideinfo;
  
  int H_basis_num = wkset->usebasis[H_num];
  Hbasis = wkset->basis_side[H_basis_num];
  Hbasis_grad = wkset->basis_grad_side[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  Hubasis = wkset->basis_side[Hu_basis_num];
  Hubasis_grad = wkset->basis_grad_side[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  Hvbasis = wkset->basis_side[Hv_basis_num];
  Hvbasis_grad = wkset->basis_grad_side[Hv_basis_num];
  
  wts = wkset->wts_side;
  //KokkosTools::print(nsource);
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  ScalarT bb = 1.0;
  int cside = wkset->currentside;
  for (int e=0; e<sideinfo.extent(0); e++) {
    if (sideinfo(e,H_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<Hbasis.extent(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (int i=0; i<Hbasis.extent(1); i++ ) {
          int resindex = offsets(H_num,i);
          ScalarT v = Hbasis(e,i,k);
          //res(e,resindex) += (Hu*normals(e,k,0)+Hv*normals(e,k,1))*v;
          res(e,resindex) += (nsource(e,k)*H*v)*wts(e,k);
        }
      }
    }
    if (sideinfo(e,Hu_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<Hubasis.extent(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (int i=0; i<Hubasis.extent(1); i++ ) {
          int resindex = offsets(Hu_num,i);
          ScalarT v = Hubasis(e,i,k);
          //res(e,resindex) += (((Hu*Hu/H + 0.5*gravity*(H*H-bb*bb)))*normals(e,k,0) + Hv*Hu/H*normals(e,k,1))*v;
          res(e,resindex) += ((nsource(e,k)*Hu + 0.0*gravity*(H*H-bb*bb)*normals(e,k,0))*v)*wts(e,k);
        }
      }
    }
    if (sideinfo(e,Hv_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<Hvbasis.extent(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (int i=0; i<Hvbasis.extent(1); i++ ) {
          int resindex = offsets(Hv_num,i);
          ScalarT v = Hvbasis(e,i,k);
          //res(e,resindex) += (((Hu*Hu/H))*normals(e,k,0) + (Hv*Hu/H + 0.5*gravity*(H*H - bb*bb))*normals(e,k,1))*v;
          res(e,resindex) += ((nsource(e,k)*Hv + 0.0*gravity*(H*H - bb*bb)*normals(e,k,1))*v)*wts(e,k);
        }
      }
    }
  }
  
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void shallowwater::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "H")
      H_num = i;
    if (varlist[i] == "Hu")
      Hu_num = i;
    if (varlist[i] == "Hv")
      Hv_num = i;
  }
}
