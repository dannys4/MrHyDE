/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "porousHDIV_weakGalerkin.hpp"

porousHDIV_WG::porousHDIV_WG(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                             const size_t & numip_side_, const int & numElem_,
                             Teuchos::RCP<FunctionManager> & functionManager_,
                             const size_t & blocknum_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_), blocknum(blocknum_) {
  
  label = "porousHDIV-WeakGalerkin";
  functionManager = functionManager_;
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  include_face = true;
  
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("pint")) {
      myvars.push_back("pint");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("pbndry")) {
      myvars.push_back("pbndry");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("pbndry","HGRAD")); // TODO: turn into HFACE-DG
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV-DG"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("t")) {
      myvars.push_back("t");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("t","HDIV-DG"));
    }
  }
  else {
    myvars.push_back("pint");
    myvars.push_back("pbndry");
    myvars.push_back("u");
    myvars.push_back("t");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HFACE"); // TODO: turn into HFACE-DG
    mybasistypes.push_back("HDIV-DG");
    mybasistypes.push_back("HDIV-DG");
  }
  
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kxx",fs.get<string>("kxx","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kxy",fs.get<string>("kxy","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kyx",fs.get<string>("kyx","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kyy",fs.get<string>("kyy","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kxz",fs.get<string>("kxz","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kzx",fs.get<string>("kzx","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kyz",fs.get<string>("kyz","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kzy",fs.get<string>("kzy","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("kzz",fs.get<string>("kzz","1.0"),numElem,numip,"ip",blocknum);
}

// ========================================================================================
// ========================================================================================

void porousHDIV_WG::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int pint_basis = wkset->usebasis[pintnum];
  int u_basis = wkset->usebasis[unum];
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip",blocknum);
    kxx = functionManager->evaluate("kxx","ip",blocknum);
    kxy = functionManager->evaluate("kxy","ip",blocknum);
    kyx = functionManager->evaluate("kyx","ip",blocknum);
    kyy = functionManager->evaluate("kyy","ip",blocknum);
    kxy = functionManager->evaluate("kxz","ip",blocknum);
    kyz = functionManager->evaluate("kyz","ip",blocknum);
    kzx = functionManager->evaluate("kzx","ip",blocknum);
    kzy = functionManager->evaluate("kzy","ip",blocknum);
    kzz = functionManager->evaluate("kzz","ip",blocknum);
  }
  
  basis = wkset->basis[u_basis];
  basis_div = wkset->basis_div[u_basis];
  
  // (u,v) + (p_0,div(v))
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT vx = 0.0;
    ScalarT vy = 0.0;
    ScalarT vz = 0.0;
    ScalarT divv = 0.0;
    AD uy = 0.0, uz = 0.0;
    
    for (size_t k=0; k<sol.dimension(2); k++ ) {
      for (size_t i=0; i<basis.dimension(1); i++ ) {
        AD pint = sol(e,pintnum,k,0);
        AD ux = sol(e,unum,k,0);
        
        if (spaceDim > 1) {
          uy = sol(e,unum,k,1);
        }
        if (spaceDim > 2) {
          uz = sol(e,unum,k,2);
        }
        
        vx = basis(e,i,k,0);
        
        if (spaceDim > 1) {
          vy = basis(e,i,k,1);
        }
        if (spaceDim > 2) {
          vz = basis(e,i,k,2);
        }
        divv = basis_div(e,i,k);
        int resindex = offsets(unum,i);
        res(e,resindex) += (ux*vx+uy*vy+uz*vz) + pint*divv;
        
      }
    }
    
  });
  
  //  (Ku,s) + (t,s)
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT sx = 0.0;
    ScalarT sy = 0.0;
    ScalarT sz = 0.0;
    AD uy = 0.0, uz = 0.0;
    AD ty = 0.0, tz = 0.0;
    
    for (size_t k=0; k<sol.dimension(2); k++ ) {
      for (size_t i=0; i<basis.dimension(1); i++ ) {
        AD ux = sol(e,unum,k,0);
        AD tx = sol(e,tnum,k,0);
        
        if (spaceDim > 1) {
          uy = sol(e,unum,k,1);
          ty = sol(e,tnum,k,1);
        }
        if (spaceDim > 2) {
          uz = sol(e,unum,k,2);
          tz = sol(e,tnum,k,2);
        }
        
        sx = basis(e,i,k,0);
        
        if (spaceDim > 1) {
          sy = basis(e,i,k,1);
        }
        if (spaceDim > 2) {
          sz = basis(e,i,k,2);
        }
        int resindex = offsets(tnum,i);
        // should be k_ij u_i s_j
        res(e,resindex) += ux*sx + uy*sy + uz*sz;
        //                               kxx(e,k)*ux*sx
        //                             + kxy(e,k)*ux*sy
        //                             + kyx(e,k)*uy*sx
        //                             + kyy(e,k)*uy*sy
        //                             + kxz(e,k)*ux*sz
        //                             + kyz(e,k)*uy*sz
        //                             + kzx(e,k)*uz*sx
        //                             + kzy(e,k)*uz*sy
        //                             + kzz(e,k)*uz*sz;
        
        res(e,resindex) += tx*sx + ty*sy + tz*sz;
        
      }
    }
    
  });
  
  //  (div(t),q_0) - (f,q_0)
  basis = wkset->basis[pint_basis];
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT qint = 0.0;
    
    for (size_t k=0; k<sol.dimension(2); k++ ) {
      for (size_t i=0; i<basis.dimension(1); i++ ) {
        AD divt = sol_div(e,tnum,k);
        
        qint = basis(e,i,k);
        int resindex = offsets(pintnum,i);
        res(e,resindex) += divt*qint - source(e,k)*qint;
        
      }
    }
    
  });
  
}


// ========================================================================================
// ========================================================================================

void porousHDIV_WG::boundaryResidual() {
  
  //    sideinfo = wkset->sideinfo;
  //    Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
  //
  //    int cside = wkset->currentside;
  //    int sidetype;
  //    sidetype = bcs(pbndrynum,cside);
  //
  //    basis = wkset->basis[pbndrynum];
  //
  //    {
  //      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
  //      bsource = functionManager->evaluate("Dirichlet pbndry " + wkset->sidename,"side ip",blocknum);
  //    }
  //
  //    // Since normals get recomputed often, this needs to be reset
  //    normals = wkset->normals;
  //
  //    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  //
  //    ScalarT qbndry = 0.0;
  //    AD ty = 0.0, tz = 0.0;
  //    ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
  //    for (int e=0; e<basis.dimension(0); e++) {
  //      for (int k=0; k<basis.dimension(2); k++ ) {
  //        for (int i=0; i<basis.dimension(1); i++ ) {
  //          AD tx = sol(e,tnum,k,0);
  //          nx = normals(e,k,0);
  //          if (spaceDim>1) {
  //            AD ty = sol(e,tnum,k,1);
  //            ny = normals(e,k,1);
  //          }
  //          if (spaceDim>2) {
  //            AD tz = sol(e,tnum,k,2);
  //            nz = normals(e,k,2);
  //          }
  //          qbndry = basis(e,i,k,0);
  //          int resindex = offsets(pbndrynum,i);
  //          res(e,resindex) += -qbndry*(tx*nx+ty*ny+tz*nz);
  //        }
  //      }
  //    }
  //
  //    basis = wkset->basis[unum];
  //
  //    ScalarT vx = 0.0, vy = 0.0, vz = 0.0;
  //    for (int e=0; e<basis.dimension(0); e++) {
  //      for (int k=0; k<basis.dimension(2); k++ ) {
  //        for (int i=0; i<basis.dimension(1); i++ ) {
  //          vx = basis(e,i,k,0);
  //          nx = normals(e,k,0);
  //          if (spaceDim>1) {
  //            vy = basis(e,i,k,1);
  //            ny = normals(e,k,1);
  //          }
  //          if (spaceDim>2) {
  //            vz = basis(e,i,k,2);
  //            nz = normals(e,k,2);
  //          }
  //          AD pbndry = sol(e,pbndrynum,k,0);
  //
  //          int resindex = offsets(unum,i);
  //          res(e,resindex) += -pbndry*(vx*nx+vy*ny+vz*nz);
  //        }
  //      }
  //    }
}


// ========================================================================================
// The edge (2D) and face (3D) contributions to the residual
// ========================================================================================

void porousHDIV_WG::faceResidual() {
  
  int pbndry_basis = wkset->usebasis[pbndrynum];
  int u_basis = wkset->usebasis[unum];
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  ScalarT vx = 0.0, vy = 0.0, vz = 0.0;
  ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
  
  // include <pbndry, v \cdot n> in velocity equation
  basis = wkset->basis_face[u_basis];
  
  for (size_t e=0; e<basis.dimension(0); e++) {
    for (size_t k=0; k<basis.dimension(2); k++ ) {
      for (size_t i=0; i<basis.dimension(1); i++ ) {
        vx = basis(e,i,k,0);
        nx = normals(e,k,0);
        if (spaceDim>1) {
          vy = basis(e,i,k,1);
          ny = normals(e,k,1);
        }
        if (spaceDim>2) {
          vz = basis(e,i,k,2);
          nz = normals(e,k,2);
        }
        AD pbndry = sol_face(e,pbndrynum,k,0);
        int resindex = offsets(unum,i);
        res(e,resindex) -= pbndry*(vx*nx+vy*ny+vz*nz);
      }
    }
  }
  
  // include -<t \cdot n, qbndry> in interface equation
  AD tx = 0.0, ty = 0.0, tz = 0.0;
  basis = wkset->basis_face[pbndry_basis];
  
  for (size_t e=0; e<basis.dimension(0); e++) {
    for (size_t k=0; k<basis.dimension(2); k++ ) {
      for (size_t i=0; i<basis.dimension(1); i++ ) {
        tx = sol_face(e,unum,k,0);
        nx = normals(e,k,0);
        if (spaceDim>1) {
          ty = sol_face(e,unum,k,1);
          ny = normals(e,k,1);
        }
        if (spaceDim>2) {
          tz = sol_face(e,unum,k,2);
          nz = normals(e,k,2);
        }
        ScalarT qbndry = basis(e,i,k);
        int resindex = offsets(pbndrynum,i);
        res(e,resindex) -= (tx*nx+ty*ny+tz*nz)*qbndry;
      }
    }
  }
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousHDIV_WG::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV_WG::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "pint")
      pintnum = i;
    if (varlist[i] == "pbndry")
      pbndrynum = i;
    if (varlist[i] == "u")
      unum = i;
    if (varlist[i] == "t")
      tnum = i;
    if (varlist[i] == "dx")
      dxnum = i;
    if (varlist[i] == "dy")
      dynum = i;
    if (varlist[i] == "dz")
      dznum = i;
  }
}
