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

#include "porousMixed.hpp"
using namespace MrHyDE;

porousMixed::porousMixed(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  label = "porousMixed";
  int spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
  
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("p")) {
      myvars.push_back("p");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV"));
    }
  }
  else {
    myvars.push_back("p");
    myvars.push_back("u");
    
    if (spaceDim == 1) { // to avoid the error in 1D HDIV
      mybasistypes.push_back("HVOL");
      mybasistypes.push_back("HGRAD");
    }
    else {
      mybasistypes.push_back("HVOL");
      mybasistypes.push_back("HDIV");
    }
  }
  
  usePermData = settings->sublist("Physics").get<bool>("use permeability data",false);
  useWells = settings->sublist("Physics").get<bool>("use well source",false);
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
  useKL = settings->sublist("Physics").get<bool>("use KL expansion",false);
  if (useKL) {
    if (settings->sublist("Physics").isSublist("KL parameters")) {
      auto KLlist = settings->sublist("Physics").sublist("KL parameters");
      permKLx = klexpansion(KLlist.sublist("x-direction").get<int>("N"),
                            KLlist.sublist("x-direction").get<double>("L"),
                            KLlist.sublist("x-direction").get<double>("sigma"),
                            KLlist.sublist("x-direction").get<double>("eta"));
      if (spaceDim > 1) {
        permKLx = klexpansion(KLlist.sublist("y-direction").get<int>("N"),
                              KLlist.sublist("y-direction").get<double>("L"),
                              KLlist.sublist("y-direction").get<double>("sigma"),
                              KLlist.sublist("y-direction").get<double>("eta"));
      }
      if (spaceDim > 2) {
        permKLx = klexpansion(KLlist.sublist("z-direction").get<int>("N"),
                              KLlist.sublist("z-direction").get<double>("L"),
                              KLlist.sublist("z-direction").get<double>("sigma"),
                              KLlist.sublist("z-direction").get<double>("eta"));
      }

    }
    else {
      // throw an error
    }
  }
  
}

// ========================================================================================
// ========================================================================================

void porousMixed::defineFunctions(Teuchos::ParameterList & fs,
                                 Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("Kinv_xx",fs.get<string>("Kinv_xx","1.0"),"ip");
  functionManager->addFunction("Kinv_yy",fs.get<string>("Kinv_yy","1.0"),"ip");
  functionManager->addFunction("Kinv_zz",fs.get<string>("Kinv_zz","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void porousMixed::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int p_basis = wkset->usebasis[pnum];
  int u_basis = wkset->usebasis[unum];
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  Vista source, bsource, Kinv_xx, Kinv_yy, Kinv_zz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    
    if (usePermData) {
      View_AD2 view_Kinv_xx("K inverse xx",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_yy("K inverse yy",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_zz("K inverse zz",wts.extent(0),wts.extent(1));
      this->updatePerm(view_Kinv_xx, view_Kinv_yy, view_Kinv_zz);
      Kinv_xx = Vista(view_Kinv_xx);
      Kinv_yy = Vista(view_Kinv_yy);
      Kinv_zz = Vista(view_Kinv_zz);
    }
    else {
      Kinv_xx = functionManager->evaluate("Kinv_xx","ip");
      Kinv_yy = functionManager->evaluate("Kinv_yy","ip");
      Kinv_zz = functionManager->evaluate("Kinv_zz","ip");
    }
    
    if (useKL) {
      View_AD2 KL_Kxx("KL K xx",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kyy("KL K yy",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kzz("KL K zz",wts.extent(0),wts.extent(1));
      this->updateKLPerm(KL_Kxx, KL_Kyy, KL_Kzz);
      
      View_AD2 new_Kxx("new K xx",wts.extent(0),wts.extent(1));
      View_AD2 new_Kyy("new K yy",wts.extent(0),wts.extent(1));
      View_AD2 new_Kzz("new K zz",wts.extent(0),wts.extent(1));
      
      parallel_for("porous HDIV update well source",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<KL_Kxx.extent(1); ++pt) {
          new_Kxx(elem,pt) = Kinv_xx(elem,pt)/exp(KL_Kxx(elem,pt));
        }
        
        if (spaceDim > 1) {
          for (size_type pt=0; pt<KL_Kyy.extent(1); ++pt) {
            new_Kyy(elem,pt) *= Kinv_yy(elem,pt)/exp(KL_Kyy(elem,pt));
          }
        }
        if (spaceDim > 2) {
          for (size_type pt=0; pt<KL_Kzz.extent(1); ++pt) {
            new_Kzz(elem,pt) *= Kinv_zz(elem,pt)/exp(KL_Kzz(elem,pt));
          }
        }
        
      });
      
      Kinv_xx = Vista(new_Kxx);
      Kinv_yy = Vista(new_Kyy);
      Kinv_zz = Vista(new_Kzz);
      
    }
    
    
    if (useWells) {
      auto h = wkset->h;
      View_AD2 source_kv("KV for source",wts.extent(0),wts.extent(1));
      parallel_for("porous HDIV update well source",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        ScalarT C = std::log(0.25*std::exp(-0.5772)*h(elem)/2.0);
        for (size_type pt=0; pt<source_kv.extent(1); pt++) {
#ifndef MrHyDE_NO_AD
          ScalarT Kval = 1.0/Kinv_xx(elem,pt).val();
#else
          ScalarT Kval = 1.0/Kinv_xx(elem,pt);
#endif
          source_kv(elem,pt) *= 2.0*PI/C*Kval;
        }
      });
      source = Vista(source_kv);
    }
  }
  
  Teuchos::TimeMonitor funceval(*volumeResidualFill);
  
  {
    // (K^-1 u,v) - (p,div v) - src*v (src not added yet)
    
    auto basis = wkset->basis[u_basis];
    auto psol = wkset->getData("p");
    auto off = subview(wkset->offsets, unum, ALL());
    
    if (spaceDim == 1) { // easier to place conditional here than on device
      auto ux = wkset->getData("u");
      auto basis_div = wkset->basis_grad[u_basis];
        
      parallel_for("porous HDIV volume resid u 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT divv = basis_div(elem,dof,pt,0);
            res(elem,off(dof)) += Kiux*vx - p*divv;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto ux = wkset->getData("u[x]");
      auto uy = wkset->getData("u[y]");
      auto basis_div = wkset->basis_div[u_basis];
      
      parallel_for("porous HDIV volume resid u 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD Kiuy = Kinv_yy(elem,pt)*uy(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += Kiux*vx + Kiuy*vy - p*divv;
          }
        }
      });
    }
    else {
      auto ux = wkset->getData("u[x]");
      auto uy = wkset->getData("u[y]");
      auto uz = wkset->getData("u[z]");
      auto basis_div = wkset->basis_div[u_basis];
      
      parallel_for("porous HDIV volume resid u 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD Kiuy = Kinv_yy(elem,pt)*uy(elem,pt)*wts(elem,pt);
          AD Kiuz = Kinv_zz(elem,pt)*uz(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT vz = basis(elem,dof,pt,2);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += Kiux*vx + Kiuy*vy + Kiuz*vz - p*divv;
          }
        }
      });
    }
  }
  
  {
    // -(div u,q) + (src,q) (src not added yet)
    
    auto basis = wkset->basis[p_basis];
    auto off = subview(wkset->offsets,pnum, ALL());
    View_AD2 udiv;
    if (spaceDim == 1) {
      udiv = wkset->getData("grad(u)[x]");
    }
    else {
      udiv = wkset->getData("div(u)");
    }
    
    parallel_for("porous HDIV volume resid div(u)",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD F = source(elem,pt) - udiv(elem,pt);
        F *= wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt,0);
          res(elem,off(dof)) += F*v;
        }
      }
    });
    
  }
}


// ========================================================================================
// ========================================================================================

void porousMixed::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(pnum,cside);
  
  auto basis = wkset->basis_side[unum];
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  View_Sc2 nx, ny, nz;
  View_AD2 ux, uy, uz;
  nx = wkset->getDataSc("nx side");
  
  if (spaceDim == 1) {
    ux = wkset->getData("u side");
  }
  else {
    ux = wkset->getData("u[x] side");
  }
  if (spaceDim > 1) {
    ny = wkset->getDataSc("ny side");
    uy = wkset->getData("u[y] side");
  }
  if (spaceDim > 2) {
    nz = wkset->getDataSc("nz side");
    uz = wkset->getData("u[z] side");
  }
  
  Vista bsource;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "Dirichlet" ) {
      bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    
  }
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto off = subview(wkset->offsets, unum, ALL());
  
  if (bcs(pnum,cside) == "Dirichlet") {
    parallel_for("porous HDIV bndry resid Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD src = bsource(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = basis(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            vdotn += basis(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            vdotn += basis(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) += src*vdotn;
        }
      }
    });
  }
  else if (bcs(pnum,cside) == "interface") {
    auto lambda = wkset->getData("aux "+auxvar+" side");
    parallel_for("porous HDIV boundary resid MS Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD lam = lambda(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = basis(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            vdotn += basis(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            vdotn += basis(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) += lam*vdotn;
        }
      }
    });
  }
   
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousMixed::computeFlux() {
  
  int spaceDim = wkset->dimension;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto uflux = subview(wkset->flux, ALL(), auxpnum, ALL());
    View_Sc2 nx, ny, nz;
    View_AD2 ux, uy, uz;
    if (spaceDim == 1) {
      nx = wkset->getDataSc("nx side");
      ux = wkset->getData("u side");
      parallel_for("porous HDIV flux ",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nx.extent(1); pt++) {
          AD udotn = ux(elem,pt)*nx(elem,pt);
          uflux(elem,pt) = udotn;
        }
      });
    }
    else if (spaceDim == 2) {
      nx = wkset->getDataSc("nx side");
      ux = wkset->getData("u[x] side");
      ny = wkset->getDataSc("ny side");
      uy = wkset->getData("u[y] side");
      parallel_for("porous HDIV flux ",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nx.extent(1); pt++) {
          AD udotn = ux(elem,pt)*nx(elem,pt);
          udotn += uy(elem,pt)*ny(elem,pt);
          uflux(elem,pt) = udotn;
        }
      });
    }
    else if (spaceDim == 3) {
      nx = wkset->getDataSc("nx side");
      ux = wkset->getData("u[x] side");
      ny = wkset->getDataSc("ny side");
      uy = wkset->getData("u[y] side");
      nz = wkset->getDataSc("nz side");
      uz = wkset->getData("u[z] side");
      
      parallel_for("porous HDIV flux ",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nx.extent(1); pt++) {
          AD udotn = ux(elem,pt)*nx(elem,pt);
          udotn += uy(elem,pt)*ny(elem,pt);
          udotn += uz(elem,pt)*nz(elem,pt);
          uflux(elem,pt) = udotn;
        }
      });
    }
    
    
  }
  
}

// ========================================================================================
// ========================================================================================

void porousMixed::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p")
      pnum = i;
    if (varlist[i] == "u")
      unum = i;
    if (varlist[i] == "dx")
      dxnum = i;
    if (varlist[i] == "dy")
      dynum = i;
    if (varlist[i] == "dz")
      dznum = i;
  }

  vector<string> auxvarlist = wkset->aux_varlist;
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p") {
      auxpnum = i;
      auxvar = "p";
    }
    if (auxvarlist[i] == "lambda") {
      auxpnum = i;
      auxvar = "lambda";
    }
    if (auxvarlist[i] == "pbndry") {
      auxpnum = i;
      auxvar = "pbndry";
    }
      
    if (auxvarlist[i] == "u")
      auxunum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousMixed::updatePerm(View_AD2 Kinv_xx, View_AD2 Kinv_yy, View_AD2 Kinv_zz) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("porous HDIV update perm",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<Kinv_xx.extent(1); pt++) {
      Kinv_xx(elem,pt) = 1.0/data(elem,0);
      Kinv_yy(elem,pt) = 1.0/data(elem,0);
      Kinv_zz(elem,pt) = 1.0/data(elem,0);
    }
  });
}


void porousMixed::updateKLPerm(View_AD2 KL_Kxx,
                               View_AD2 KL_Kyy, View_AD2 KL_Kzz) {
  
  //int spaceDim = wkset->dimension;
  int N = permKLx.N;
  
  bool foundUQ = false;
  auto KLUQcoeffs = wkset->getParameter("KLUQcoeffs",foundUQ);
  int prog = 0;
  auto xpts = wkset->getDataSc("x");
  
  if (foundUQ) {
    parallel_for("porous KL update perm",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      
      for (size_type k=0; k<KLUQcoeffs.extent(0); ++k) {
        ScalarT eval = permKLx.getEval(k);
        for (size_type pt=0; pt<xpts.extent(1); ++pt) {
          ScalarT evec = permKLx.getEvec(k,xpts(elem,pt));
          KL_Kxx(elem,pt) += KLUQcoeffs(k)*sqrt(eval)*evec;
        }
      }
    });
    
    prog += KLUQcoeffs.extent(0);
  }
  
  bool foundStoch = false;
  auto KLStochcoeffs = wkset->getParameter("KLStochcoeffs",foundStoch);
  
  if (foundStoch) {
    parallel_for("porous KL update perm",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      
      for (int k=prog; k<N; ++k) {
        ScalarT eval = permKLx.getEval(k);
        for (size_type pt=0; pt<xpts.extent(1); ++pt) {
          ScalarT evec = permKLx.getEvec(k,xpts(elem,pt));
          KL_Kxx(elem,pt) += KLStochcoeffs(k-prog)*sqrt(eval)*evec;
        }
      }
    });
  }
}


// ========================================================================================
// ========================================================================================

std::vector<string> porousMixed::getDerivedNames() {
  std::vector<string> derived;
  derived.push_back("permeability_x");
  derived.push_back("permeability_y");
  derived.push_back("permeability_z");
  return derived;
}

// ========================================================================================
// ========================================================================================

std::vector<View_AD2> porousMixed::getDerivedValues() {
  std::vector<View_AD2> derived;
  
  View_AD2 K_xx, K_yy, K_zz;
  
  Vista Kinv_xx, Kinv_yy, Kinv_zz;
  int spaceDim = wkset->dimension;
  
  // First compute Kinvxx, Kinvyy, Kinvzz
  
  auto wts = wkset->wts;
  {
    if (usePermData) {
      View_AD2 view_Kinv_xx("K inverse xx",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_yy("K inverse yy",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_zz("K inverse zz",wts.extent(0),wts.extent(1));
      this->updatePerm(view_Kinv_xx, view_Kinv_yy, view_Kinv_zz);
      Kinv_xx = Vista(view_Kinv_xx);
      Kinv_yy = Vista(view_Kinv_yy);
      Kinv_zz = Vista(view_Kinv_zz);
    }
    else {
      Kinv_xx = functionManager->evaluate("Kinv_xx","ip");
      Kinv_yy = functionManager->evaluate("Kinv_yy","ip");
      Kinv_zz = functionManager->evaluate("Kinv_zz","ip");
    }
    
    if (useKL) {
      View_AD2 KL_Kxx("KL K xx",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kyy("KL K yy",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kzz("KL K zz",wts.extent(0),wts.extent(1));
      this->updateKLPerm(KL_Kxx, KL_Kyy, KL_Kzz);
      
      View_AD2 new_Kxx("new K xx",wts.extent(0),wts.extent(1));
      View_AD2 new_Kyy("new K yy",wts.extent(0),wts.extent(1));
      View_AD2 new_Kzz("new K zz",wts.extent(0),wts.extent(1));
      
      parallel_for("porous HDIV update well source",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<KL_Kxx.extent(1); ++pt) {
          new_Kxx(elem,pt) = Kinv_xx(elem,pt)/exp(KL_Kxx(elem,pt));
        }
        if (spaceDim > 1) {
          for (size_type pt=0; pt<KL_Kyy.extent(1); ++pt) {
            new_Kyy(elem,pt) *= Kinv_yy(elem,pt)/exp(KL_Kyy(elem,pt));
          }
        }
        if (spaceDim > 2) {
          for (size_type pt=0; pt<KL_Kzz.extent(1); ++pt) {
            new_Kzz(elem,pt) *= Kinv_zz(elem,pt)/exp(KL_Kzz(elem,pt));
          }
        }
        
      });
      
      Kinv_xx = Vista(new_Kxx);
      Kinv_yy = Vista(new_Kyy);
      Kinv_zz = Vista(new_Kzz);
    }
    
  }
  
  K_xx = View_AD2("K xx",wts.extent(0),wts.extent(1));
  K_yy = View_AD2("K yy",wts.extent(0),wts.extent(1));
  K_zz = View_AD2("K zz",wts.extent(0),wts.extent(1));
    
  parallel_for("porous HDIV update well source",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<K_xx.extent(1); ++pt) {
      K_xx(elem,pt) = 1.0/Kinv_xx(elem,pt);
    }
    if (spaceDim > 1) {
      for (size_type pt=0; pt<K_yy.extent(1); ++pt) {
        K_yy(elem,pt) = 1.0/Kinv_yy(elem,pt);
      }
    }
    for (size_type pt=0; pt<K_zz.extent(1); ++pt) {
      K_zz(elem,pt) = 1.0/Kinv_zz(elem,pt);
    }
  });
  
  derived.push_back(K_xx);
  derived.push_back(K_yy);
  derived.push_back(K_zz);
  
  return derived;
}
