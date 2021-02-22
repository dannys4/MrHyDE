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

#include "boundaryCell.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

BoundaryCell::BoundaryCell(const Teuchos::RCP<CellMetaData> & cellData_,
                           const DRV nodes_,
                           const Kokkos::View<LO*,AssemblyDevice> localID_,
                           const Kokkos::View<LO*,AssemblyDevice> sideID_,
                           const int & sidenum_, const string & sidename_,
                           const int & cellID_,
                           LIDView LIDs_,
                           Kokkos::View<int****,HostDevice> sideinfo_,
                           Teuchos::RCP<discretization> & disc_) :
cellData(cellData_), localElemID(localID_), localSideID(sideID_),
sidenum(sidenum_), cellID(cellID_), nodes(nodes_), sideinfo(sideinfo_), sidename(sidename_), LIDs(LIDs_), disc(disc_)   {

  numElem = nodes.extent(0);

  auto LIDs_tmp = Kokkos::create_mirror_view(LIDs);
  Kokkos::deep_copy(LIDs_tmp,LIDs);  
  LIDs_host = LIDView_host("LIDs on host",LIDs.extent(0), LIDs.extent(1));
  Kokkos::deep_copy(LIDs_host,LIDs_tmp);
  
  if (cellData->storeAll) {
    int numip = cellData->ref_side_ip[0].extent(0);
    int dimension = cellData->dimension;
    ip = View_Sc3("physical ip",numElem, numip, dimension);
    normals = View_Sc3("physical normals",numElem, numip, dimension);
    tangents = View_Sc3("physical tangents",numElem, numip, dimension);
    wts = View_Sc2("physical wts",numElem, numip);
    hsize = View_Sc1("physical meshsize",numElem);
    orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
    disc->getPhysicalBoundaryData(cellData, nodes, localElemID, localSideID, orientation,
                                  ip, wts, normals, tangents, hsize,
                                  basis, basis_grad, basis_curl, basis_div, true, true);
    
  }
  else {
    orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
    disc->getPhysicalOrientations(cellData, localElemID, orientation, false);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParams(LIDView paramLIDs_) {
  paramLIDs = paramLIDs_;
  paramLIDs_host = LIDView_host("param LIDs on host", paramLIDs.extent(0), paramLIDs.extent(1));
  Kokkos::deep_copy(paramLIDs_host, paramLIDs);
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                                        const vector<DRV> & asideBasis,
                                        const vector<DRV> & asideBasisGrad) {
  
  auxbasisPointers = abasis_pointers;
  auxside_basis = asideBasis;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages) {
  vector<int> usebasis = usebasis_;
  
  // Set up the containers for usual solution storage
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numDOF_host.extent(0); i++) {
    if (cellData->numDOF_host(i) > maxnbasis) {
      maxnbasis = cellData->numDOF_host(i);
    }
  }
  u = View_Sc3("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  if (cellData->requiresAdjoint) {
    phi = View_Sc3("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  }
  if (cellData->requiresTransient) {
    u_prev = View_Sc4("u previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
    u_stage = View_Sc4("u stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  auto numParamDOF = cellData->numParamDOF;
  
  int maxnbasis = 0;
  for (size_type i=0; i<numParamDOF.extent(0); i++) {
    if (numParamDOF(i) > maxnbasis) {
      maxnbasis = numParamDOF(i);
    }
  }
  param = View_Sc3("param",numElem,numParamDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  auto numAuxDOF = Kokkos::create_mirror_view(cellData->numAuxDOF);
  Kokkos::deep_copy(numAuxDOF,cellData->numAuxDOF);
  int maxnbasis = 0;
  for (size_type i=0; i<numAuxDOF.extent(0); i++) {
    if (numAuxDOF(i) > maxnbasis) {
      maxnbasis = numAuxDOF(i);
    }
  }
  aux = View_Sc3("aux",numElem,numAuxDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateWorksetBasis() {
  
  if (cellData->storeAll) {
    wkset->wts_side = wts;
    wkset->h = hsize;
    wkset->setIP(ip," side");
    wkset->setNormals(normals);
    wkset->setTangents(tangents);
    wkset->basis_side = basis;
    wkset->basis_grad_side = basis_grad;
  }
  else {
    int numip = cellData->ref_side_ip[0].extent(0);
    int dimension = cellData->dimension;
    View_Sc3 tip("physical ip",numElem, numip, dimension);
    View_Sc3 tnormals("physical normals",numElem, numip, dimension);
    View_Sc3 ttangents("physical tangents",numElem, numip, dimension);
    View_Sc2 twts("physical wts",numElem, numip);
    View_Sc1 thsize("physical meshsize",numElem);
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalBoundaryData(cellData, nodes, localElemID,
                                  localSideID, orientation,
                                  tip, twts, tnormals, ttangents, thsize,
                                  tbasis, tbasis_grad, tbasis_curl, tbasis_div, true, false);
    
    wkset->wts_side = twts;
    wkset->h = thsize;
    wkset->setIP(tip," side");
    wkset->setNormals(tnormals);
    wkset->setTangents(ttangents);
    wkset->basis_side = tbasis;
    wkset->basis_grad_side = tbasis_grad;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeSoln(const int & seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  wkset->computeSolnSideIP();
  //wkset->computeParamSideIP();
  
  
  if (wkset->numAux > 0) {
    
    auto numAuxDOF = cellData->numAuxDOF;
    
    for (size_type var=0; var<numAuxDOF.extent(0); var++) {
      auto abasis = auxside_basis[auxusebasis[var]];
      auto off = subview(auxoffsets,var,ALL());
      string varname = wkset->aux_varlist[var];
      auto local_aux = wkset->getData("aux "+varname+" side");
      Kokkos::deep_copy(local_aux,0.0);
      //auto local_aux = Kokkos::subview(wkset->local_aux_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
      auto localID = localElemID;
      auto varaux = subview(aux,ALL(),var,ALL());
      if (seedwhat == 4) {
        parallel_for("bcell aux 4",
                     TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, 32),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
            for (size_type dof=0; dof<abasis.extent(1); ++dof) {
              AD auxval = AD(maxDerivs,off(dof), varaux(localID(elem),dof));
              local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
            }
          }
        });
      }
      else {
        parallel_for("bcell aux 5",
                     TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, 32),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
            for (size_type dof=0; dof<abasis.extent(1); ++dof) {
              AD auxval = varaux(localID(elem),dof);
              local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
            }
          }
        });
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                 const bool & compute_jacobian, const bool & compute_sens,
                                 const int & num_active_params, const bool & compute_disc_sens,
                                 const bool & compute_aux_sens, const bool & store_adjPrev,
                                 View_Sc3 local_res,
                                 View_Sc3 local_J) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  // Boundary contribution
  {
    Teuchos::TimeMonitor localtimer(*boundaryResidualTimer);
    
    this->updateWorksetBasis();
    
    wkset->sidename = sidename;
    wkset->currentside = sidenum;
    
    int seedwhat = 0;
    if (compute_jacobian) {
      if (compute_disc_sens) {
        seedwhat = 3;
      }
      else if (compute_aux_sens) {
        seedwhat = 4;
      }
      else {
        seedwhat = 1;
      }
    }
    
    if (isTransient) {
      wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat);
    }
    else { // steady-state
      wkset->computeSolnSteadySeeded(u, seedwhat);
    }
    wkset->computeParamSteadySeeded(param, seedwhat);
    
    this->computeSoln(seedwhat);
    wkset->computeParamSideIP();
    
    //wkset->resetResidual(numElem);
    wkset->resetResidual();
    
    cellData->physics_RCP->boundaryResidual(cellData->myBlock);
    
  }
  
  {
    Teuchos::TimeMonitor localtimer(*jacobianFillTimer);
    
    // Use AD residual to update local Jacobian
    if (compute_jacobian) {
      if (compute_disc_sens) {
        this->updateParamJac(local_J);
      }
      else if (compute_aux_sens){
        this->updateAuxJac(local_J);
      }
      else {
        this->updateJac(isAdjoint, local_J);
      }
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*residualFillTimer);
    
    // Update the local residual (forward mode)
    if (isAdjoint) {
      this->updateAdjointRes(compute_sens, local_res);
    }
    else {
      this->updateRes(compute_sens, local_res);
    }
    
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateRes(const bool & compute_sens, View_Sc3 local_res) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (compute_sens) {
    
    parallel_for("bcell update res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int r=0; r<local_res.extent(2); r++) {
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("bcell update res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j)).val();
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateAdjointRes(const bool & compute_sens, View_Sc3 local_res) {
  View_AD2 adjres_AD = wkset->adjrhs;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for("bcell update adjoint res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (int r=0; r<maxDerivs; r++) {
            local_res(elem,offsets(n,j),r) -= adjres_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("bcell update adjoint res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(elem,offsets(n,j),0) -= adjres_AD(elem,offsets(n,j)).val();
        }
      }
    });
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateJac(const bool & useadjoint, View_Sc3 local_J) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for("bcell update jac sens",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("bcell update jac",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateParamJac(View_Sc3 local_J) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = cellData->numParamDOF;
  
  parallel_for("bcell update param jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, 32),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(elem,offsets(n,j),paramoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateAuxJac(View_Sc3 local_J) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto aoffsets = auxoffsets;
  auto numAuxDOF = cellData->numAuxDOF;
  
  parallel_for("bcell update aux jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, 32),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),aoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute boundary regularization given the boundary discretized parameters
///////////////////////////////////////////////////////////////////////////////////////

AD BoundaryCell::computeBoundaryRegularization(const vector<ScalarT> reg_constants,
                                               const vector<int> reg_types,
                                               const vector<int> reg_indices,
                                               const vector<string> reg_sides) {
  
  AD reg = 0.0;
  
  //bool seedParams = true;
  //vector<vector<AD> > param_AD;
  //for (int n=0; n<paramindex.size(); n++) {
  //  param_AD.push_back(vector<AD>(paramindex[n].size()));
  //}
  //this->setLocalADParams(param_AD,seedParams);
  //int numip = wkset->numip;
  int numParams = reg_indices.size();
  /*
   for (int side=0; side<cellData->numSides; side++) {
   for (int e=0; e<numElem; e++) {
   if (sideinfo(e,0,side,0) > 0) { // Just checking the first variable should be sufficient
   onside = true;
   sname = sidenames[sideinfo(e,0,side,1)];
   }
   }
   
   if (onside) {*/
  
  //int sidetype = sideinfo[e](side,0); // 0-not on bndry, 1-Dirichlet bndry, 2-Neumann bndry
  //if (sidetype > 0) {
  //wkset->updateSide(nodes, sideip[side], sideijac[side], side);
  
  //    wkset->updateSide(nodes, sideip[side], sidewts[side],normals[side],sideijac[side], side);
  
  int numip = wts.extent(1);
  //int gside = sideinfo[e](side,1); // =-1 if is an interior edge
  
  //DRV side_weights = wkset->wts_side;
  int paramIndex, reg_type;
  ScalarT reg_constant;
  string reg_side;
  size_t found;
  
  // TMW needs to be updated
  /*
  for (int i = 0; i < numParams; i++) {
    paramIndex = reg_indices[i];
    reg_constant = reg_constants[i];
    reg_type = reg_types[i];
    reg_side = reg_sides[i];
    found = reg_side.find(sidename);
    if (found != string::npos) {
      
      //wkset->updateSide(sidenum, cellID);
      this->updateWorksetBasis();
      wkset->computeParamSideIP(sidenum, param, 3);
      
      AD p, dpdx, dpdy, dpdz; // parameters
      ScalarT offset = 1.0e-5;
      for (size_t e=0; e<numElem; e++) {
        //if (sideinfo(e,0,side,0) > 0) {
        for (int k = 0; k < numip; k++) {
          p = wkset->local_param_side(e,paramIndex,k,0);
          // L2
          if (reg_type == 0) {
            reg += 0.5*reg_constant*p*p*wts(e,k);
          }
          else {
            AD sx, sy ,sz;
            AD normal_dot;
            dpdx = wkset->local_param_grad_side(e,paramIndex,k,0); // param 0 in single trac inversion
            if (cellData->dimension > 1) {
              dpdy = wkset->local_param_grad_side(e,paramIndex,k,1);
            }
            if (cellData->dimension > 2) {
              dpdz = wkset->local_param_grad_side(e,paramIndex,k,2);
            }
            if (cellData->dimension == 1) {
              normal_dot = dpdx*normals(e,k,0);
              sx = dpdx - normal_dot*normals(e,k,0);
            }
            else if (cellData->dimension == 2) {
              normal_dot = dpdx*normals(e,k,0) + dpdy*normals(e,k,1);
              sx = dpdx - normal_dot*normals(e,k,0);
              sy = dpdy - normal_dot*normals(e,k,1);
            }
            else if (cellData->dimension == 3) {
              normal_dot = dpdx*normals(e,k,0) + dpdy*normals(e,k,1) + dpdz*normals(e,k,2);
              sx = dpdx - normal_dot*normals(e,k,0);
              sy = dpdy - normal_dot*normals(e,k,1);
              sz = dpdz - normal_dot*normals(e,k,2);
            }
            // H1
            if (reg_type == 1) {
              reg += 0.5*reg_constant*(sx*sx + sy*sy + sz*sz)*wts(e,k);
            }
            // TV
            else if (reg_type == 2) {
              reg += reg_constant*sqrt(sx*sx + sy*sy + sz*sz + offset*offset)*wts(e,k);
            }
          }
        }
        //}
      }
    }
  }
  //}
  //}
  //}
  */
  //cout << "reg = " << reg << endl;
  
  return reg;
  
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

View_Sc2 BoundaryCell::getDirichlet() {
  
  View_Sc2 dvals("initial values",numElem,LIDs.extent(1));
  this->updateWorksetBasis();
  //wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  
  Kokkos::View<string**,HostDevice> bcs = wkset->var_bcs;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  auto cnormals = normals;
  
  for (size_t n=0; n<wkset->varlist.size(); n++) {
    if (bcs(n,sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      auto dip = cellData->physics_RCP->getDirichlet(ip,n,
                                                     cellData->myBlock,
                                                     sidename,
                                                     wkset);
      int bind = wkset->usebasis[n];
      std::string btype = cellData->basis_types[bind];
      auto cbasis = basis[bind];
      
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bcell fill Dirichlet",
                     RangePolicy<AssemblyExec>(0,cwts.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,0)*cwts(e,j);
            }
          }
        });
      }
      else if (btype == "HDIV"){
        parallel_for("bcell fill Dirichlet HDIV",
                     RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++) {
                dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,s)*cnormals(e,j,s)*cwts(e,j);
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // not implemented yet
      }
    }
  }
  return dvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

View_Sc3 BoundaryCell::getMass() {
  
  View_Sc3 mass("local mass", numElem, LIDs.extent(1), LIDs.extent(1));
  
  Kokkos::View<string**,HostDevice> bcs = wkset->var_bcs;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    if (bcs(n,sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      int bind = wkset->usebasis[n];
      auto cbasis = basis[bind];
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      std::string btype = cellData->basis_types[bind];
      
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bcell compute mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
              }
            }
          }
        });
      }
      else if (btype == "HDIV"){
        auto cnormals = normals;
        parallel_for("bcell compute mass HDIV",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,s)*cnormals(e,k,s)*cbasis(e,j,k,s)*cnormals(e,k,s)*cwts(e,k);
                }
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // not implemented yet
      }
    }
  }
  return mass;
}

