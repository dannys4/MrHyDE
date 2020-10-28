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

#include "cell.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

cell::cell(const Teuchos::RCP<CellMetaData> & cellData_,
           const DRV nodes_,
           const Kokkos::View<LO*,AssemblyDevice> localID_,
           LIDView LIDs_,
           Kokkos::View<int****,HostDevice> sideinfo_,
           Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orientation_) :
LIDs(LIDs_), cellData(cellData_), localElemID(localID_),
sideinfo(sideinfo_), nodes(nodes_), orientation(orientation_)
{
  numElem = nodes.extent(0);
  useSensors = false;
  
  LIDs_host = LIDView_host("LIDs on host",LIDs.extent(0), LIDs.extent(1)); //Kokkos::create_mirror_view(LIDs);
  Kokkos::deep_copy(LIDs_host,LIDs);
  
  Teuchos::TimeMonitor localtimer(*buildBasisTimer);
  
  int dimension = cellData->dimension;
  int numip = cellData->ref_ip.extent(0);
  
  DRV tmpip("tmp ip", numElem, numip, dimension);
  CellTools::mapToPhysicalFrame(tmpip, cellData->ref_ip, nodes, *(cellData->cellTopo));
  ip = Kokkos::View<ScalarT***,AssemblyDevice>("ip",numElem,numip,dimension);
  Kokkos::deep_copy(ip,tmpip);
  
  DRV jacobian("jacobian", numElem, numip, dimension, dimension);
  CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
  
  DRV jacobianDet("determinant of jacobian", numElem, numip);
  DRV jacobianInv("inverse of jacobian", numElem, numip, dimension, dimension);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  CellTools::setJacobianInv(jacobianInv, jacobian);
  
  DRV tmpwts("tmp ip wts", numElem, numip);
  FuncTools::computeCellMeasure(tmpwts, jacobianDet, cellData->ref_wts);
  wts = Kokkos::View<ScalarT**,AssemblyDevice>("ip wts",numElem,numip);
  Kokkos::deep_copy(wts,tmpwts);
  
  this->computeSize();
 
  for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
    
    int numb = cellData->basis_pointers[i]->getCardinality();
    
    Kokkos::View<ScalarT****,AssemblyDevice> basis_vals, basis_grad_vals, basis_curl_vals, basis_node_vals;
    Kokkos::View<ScalarT***,AssemblyDevice> basis_div_vals;
    
    if (cellData->basis_types[i] == "HGRAD"){
      DRV bvals("basis",numElem,numb,numip);
      DRV tmp_bvals("basis tmp",numElem,numb,numip);
      FuncTools::HGRADtransformVALUE(tmp_bvals, cellData->ref_basis[i]);
      OrientTools::modifyBasisByOrientation(bvals, tmp_bvals, orientation,
                                            cellData->basis_pointers[i].get());
      basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis values", numElem, numb, numip, 1); // needs to be rank-4
      parallel_for("cell basis vals HGRAD",RangePolicy<AssemblyExec>(0,basis_vals.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<basis_vals.extent(1); dof++) {
          for (size_type pt=0; pt<basis_vals.extent(2); pt++) {
            basis_vals(elem,dof,pt,0) = bvals(elem,dof,pt);
          }
        }
      });
      
      DRV bgrad_tmp("basis grad tmp",numElem,numb,numip,dimension);
      DRV bgrad_vals("basis grad",numElem,numb,numip,dimension);
      FuncTools::HGRADtransformGRAD(bgrad_tmp, jacobianInv, cellData->ref_basis_grad[i]);
      OrientTools::modifyBasisByOrientation(bgrad_vals, bgrad_tmp, orientation,
                                            cellData->basis_pointers[i].get());
      basis_grad_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis vals",numElem,numb,numip,dimension);
      Kokkos::deep_copy(basis_grad_vals,bgrad_vals);
      
    }
    else if (cellData->basis_types[i] == "HVOL"){

      DRV bvals("basis",numElem,numb,numip);
      FuncTools::HGRADtransformVALUE(bvals, cellData->ref_basis[i]);

      basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis values", numElem, numb, numip, 1); // needs to be rank-4
      parallel_for("cell basis vals HVOL",RangePolicy<AssemblyExec>(0,basis_vals.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<basis_vals.extent(1); dof++) {
          for (size_type pt=0; pt<basis_vals.extent(2); pt++) {
            basis_vals(elem,dof,pt,0) = bvals(elem,dof,pt);
          }
        }
      });
    }
    else if (cellData->basis_types[i] == "HDIV"){
      
      DRV bvals("basis",numElem,numb,numip,dimension);
      DRV bvals_tmp("basis tmp",numElem,numb,numip,dimension);
      FuncTools::HDIVtransformVALUE(bvals_tmp, jacobian, jacobianDet, cellData->ref_basis[i]);
      OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                            cellData->basis_pointers[i].get());
      basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis values", numElem, numb, numip, dimension);
      Kokkos::deep_copy(basis_vals,bvals);
      
      if (cellData->requireBasisAtNodes) {
        DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
        DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
        FuncTools::HDIVtransformVALUE(bvals_tmp, jacobian, jacobianDet, cellData->ref_basis_nodes[i]);
        OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_node_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis values", numElem, numb, numip, dimension);
        Kokkos::deep_copy(basis_node_vals,bnode_vals);
      }
      
      DRV bdiv_vals("basis div",numElem,numb,numip);
      DRV bdiv_vals_tmp("basis div tmp",numElem,numb,numip);
      FuncTools::HDIVtransformDIV(bdiv_vals_tmp, jacobianDet, cellData->ref_basis_div[i]);
      OrientTools::modifyBasisByOrientation(bdiv_vals, bdiv_vals_tmp, orientation,
                                            cellData->basis_pointers[i].get());
      basis_div_vals = Kokkos::View<ScalarT***,AssemblyDevice>("basis div values", numElem, numb, numip); // needs to be rank-3
      Kokkos::deep_copy(basis_div_vals,bdiv_vals);
    }
    else if (cellData->basis_types[i] == "HCURL"){
      
      DRV bvals("basis",numElem,numb,numip,dimension);
      DRV bvals_tmp("basis tmp",numElem,numb,numip,dimension);
      FuncTools::HCURLtransformVALUE(bvals_tmp, jacobianInv, cellData->ref_basis[i]);
      OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                            cellData->basis_pointers[i].get());
      basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis values", numElem, numb, numip, dimension);
      Kokkos::deep_copy(basis_vals,bvals);
      
      if (cellData->requireBasisAtNodes) {
        DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
        DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
        FuncTools::HCURLtransformVALUE(bvals_tmp, jacobianInv, cellData->ref_basis_nodes[i]);
        OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_node_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis values", numElem, numb, numip, dimension);
        Kokkos::deep_copy(basis_node_vals,bnode_vals);
        
      }

      DRV bcurl_vals("basis curl",numElem,numb,numip,dimension);
      DRV bcurl_vals_tmp("basis curl tmp",numElem,numb,numip,dimension);
      FuncTools::HCURLtransformCURL(bcurl_vals_tmp, jacobian, jacobianDet, cellData->ref_basis_curl[i]);
      OrientTools::modifyBasisByOrientation(bcurl_vals, bcurl_vals_tmp, orientation,
                                            cellData->basis_pointers[i].get());
      basis_curl_vals = Kokkos::View<ScalarT****,AssemblyDevice>("basis curl values", numElem, numb, numip, dimension);
      Kokkos::deep_copy(basis_curl_vals, bcurl_vals);
      
    }
    basis.push_back(basis_vals);
    basis_grad.push_back(basis_grad_vals);
    basis_div.push_back(basis_div_vals);
    basis_curl.push_back(basis_curl_vals);
    basis_nodes.push_back(basis_node_vals);
  }
  
  if (cellData->build_face_terms) {
    Teuchos::TimeMonitor localtimer(*buildFaceBasisTimer);
    for (size_type side=0; side<cellData->numSides; side++) {
      auto ref_ip = cellData->ref_side_ip[side];
      auto ref_wts = cellData->ref_side_wts[side];
      
      int dimension = cellData->dimension;
      int numip = ref_ip.extent(0);
      
      // Step 1: fill in ip_side, wts_side and normals
      DRV sip("side ip", numElem, numip, dimension);
      DRV jac("bijac", numElem, numip, dimension, dimension);
      DRV jacDet("bijacDet", numElem, numip);
      DRV jacInv("bijacInv", numElem, numip, dimension, dimension);
      DRV swts("wts_side", numElem, numip);
      DRV snormals("normals", numElem, numip, dimension);
      DRV tangents("tangents", numElem, numip, dimension);
      
      CellTools::mapToPhysicalFrame(sip, ref_ip, nodes, *(cellData->cellTopo));
      CellTools::setJacobian(jac, ref_ip, nodes, *(cellData->cellTopo));
      CellTools::setJacobianInv(jacInv, jac);
      CellTools::setJacobianDet(jacDet, jac);
      
      if (dimension == 2) {
        auto ref_tangents = cellData->ref_side_tangents[side];
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(tangents, jac, ref_tangents);
        
        DRV rotation("rotation matrix",dimension,dimension);
        rotation(0,0) = 0;  rotation(0,1) = 1;
        rotation(1,0) = -1; rotation(1,1) = 0;
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(snormals, rotation, tangents);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(swts, tangents, Intrepid2::NORM_TWO);
        Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(swts, swts, ref_wts);
        
      }
      else if (dimension == 3) {
        
        auto ref_tangentsU = cellData->ref_side_tangentsU[side];
        auto ref_tangentsV = cellData->ref_side_tangentsV[side];
        
        DRV faceTanU("face tangent U", numElem, numip, dimension);
        DRV faceTanV("face tangent V", numElem, numip, dimension);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanU, jac, ref_tangentsU);
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanV, jac, ref_tangentsV);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::vecprod(snormals, faceTanU, faceTanV);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(swts, snormals, Intrepid2::NORM_TWO);
        Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(swts, swts, ref_wts);
        
      }
      
      // scale the normal vector (we need unit normal...)
      
      this->rescaleNormals(snormals);
      
      Kokkos::View<ScalarT***,AssemblyDevice> sideip("side ip", numElem, numip, dimension);
      Kokkos::View<ScalarT***,AssemblyDevice> sidenormals("side normals", numElem, numip, dimension);
      Kokkos::View<ScalarT**,AssemblyDevice> sidewts("side wts", numElem, numip);
      
      Kokkos::deep_copy(sideip,sip);
      Kokkos::deep_copy(sidenormals,snormals);
      Kokkos::deep_copy(sidewts,swts);
      
      ip_face.push_back(sideip);
      wts_face.push_back(sidewts);
      normals_face.push_back(sidenormals);
      
      // Step 2: define basis functions at these integration points
      vector<Kokkos::View<ScalarT****,AssemblyDevice> > currbasis, currbasisgrad;
      for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
        int numb = cellData->basis_pointers[i]->getCardinality();
        Kokkos::View<ScalarT****,AssemblyDevice> basis_vals, basis_grad_vals;//, basis_div_vals, basis_curl_vals;
        
        auto ref_basis_vals = cellData->ref_side_basis[side][i];
        
        if (cellData->basis_types[i] == "HGRAD"){
          
          DRV bvals_tmp("tmp basis_vals",numElem, numb, numip);
          DRV bvals("basis_vals",numElem, numb, numip);
          FuncTools::HGRADtransformVALUE(bvals_tmp, ref_basis_vals);
          OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
          parallel_for("cell basis vals HVOL",RangePolicy<AssemblyExec>(0,basis_vals.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type dof=0; dof<basis_vals.extent(1); dof++) {
              for (size_type pt=0; pt<basis_vals.extent(2); pt++) {
                basis_vals(elem,dof,pt,0) = bvals(elem,dof,pt);
              }
            }
          });
          
          auto ref_basis_grad_vals = cellData->ref_side_basis_grad[side][i];
          DRV bgrad_vals_tmp("tmp basis_grad_vals",numElem, numb, numip, dimension);
          DRV bgrad_vals("basis_grad_vals",numElem, numb, numip, dimension);
          FuncTools::HGRADtransformGRAD(bgrad_vals_tmp, jacInv, ref_basis_grad_vals);
          OrientTools::modifyBasisByOrientation(bgrad_vals, bgrad_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_grad_vals = Kokkos::View<ScalarT****,AssemblyDevice>("face basis grad vals",numElem,numb,numip,dimension); // Needs to be rank-4
          Kokkos::deep_copy(basis_grad_vals,bgrad_vals);
          
        }
        else if (cellData->basis_types[i] == "HVOL"){
          
          DRV bvals("basis_vals",numElem, numb, numip);
          FuncTools::HGRADtransformVALUE(bvals, ref_basis_vals);
          basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
          parallel_for("cell basis vals HVOL",RangePolicy<AssemblyExec>(0,basis_vals.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type dof=0; dof<basis_vals.extent(1); dof++) {
              for (size_type pt=0; pt<basis_vals.extent(2); pt++) {
                basis_vals(elem,dof,pt,0) = bvals(elem,dof,pt);
              }
            }
          });
          
        }
        else if (cellData->basis_types[i] == "HDIV"){
          
          DRV bvals_tmp("tmp basis_vals",numElem, numb, numip, dimension);
          DRV bvals("basis_vals",numElem, numb, numip, dimension);
          
          FuncTools::HDIVtransformVALUE(bvals_tmp, jac, jacDet, ref_basis_vals);
          OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("face basis vals",numElem,numb,numip,dimension);
          Kokkos::deep_copy(basis_vals,bvals);
          
        }
        else if (cellData->basis_types[i] == "HCURL"){
          //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
        }
        else if (cellData->basis_types[i] == "HFACE"){
          
          DRV bvals("basis_vals",numElem, numb, numip);
          DRV bvals_tmp("basisvals_Transformed",numElem, numb, numip);
          FuncTools::HGRADtransformVALUE(bvals_tmp, ref_basis_vals);
          OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = Kokkos::View<ScalarT****,AssemblyDevice>("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
          parallel_for("cell basis vals HVOL",RangePolicy<AssemblyExec>(0,basis_vals.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type dof=0; dof<basis_vals.extent(1); dof++) {
              for (size_type pt=0; pt<basis_vals.extent(2); pt++) {
                basis_vals(elem,dof,pt,0) = bvals(elem,dof,pt);
              }
            }
          });
          
        }
        
        currbasis.push_back(basis_vals);
        currbasisgrad.push_back(basis_grad_vals);
      }
      basis_face.push_back(currbasis);
      basis_grad_face.push_back(currbasisgrad);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSize() {
  // These computations are performed on Host until the std namespace is removed

  hsize = Kokkos::View<ScalarT*,AssemblyDevice>("element sizes", numElem);
  auto host_wts = Kokkos::create_mirror_view(wts);
  Kokkos::deep_copy(host_wts,wts);
  
  auto host_hsize = Kokkos::create_mirror_view(hsize); 
  parallel_for("cell hsize",RangePolicy<HostExec>(0,host_wts.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    ScalarT vol = 0.0;
    for (size_type i=0; i<host_wts.extent(1); i++) {
      vol += host_wts(e,i);
    }
    ScalarT dimscl = 1.0/(ScalarT)ip.extent(2);
    host_hsize(e) = pow(vol,dimscl);
  });
  Kokkos::deep_copy(hsize,host_hsize);

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::rescaleNormals(DRV snormals) {
  // These computations are performed on Host until the std namespace is removed

  auto host_snormals = Kokkos::create_mirror_view(snormals);
  Kokkos::deep_copy(host_snormals, snormals);
  parallel_for("cell normal unnecessary rescale",RangePolicy<HostExec>(0,host_snormals.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    for (size_type j=0; j<host_snormals.extent(1); j++ ) {
      ScalarT normalLength = 0.0;
      for (size_type sd=0; sd<host_snormals.extent(2); sd++) {
        normalLength += host_snormals(e,j,sd)*host_snormals(e,j,sd);
      }
      normalLength = std::sqrt(normalLength);
      for (size_type sd=0; sd<host_snormals.extent(2); sd++) {
        host_snormals(e,j,sd) = host_snormals(e,j,sd) / normalLength;
      }
    }
  });
  Kokkos::deep_copy(snormals, host_snormals);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParams(LIDView paramLIDs_) {
  
  paramLIDs = paramLIDs_;
  paramLIDs_host = LIDView_host("param LIDs on host", paramLIDs.extent(0), paramLIDs.extent(1));//Kokkos::create_mirror_view(paramLIDs);
  Kokkos::deep_copy(paramLIDs_host, paramLIDs);
  
  // This has now been set
  //numParamDOF = cellData->numParamDOF;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void cell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
                                const vector<DRV> & abasisGrad, const vector<vector<DRV> > & asideBasis,
                                const vector<vector<DRV> > & asideBasisGrad) {
  
  for (size_t b=0; b<abasis_pointers.size(); b++) {
    auxbasisPointers.push_back(abasis_pointers[b]);
  }
  for (size_t b=0; b<abasis.size(); b++) {
    auxbasis.push_back(abasis[b]);
    //auxbasisGrad.push_back(abasisGrad[b]);
  }
  
  for (size_t s=0; s<asideBasis.size(); s++) {
    auxside_basis.push_back(asideBasis[s]);
    //auxside_basisGrad.push_back(asideBasisGrad[s]);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void cell::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the regular parameters (everything but discretized)
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  cellData->physics_RCP->updateParameters(params, paramnames);
}


///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages) {
  vector<int> usebasis = usebasis_;
  //num_stages = nstages;
  
  // Set up the containers for usual solution storage
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numDOF_host.extent(0); i++) {
    if (cellData->numDOF_host(i) > maxnbasis) {
      maxnbasis = cellData->numDOF_host(i);
    }
  }
  //maxnbasis *= nstages;
  u = Kokkos::View<ScalarT***,AssemblyDevice>("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  phi = Kokkos::View<ScalarT***,AssemblyDevice>("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  
  // This does add a little extra un-used memory for steady-state problems, but not a concern
  u_prev = Kokkos::View<ScalarT****,AssemblyDevice>("u previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
  phi_prev = Kokkos::View<ScalarT****,AssemblyDevice>("phi previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
  
  u_stage = Kokkos::View<ScalarT****,AssemblyDevice>("u stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
  phi_stage = Kokkos::View<ScalarT****,AssemblyDevice>("phi stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
 
  u_avg = Kokkos::View<ScalarT***,AssemblyDevice>("u spatial average",numElem,cellData->numDOF.extent(0),cellData->dimension);
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  //wkset->paramusebasis = pusebasis_;
  
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numParamDOF.extent(0); i++) {
    if (cellData->numParamDOF(i) > maxnbasis) {
      maxnbasis = cellData->numParamDOF(i);
    }
  }
  param = Kokkos::View<ScalarT***,AssemblyDevice>("param",numElem,cellData->numParamDOF.extent(0),maxnbasis);
  param_avg = Kokkos::View<ScalarT***,AssemblyDevice>("param",numElem,cellData->numParamDOF.extent(0), cellData->dimension);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numAuxDOF.extent(0); i++) {
    if (cellData->numAuxDOF(i) > maxnbasis) {
      maxnbasis = cellData->numAuxDOF(i);
    }
  }
  aux = Kokkos::View<ScalarT***,AssemblyDevice>("aux",numElem,cellData->numAuxDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateWorksetBasis() {
  //wkset->ip = ip;
  wkset->wts = wts;
  wkset->h = hsize;
  
  // Check if we have a full workset
  if (ip.extent(0) < wkset->ip.extent(0)) {
    auto wip = wkset->ip;
    parallel_for("wkset transient soln 1",RangePolicy<AssemblyExec>(0,ip.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<ip.extent(1); pt++) {
        for (size_type dim=0; dim<ip.extent(2); dim++) {
          wip(elem,pt,dim) = ip(elem,pt,dim);
        }
      }
    });
  }
  else {
    Kokkos::deep_copy(wkset->ip,ip);
  }
  
  wkset->basis = basis;
  wkset->basis_grad = basis_grad;
  wkset->basis_div = basis_div;
  wkset->basis_curl = basis_curl;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnVolIP() {
  // seedwhat key: 0-nothing; 1-sol; 2-soldot; 3-disc.params.; 4-aux.vars
  // Note: seeding u_dot is now deprecated
  
  Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  //wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  this->updateWorksetBasis();
  Kokkos::fence();

  wkset->computeSolnVolIP();
  Kokkos::fence();
  
  //wkset->computeParamVolIP(param, seedwhat);
  if (cellData->compute_sol_avg) {
    this->computeSolAvg();
  }
  Kokkos::fence();

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolAvg() {
  
  // THIS FUNCTION ASSUMES THAT THE WORKSET BASIS HAS BEEN UPDATED
  // AND THE SOLUTION HAS BEEN COMPUTED AT THE VOLUMETRIC IP
  
  Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  Kokkos::View<AD****,AssemblyDevice> sol = wkset->local_soln;
 
  auto cwts = wts;
  auto avg = u_avg; 

  parallel_for("cell sol avg",RangePolicy<AssemblyExec>(0,avg.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    for (size_type dof=0; dof<sol.extent(1); dof++) {
      for (size_type dim=0; dim<sol.extent(3); dim++) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
          solavg += sol(elem,dof,pt,dim).val()*cwts(elem,pt);
        }
        avg(elem,dof,dim) = solavg/avgwt;
      }
    }
  });
  
  if (param_avg.extent(1) > 0) {
    Kokkos::View<AD****,AssemblyDevice> psol = wkset->local_param;
    auto pavg = param_avg;

    parallel_for("cell param avg",RangePolicy<AssemblyExec>(0,pavg.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
      ScalarT avgwt = 0.0;
      for (size_type pt=0; pt<cwts.extent(1); pt++) {
        avgwt += cwts(elem,pt);
      }
      for (size_type dof=0; dof<psol.extent(1); dof++) {
        for (size_type dim=0; dim<psol.extent(3); dim++) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<psol.extent(2); pt++) {
            solavg += psol(elem,dof,pt,dim).val()*cwts(elem,pt);
          }
          pavg(elem,dof,dim) = solavg/avgwt;
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateWorksetFaceBasis(const size_t & facenum) {
  
  //wkset->ip_side = ip_face[facenum];
  wkset->wts_side = wts_face[facenum];
  //wkset->normals = normals_face[facenum];
  wkset->h = hsize;
  
  if (ip_face[facenum].extent(0) < wkset->ip_side.extent(0)) {
    auto fip = ip_face[facenum];
    auto wip = wkset->ip_side;
    auto wnorm = wkset->normals;
    auto fnorm = normals_face[facenum];
    parallel_for("wkset transient soln 1",RangePolicy<AssemblyExec>(0,fip.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<fip.extent(1); pt++) {
        for (size_type dim=0; dim<fip.extent(2); dim++) {
          wip(elem,pt,dim) = fip(elem,pt,dim);
          wnorm(elem,pt,dim) = fnorm(elem,pt,dim);
        }
      }
    });
  }
  else {
    Kokkos::deep_copy(wkset->ip_side,ip_face[facenum]);
    Kokkos::deep_copy(wkset->normals,normals_face[facenum]);
  }
  
  wkset->basis_face = basis_face[facenum];
  wkset->basis_grad_face = basis_grad_face[facenum];
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnFaceIP(const size_t & facenum) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  this->updateWorksetFaceBasis(facenum);
  //wkset->updateFace(nodes, orientation, facenum);
  wkset->computeSolnFaceIP();
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetPrevSoln() {
  
  auto sol = u;
  auto sol_prev = u_prev;
  
  // shift previous step solns
  if (sol_prev.extent(3)>1) {
    parallel_for("cell shift prev soln",RangePolicy<AssemblyExec>(0,sol_prev.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type i=0; i<sol_prev.extent(1); i++) {
        for (size_type j=0; j<sol_prev.extent(2); j++) {
          for (size_type s=sol_prev.extent(3)-1; s>0; s--) {
            sol_prev(e,i,j,s) = sol_prev(e,i,j,s-1);
          }
        }
      }
    });
  }
  
  // copy current u into first step
  parallel_for("cell copy prev soln",RangePolicy<AssemblyExec>(0,sol.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    for (size_type i=0; i<sol.extent(1); i++) {
      for (size_type j=0; j<sol.extent(2); j++) {
        sol_prev(e,i,j,0) = sol(e,i,j);
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetStageSoln() {
  
  auto sol = u;
  auto sol_stage = u_stage;
  
  parallel_for("cell reset stage soln",RangePolicy<AssemblyExec>(0,sol_stage.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    for (size_type i=0; i<sol_stage.extent(1); i++) {
      for (size_type j=0; j<sol_stage.extent(2); j++) {
        for (size_type k=0; k<sol_stage.extent(3); k++) {
          sol_stage(e,i,j,k) = sol(e,i,j);
        }
      }
    }
  });
  //KokkosTools::print(u_stage);
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateStageSoln() {
  
  auto sol = u;
  auto sol_stage = u_stage;
  
  // add u into the current stage soln (done after stage solution is computed)
  auto snum = wkset->current_stage_KV;
  parallel_for("cell update stage soln",RangePolicy<AssemblyExec>(0,sol_stage.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    int stage = snum(0);
    for (size_type i=0; i<sol_stage.extent(1); i++) {
      for (size_type j=0; j<sol_stage.extent(2); j++) {
        sol_stage(e,i,j,stage) = sol(e,i,j);
      }
    }
  });
  //KokkosTools::print(u_stage);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                         const bool & assemble_volume_terms,
                         const bool & assemble_face_terms) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  bool fixJacDiag = false;
  
  wkset->resetResidual();
  
  if (isAdjoint) {
    wkset->resetAdjointRHS();
  }
  
  //////////////////////////////////////////////////////////////
  // Compute the AD-seeded solutions at integration points
  //////////////////////////////////////////////////////////////
  
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
   
  if (!(cellData->multiscale)) {
    if (isTransient) {
      wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat);
    }
    else { // steady-state
      wkset->computeSolnSteadySeeded(u, seedwhat);
    }
    Kokkos::fence();
    this->computeSolnVolIP();
    wkset->computeParamVolIP(param, seedwhat);
    
  }
  Kokkos::fence();
  
  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  if (assemble_volume_terms) {
    Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (cellData->multiscale) {
      int sgindex = subgrid_model_index[subgrid_model_index.size()-1];
      subgridModels[sgindex]->subgridSolver(u, phi, wkset->time, isTransient, isAdjoint,
                                            compute_jacobian, compute_sens, num_active_params,
                                            compute_disc_sens, compute_aux_sens,
                                            *wkset, subgrid_usernum, 0,
                                            subgradient, store_adjPrev);
      fixJacDiag = true;
    }
    else {
      cellData->physics_RCP->volumeResidual(cellData->myBlock);
    }
  }
  Kokkos::fence();
  
  // Edge/face contribution
  if (assemble_face_terms) {
    Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (cellData->multiscale) {
      // do nothing
    }
    else {
      for (size_t s=0; s<cellData->numSides; s++) {
        this->computeSolnFaceIP(s);
        cellData->physics_RCP->faceResidual(cellData->myBlock);
      }
    }
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
  
  if (compute_jacobian && fixJacDiag) {
    this->fixDiagJac(local_J, local_res);
  }
  
  // Update the local residual
  {
    Teuchos::TimeMonitor localtimer(*residualFillTimer);
    if (isAdjoint) {
      this->updateAdjointRes(compute_sens, local_res);
    }
    else {
      this->updateRes(compute_sens, local_res);
    }
  }
  
  {
    if (isAdjoint) {
      Teuchos::TimeMonitor localtimer(*adjointResidualTimer);
      this->updateAdjointRes(compute_jacobian, isTransient,
                             compute_aux_sens, store_adjPrev,
                             local_J, local_res);
      
      
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for("cell update res sens",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type r=0; r<local_res.extent(2); r++) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("cell update res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(e,offsets(n,j),0) -= res_AD(e,offsets(n,j)).val();
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  auto adjres_AD = wkset->adjrhs;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for("cell update adjoint res sens",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (int r=0; r<maxDerivs; r++) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= adjres_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("cell update adjoint res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(e,offsets(n,j),0) -= adjres_AD(e,offsets(n,j)).val();
        }
      }
    });
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAdjointRes(const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  // Update residual (adjoint mode)
  // Adjoint residual: -dobj/du - J^T * phi + 1/dt*M^T * phi_prev
  // J = 1/dtM + A
  // adj_prev stores 1/dt*M^T * phi_prev where M is evaluated at appropriate time
  
  // TMW: This will not work on a GPU
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (!(cellData->mortar_objective) && cellData->response_type != "discrete") {
    for (size_t w=1; w < cellData->dimension+2; w++) {
      
      Kokkos::View<AD**,AssemblyDevice> obj = computeObjective(wkset->time, 0, w);
      
      //int numDerivs = 0;
      if (useSensors) {
        if (numSensors > 0) {
          
          for (size_t s=0; s<numSensors; s++) {
            int e = sensorElem[s];
            auto cobj = Kokkos::subview(obj,Kokkos::ALL(), s);
            for (size_type n=0; n<numDOF.extent(0); n++) {
              auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
              Kokkos::View<int[2],AssemblyDevice> scratch("scratch pad");
              auto scratch_host = Kokkos::create_mirror_view(scratch);
              scratch_host(0) = n;
              scratch_host(1) = e;
              Kokkos::deep_copy(scratch,scratch_host);
              auto sres = Kokkos::subview(local_res,e,Kokkos::ALL(),0);
              if (w == 1) {
                auto sbasis = Kokkos::subview(sensorBasis[s][wkset->usebasis[n]],0,Kokkos::ALL(),s);
                parallel_for("cell adjust adjoint res sensor",RangePolicy<AssemblyExec>(0,cellData->numDOF_host(n)), KOKKOS_LAMBDA (const int j ) {
                  int nn = scratch(0);
                  int elem = scratch(1);
                  for (int i=0; i<numDOF(nn); i++) {
                    sres(off(j)) += -cobj(elem).fastAccessDx(off(i))*sbasis(j);
                  }
                });
              }
              else {
                auto sbasis = Kokkos::subview(sensorBasisGrad[s][wkset->usebasis[n]],0,Kokkos::ALL(),s,w-2);
                parallel_for("cell adjust adjoint res sensor grad", RangePolicy<AssemblyExec>(0,cellData->numDOF_host(n)), KOKKOS_LAMBDA (const int j ) {
                  int nn = scratch(0);
                  int elem = scratch(1);
                  for (int i=0; i<numDOF(nn); i++) {
                    sres(off(j)) += -cobj(elem).fastAccessDx(off(i))*sbasis(j);
                  }
                });
              }
            }
          }
        }
      }
      else {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          Kokkos::View<int[2],AssemblyDevice> scratch("scratch pad");
          auto scratch_host = Kokkos::create_mirror_view(scratch);
          scratch_host(0) = n;
          Kokkos::deep_copy(scratch,scratch_host);
          if (w==1) {
            int bnum = wkset->usebasis[n];
            DRV sbasis = basis[bnum];
            
            std::string btype = wkset->basis_types[bnum];
            if (btype == "HDIV" || btype == "HCURL") {
              parallel_for("cell adjust adjoint res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
                int nn = scratch(0);
                for (int j=0; j<numDOF(nn); j++) {
                  for (int i=0; i<numDOF(nn); i++) {
                    for (size_type s=0; s<sbasis.extent(2); s++) {
                      for (size_type d=0; d<sbasis.extent(3); d++) {
                        ScalarT Jval2 = sbasis(e,j,s,d);
                        local_res(e,offsets(nn,j),0) += -obj(e,s).fastAccessDx(offsets(nn,i))*Jval2;
                      }
                    }
                  }
                }
              });
            }
            else {
              parallel_for("cell adjust adjoint res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
                int nn = scratch(0);
                for (int j=0; j<numDOF(nn); j++) {
                  for (int i=0; i<numDOF(nn); i++) {
                    for (size_type s=0; s<sbasis.extent(2); s++) {
                      local_res(e,offsets(nn,j),0) += -obj(e,s).fastAccessDx(offsets(nn,i))*sbasis(e,j,s);
                    }
                  }
                }
              });
            }
          }
          else {
            auto sbasis = Kokkos::subview(basis_grad[wkset->usebasis[n]],Kokkos::ALL(),
                                          Kokkos::ALL(), Kokkos::ALL(), w-2);
            parallel_for("cell adjust adjoint res grad",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
              int nn = scratch(0);
              for (int j=0; j<numDOF(nn); j++) {
                for (int i=0; i<numDOF(nn); i++) {
                  for (size_type s=0; s<sbasis.extent(2); s++) {
                    local_res(e,offsets(nn,j),0) += -obj(e,s).fastAccessDx(offsets(nn,i))*sbasis(e,j,s);
                  }
                }
              }
            });
          }
        }
      }
    }
  }
  if (compute_jacobian) {
    parallel_for("cell adjust adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_res(e,offsets(n,j),0) += -local_J(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
            }
          }
        }
      }
    });
    
    if (isTransient) {
      
      // Previous step contributions for the residual
      parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),0) += -adj_prev(e,offsets(n,j),0);
          }
        }
      });
      /*
      // Previous stage contributions for the residual
      if (adj_stage_prev.extent(2) > 0) {
        parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type n=0; n<numDOF.extent(0); n++) {
            for (int j=0; j<numDOF(n); j++) {
              local_res(e,offsets(n,j),0) += -adj_stage_prev(e,offsets(n,j),0);
            }
          }
        });
      }
      */
      
      if (!compute_aux_sens && store_adjPrev) {
        
        //////////////////////////////////////////
        // Multi-step
        //////////////////////////////////////////
        
        // Move vectors up
        parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,adj_prev.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type step=1; step<adj_prev.extent(2); step++) {
            for (size_type n=0; n<adj_prev.extent(1); n++) {
              adj_prev(e,n,step-1) = adj_prev(e,n,step);
            }
          }
          size_type numsteps = adj_prev.extent(2);
          for (size_type n=0; n<adj_prev.extent(1); n++) {
            adj_prev(e,n,numsteps-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        int seedwhat = 2; // 2 for J wrt previous step solutions
        for (size_type step=0; step<u_prev.extent(3); step++) {
          wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat, step);
          wkset->computeParamVolIP(param, seedwhat);
          this->computeSolnVolIP();
       
          wkset->resetResidual();
          
          cellData->physics_RCP->volumeResidual(cellData->myBlock);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = Kokkos::subview(adj_prev,Kokkos::ALL(), Kokkos::ALL(), step);
          parallel_for("cell adjust transient adjoint jac 2",RangePolicy<AssemblyExec>(0,Jdot.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
                  }
                }
                cadj(e,offsets(n,j)) += aPrev;
              }
            }
          });
        }
        
        //////////////////////////////////////////
        // Multi-stage
        //////////////////////////////////////////
        /*
        // Move vectors up
        parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,adj_stage_prev.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type stage=1; stage<adj_stage_prev.extent(2); stage++) {
            for (size_type n=0; n<adj_stage_prev.extent(1); n++) {
              adj_stage_prev(e,n,stage-1) = adj_stage_prev(e,n,stage);
            }
          }
          size_type numstages = adj_stage_prev.extent(2);
          for (size_type n=0; n<adj_stage_prev.extent(1); n++) {
            adj_stage_prev(e,n,numstages-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        seedwhat = 3; // 3 for J wrt previous stage solutions
        for (size_type stage=0; stage<u_prev.extent(3); stage++) {
          wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat, stage);
          wkset->computeParamVolIP(param, seedwhat);
          this->computeSolnVolIP();
          
          wkset->resetResidual();
          
          cellData->physics_RCP->volumeResidual(cellData->myBlock);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = Kokkos::subview(adj_stage_prev,Kokkos::ALL(), Kokkos::ALL(), stage);
          parallel_for("cell adjust transient adjoint jac 2",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
                  }
                }
                cadj(e,offsets(n,j)) += aPrev;
              }
            }
          });
        }*/
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for("cell update jac adj",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("cell update jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(n,j),offsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  AssemblyExec::execution_space().fence();
}

///////////////////////////////////////////////////////////////////////////////////////
// Place ones on the diagonal of the Jacobian if
///////////////////////////////////////////////////////////////////////////////////////

void cell::fixDiagJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  ScalarT JTOL = 1.0E-14;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  parallel_for("cell fix diag",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type var=0; var<offsets.extent(0); var++) {
      for (int dof=0; dof<numDOF(var); dof++) {
        int diag = offsets(var,dof);
        if (abs(local_J(elem,diag,diag)) < JTOL) {
          local_res(elem,diag,0) = -u(elem,var,dof);
          for (int j=0; j<numDOF(var); j++) {
            ScalarT scale = 1.0/((ScalarT)numDOF(var)-1.0);
            local_J(elem,diag,offsets(var,j)) = -scale;
            if (j!=dof)
              local_res(elem,diag,0) += scale*u(elem,var,j);
          }
          local_J(elem,diag,diag) = 1.0;
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = cellData->numParamDOF;
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  parallel_for("cell update param jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    for (size_type n=0; n<numDOF.extent(0); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(e,offsets(n,j),paramoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto aoffsets = auxoffsets;
  auto numDOF = cellData->numDOF;
  auto numAuxDOF = cellData->numAuxDOF;
  
  parallel_for("cell update aux jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
    for (size_type n=0; n<numDOF.extent(0); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(e,offsets(n,j),auxoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> cell::getInitial(const bool & project, const bool & isAdjoint) {
  Kokkos::View<ScalarT**,AssemblyDevice> initialvals("initial values",numElem,LIDs.extent(1));
  this->updateWorksetBasis();
  
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  
  if (project) { // works for any basis
    Kokkos::View<ScalarT***,AssemblyDevice> initialip = cellData->physics_RCP->getInitial(wkset->ip,
                                                                                          cellData->myBlock,
                                                                                          project,
                                                                                          wkset);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = basis[wkset->usebasis[n]];
      auto off = Kokkos::subview(offsets, n, Kokkos::ALL());
      auto initvar = Kokkos::subview(initialip, Kokkos::ALL(), n, Kokkos::ALL());
      parallel_for("cell get init",RangePolicy<AssemblyExec>(0,initvar.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for(size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          for(size_type pt=0; pt<cwts.extent(1); pt++ ) {
            initialvals(elem,off(dof)) += initvar(elem,pt)*cbasis(elem,dof,pt,0)*cwts(elem,pt);
          }
        }
      });
    }
  }
  else { // only works if using HGRAD linear basis
    
    Kokkos::View<ScalarT***,AssemblyDevice> initialnodes = cellData->physics_RCP->getInitial(nodes,
                                                                                             cellData->myBlock,
                                                                                             project,
                                                                                             wkset);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto off = Kokkos::subview( offsets, n, Kokkos::ALL());
      auto initvar = Kokkos::subview(initialnodes, Kokkos::ALL(), n, Kokkos::ALL());
      parallel_for("cell get init interp",RangePolicy<AssemblyExec>(0,initialnodes.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for(size_type dof=0; dof<initvar.extent(1); dof++ ) {
          initialvals(elem,off(dof)) = initvar(elem,dof);
        }
      });
    }
  }
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> cell::getMass() {
  Kokkos::View<ScalarT***,AssemblyDevice> mass("local mass",numElem,
                                               LIDs.extent(1),
                                               LIDs.extent(1));
  
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    auto cbasis = basis[wkset->usebasis[n]];
    auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
    parallel_for("cell get mass",RangePolicy<AssemblyExec>(0,mass.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for(size_type i=0; i<cbasis.extent(1); i++ ) {
        for(size_type j=0; j<cbasis.extent(1); j++ ) {
          for(size_type k=0; k<cbasis.extent(2); k++ ) {
            mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
          }
        }
      }
    });
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at the integration points given the solution and solve time
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeResponse(const int & seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*responseTimer);
  // Assumes that u has already been filled
  
  // seedwhat indicates what needs to be seeded
  // seedwhat == 0 => seed nothing
  // seedwhat == 1 => seed sol
  // seedwhat == j (j>1) => seed (j-1)-derivative of sol
  
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = cellData->numParamDOF;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  Kokkos::View<AD***,AssemblyDevice> response;
  bool useSensors = false;
  if (cellData->response_type == "pointwise") {
    useSensors = true;
  }
  
  size_t numip = ip.extent(1);
  if (useSensors) {
    numip = sensorLocations.size();
  }
  
  
  if (numip > 0) {
    
    this->updateWorksetBasis();
    
    // Extract the local solution at this time
    // We automatically seed the AD and adjust it below
    Kokkos::View<AD***,AssemblyDevice> u_dof("u_dof",numElem,numDOF.extent(0),LIDs.extent(1)); //(numElem, numVars, numDOF)
    
    parallel_for("cell response get u",RangePolicy<AssemblyExec>(0,u_dof.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) { // numDOF is on device
        for( int i=0; i<numDOF(n); i++ ) {
          u_dof(e,n,i) = AD(maxDerivs,offsets(n,i),u(e,n,i)); // offsets is on device
        }
      }
    });
    
    // Map the local solution to the solution and gradient at ip
    Kokkos::View<AD****,AssemblyDevice> u_ip("u_ip",numElem,numDOF.extent(0),
                                             numip,cellData->dimension);
    Kokkos::View<AD****,AssemblyDevice> ugrad_ip("ugrad_ip",numElem,numDOF.extent(0),
                                                 numip,cellData->dimension);
        
    // Need to rewrite this using useSensors on outside
    if (useSensors) {
      for (size_t ee=0; ee<sensorElem.size(); ee++) {
        int eind = sensorElem[ee];
        for (size_type var=0; var<numDOF.extent(0); var++) {
          DRV cbasis = sensorBasis[ee][wkset->usebasis[var]];
          DRV cbasis_grad = sensorBasisGrad[ee][wkset->usebasis[var]];
          auto u_sv = Kokkos::subview(u_ip, eind, var, ee, Kokkos::ALL());
          auto u_dof_sv = Kokkos::subview(u_dof, eind, var, Kokkos::ALL());
          auto ugrad_sv = Kokkos::subview(ugrad_ip, eind, var, ee, Kokkos::ALL());
          
          parallel_for(RangePolicy<AssemblyExec>(0,cbasis.extent(1)), KOKKOS_LAMBDA (const int dof ) {
            u_sv(0) += u_dof_sv(dof)*cbasis(0,dof,0);
            for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
              ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
            }
          });
        }
      }
    
    }
    else {
      for (size_type var=0; var<numDOF.extent(0); var++) {
        int bnum = wkset->usebasis[var];
        std::string btype = wkset->basis_types[wkset->usebasis[var]];
        if (btype == "HCURL" || btype == "HDIV") {
          auto ref_basis = cellData->ref_basis[wkset->usebasis[var]];
          auto cbasis = basis[wkset->usebasis[var]];
          auto u_sv = Kokkos::subview(u_ip, Kokkos::ALL(), var, Kokkos::ALL(), Kokkos::ALL());
          Kokkos::View<AD***,AssemblyDevice> u_tmp("tmp subview",u_sv.extent(0), u_sv.extent(1), u_sv.extent(2));
          auto u_dof_sv = Kokkos::subview(u_dof, Kokkos::ALL(), var, Kokkos::ALL());
          /*
          parallel_for(RangePolicy<AssemblyExec>(0,u_tmp.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<cbasis.extent(1); i++ ) {
              for (size_type j=0; j<cbasis.extent(2); j++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++) {
                  u_tmp(e,j,s) += u_dof_sv(e,i)*ref_basis(i,j,s);
                }
              }
            }
            
          });
          */
          parallel_for(RangePolicy<AssemblyExec>(0,u_sv.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<cbasis.extent(1); i++ ) {
              for (size_type j=0; j<cbasis.extent(2); j++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++) {
                  u_sv(e,j,s) += u_dof_sv(e,i)*cbasis(e,i,j,s);
                }
              }
            }
          });
          
          
          int numElem = cbasis.extent(0);
          int numb = cbasis.extent(1);
          int numip = cbasis.extent(2);
          int dimension = cbasis.extent(3);
          
          DRV jacobian("jacobian", cbasis.extent(0), cbasis.extent(2),
                       cbasis.extent(3), cbasis.extent(3));
          CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
          DRV jacobianDet("determinant of jacobian", cbasis.extent(0), cbasis.extent(2));
          DRV jacobianInv("inverse of jacobian", cbasis.extent(0),
                          cbasis.extent(2), cbasis.extent(3), cbasis.extent(3));
          CellTools::setJacobianDet(jacobianDet, jacobian);
          CellTools::setJacobianInv(jacobianInv, jacobian);
          
          DRV basis_tmp("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HDIVtransformVALUE(basis_tmp, jacobian, jacobianDet, ref_basis);
          
          DRV basis_tmp2("basis tmp",numElem,numb,numip,dimension);
          OrientTools::modifyBasisByOrientation(basis_tmp2, basis_tmp, orientation,
                                                cellData->basis_pointers[bnum].get());
          
          Kokkos::deep_copy(basis_tmp,0.0);
          OrientTools::modifyBasisByOrientation(basis_tmp, basis_tmp2, orientation,
                                                cellData->basis_pointers[bnum].get());
          
          Kokkos::View<AD***,AssemblyDevice> u_tmp2("tmp subview",u_sv.extent(0), u_sv.extent(1), u_sv.extent(2));
          
          parallel_for(RangePolicy<AssemblyExec>(0,u_sv.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<cbasis.extent(1); i++ ) {
              for (size_type j=0; j<cbasis.extent(2); j++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++) {
                  u_tmp2(e,j,s) += u_dof_sv(e,i)*basis_tmp(e,i,j,s);
                }
              }
            }
          });
          
          Kokkos::View<AD***,AssemblyDevice> u_tmp3("tmp subview",u_sv.extent(0), u_sv.extent(1), u_sv.extent(2));
          //Kokkos::deep_copy(u_tmp3,u_sv);
          //KokkosTools::print(u_tmp3,"u_sv");
          //KokkosTools::print(u_tmp2,"u_tmp2");
          
          //Kokkos::deep_copy(u_tmp,u_sv);
          //DRV JB("basis",numElem,numb,numip,dimension);
          for (int e=0; e<numElem; e++) {
            //for (int i=0; i<numb; i++) {
              for (int j=0; j<numip; j++) {
                for (int d=0; d<dimension; d++) {
                  for (int d2=0; d2<dimension; d2++) {
                    //JB(e,i,j,d) += basis_tmp(e,i,j,d)*jacobian(e,j,d2,d);
                    u_tmp(e,j,d) += u_sv(e,j,d)*jacobianDet(e,j)*jacobianInv(e,j,d,d2);
                    u_tmp3(e,j,d) += u_tmp2(e,j,d)*jacobianDet(e,j)*jacobianInv(e,j,d,d2);
                    //u_sv(e,j,d) *= jacobianDet(e,j);
                    //JB(e,i,j,d) += sbasis(e,i,j,d)*jacobianDet(e,j)*jacobianInv(e,j,d,d2);
                  }
                }
              }
            //}
          }
          
          //KokkosTools::print(u_tmp,"u_tmp");
          //KokkosTools::print(u_tmp3,"u_tmp3");
          
          for (int e=0; e<numElem; e++) {
            for (int j=0; j<numip; j++) {
              for (int d=0; d<dimension; d++) {
                ScalarT fudge = std::pow(-1,d)*std::pow(2.0,dimension-1);
                ScalarT tmp = u_sv(e,j,d).val();
                u_sv(e,j,d) = fudge*u_tmp(e,j,d);
                u_sv(e,j,d).val() = tmp;
              }
            }
          }
          //Kokkos::deep_copy(u_sv,u_tmp);
          /*
          parallel_for(RangePolicy<AssemblyExec>(0,u_sv.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type j=0; j<u_sv.extent(1); j++ ) {
              for (size_type s=0; s<u_sv.extent(2); s++) {
                ScalarT tmp = u_sv(e,j,s).val();
                u_sv(e,j,s) = u_tmp(e,j,s);
                u_sv(e,j,s) += -u_sv(e,j,s).val() + tmp;
              }
            }
          });
           */
        }
        else {
          auto cbasis = basis[wkset->usebasis[var]];
          
          auto u_sv = Kokkos::subview(u_ip, Kokkos::ALL(), var, Kokkos::ALL(), 0);
          auto u_dof_sv = Kokkos::subview(u_dof, Kokkos::ALL(), var, Kokkos::ALL());
          parallel_for(RangePolicy<AssemblyExec>(0,u_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<cbasis.extent(1); i++ ) {
              for (size_type j=0; j<cbasis.extent(2); j++ ) {
                u_sv(e,j) += u_dof_sv(e,i)*cbasis(e,i,j,0);
              }
            }
          });
        }
        
        if (btype == "HGRAD") {
          auto cbasis_grad = basis_grad[wkset->usebasis[var]];
          auto u_dof_sv = Kokkos::subview(u_dof, Kokkos::ALL(), var, Kokkos::ALL());
          auto ugrad_sv = Kokkos::subview(ugrad_ip, Kokkos::ALL(), var, Kokkos::ALL(), Kokkos::ALL());
          parallel_for(RangePolicy<AssemblyExec>(0,u_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<cbasis_grad.extent(1); i++ ) {
              for (size_type j=0; j<cbasis_grad.extent(2); j++ ) {
                for (size_type s=0; s<cbasis_grad.extent(3); s++) {
                  ugrad_sv(e,j,s) += u_dof_sv(e,i)*cbasis_grad(e,i,j,s);
                }
              }
            }
          });
        }
      }
      
    }
        
    // Adjust the AD based on seedwhat
    if (seedwhat == 0) { // remove all seeding
      parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for(size_type j=0; j<u_ip.extent(2); j++ ) {
            for (size_type s=0; s<ugrad_ip.extent(3); s++) {
              u_ip(e,n,j,s) = u_ip(e,n,j,s).val();
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      });
    }
    else if (seedwhat == 1) { // remove seeding on gradient
      parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for(size_type j=0; j<ugrad_ip.extent(2); j++ ) {
            for (size_type s=0; s<ugrad_ip.extent(3); s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      });
      
    }
    else {
      for (int s=0; s<(int)cellData->dimension; s++) {
        auto ugrad_sv = Kokkos::subview(ugrad_ip, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), s);
        if ((seedwhat-2) == s) {
          parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for(size_type j=0; j<ugrad_sv.extent(2); j++ ) {
                ScalarT tmp = ugrad_sv(e,n,j).val();
                ugrad_sv(e,n,j) = u_ip(e,n,j,0);
                ugrad_sv(e,n,j) += -u_ip(e,n,j,0).val() + tmp;
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for(size_type j=0; j<ugrad_sv.extent(2); j++ ) {
                ugrad_sv(e,n,j) = ugrad_sv(e,n,j).val();
              }
            }
          });
        }
       
      }
      parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for(size_type j=0; j<u_ip.extent(2); j++ ) {
            for(size_type s=0; s<u_ip.extent(3); s++ ) {
              u_ip(e,n,j,s) = u_ip(e,n,j,s).val();
            }
          }
        }
      });
    }
    
    //bool seedParams = false;
    //if (seedwhat == 0) {
    //  seedParams = true;
    //}
    
    Kokkos::View<AD****,AssemblyDevice> param_ip;
    Kokkos::View<AD****,AssemblyDevice> paramgrad_ip;
    
    if (numParamDOF.extent(0) > 0) {
      // Extract the local solution at this time
      // We automatically seed the AD and adjust it below
      Kokkos::View<AD***,AssemblyDevice> param_dof("param dof",numElem,numParamDOF.extent(0),paramLIDs.extent(1));
      parallel_for("cell response get p",RangePolicy<AssemblyExec>(0,u_dof.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numParamDOF.extent(0); n++) {
          for(int i=0; i<numParamDOF(n); i++ ) {
            param_dof(e,n,i) = AD(maxDerivs,paramoffsets(n,i),param(e,n,i));
          }
        }
      });
      
      // Map the local solution to the solution and gradient at ip
      param_ip = Kokkos::View<AD****,AssemblyDevice>("u_ip",numElem,numParamDOF.extent(0),
                                                     numip,cellData->dimension);
      paramgrad_ip = Kokkos::View<AD****,AssemblyDevice>("ugrad_ip",numElem,numParamDOF.extent(0),
                                                         numip,cellData->dimension);
      
      if (useSensors) {
        for (size_t ee=0; ee<sensorElem.size(); ee++) {
          int eind = sensorElem[ee];
          for (size_type var=0; var<numParamDOF.extent(0); var++) {
            auto cbasis = param_sensorBasis[ee][wkset->paramusebasis[var]];
            auto cbasis_grad = param_sensorBasisGrad[ee][wkset->paramusebasis[var]];
            auto p_sv = Kokkos::subview(param_ip, eind, var, ee, Kokkos::ALL());
            auto p_dof_sv = Kokkos::subview(param_dof, eind, var, Kokkos::ALL());
            auto pgrad_sv = Kokkos::subview(paramgrad_ip, eind, var, ee, Kokkos::ALL());
            
            parallel_for(RangePolicy<AssemblyExec>(0,cbasis.extent(1)), KOKKOS_LAMBDA (const size_type dof ) {
              p_sv(0) += p_dof_sv(dof)*cbasis(0,dof,0);
              for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
              }
            });
          }
        }
      }
      else {
        for (size_type var=0; var<numParamDOF.extent(0); var++) {
          
          auto cbasis = basis[wkset->paramusebasis[var]];
          auto cbasis_grad = basis_grad[wkset->paramusebasis[var]];
          
          auto p_sv = Kokkos::subview(param_ip, Kokkos::ALL(), var, Kokkos::ALL(), 0);
          auto p_dof_sv = Kokkos::subview(param_dof, Kokkos::ALL(), var, Kokkos::ALL());
          auto pgrad_sv = Kokkos::subview(paramgrad_ip, Kokkos::ALL(), var, Kokkos::ALL(), Kokkos::ALL());
          
          parallel_for(RangePolicy<AssemblyExec>(0,param_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            
            for (size_type i=0; i<cbasis.extent(1); i++ ) {
              for (size_type j=0; j<cbasis.extent(2); j++ ) {
                p_sv(e,j) += p_dof_sv(e,i)*cbasis(e,i,j,0);
                for (size_type s=0; s<cbasis_grad.extent(3); s++) {
                  pgrad_sv(e,j,s) += p_dof_sv(e,i)*cbasis_grad(e,i,j,s);
                }
              }
            }
            
          });
        }
        
      }
      
      
      // Adjust the AD based on seedwhat
      if (seedwhat == 0) { // remove seeding on grad
        parallel_for(RangePolicy<AssemblyExec>(0,paramgrad_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type n=0; n<numParamDOF.extent(0); n++) {
            for(size_type j=0; j<paramgrad_ip.extent(2); j++ ) {
              for (size_type s=0; s<paramgrad_ip.extent(3); s++) {
                paramgrad_ip(e,n,j,s) = paramgrad_ip(e,n,j,s).val();
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,param_ip.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type n=0; n<numParamDOF.extent(0); n++) {
            for(size_type j=0; j<param_ip.extent(2); j++ ) {
              param_ip(e,n,j,0) = param_ip(e,n,j,0).val();
              for (size_type s=0; s<paramgrad_ip.extent(3); s++) {
                paramgrad_ip(e,n,j,s) = paramgrad_ip(e,n,j,s).val();
              }
            }
          }
        });
      }
    }
    
    if (useSensors) {
      if (sensorLocations.size() > 0){
        response = cellData->physics_RCP->getPointResponse(cellData->myBlock, u_ip, ugrad_ip, param_ip,
                                                           paramgrad_ip, sensorPoints,
                                                           wkset->time, wkset);
      }
    }
    else {
      response = cellData->physics_RCP->getResponse(cellData->myBlock, u_ip, ugrad_ip, param_ip,
                                                    paramgrad_ip, ip,
                                                    wkset->time, wkset);
    }
  }
  
  return response;
  
}


///////////////////////////////////////////////////////////////////////////////////////
// Compute the objective function given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD**,AssemblyDevice> cell::computeObjective(const ScalarT & solvetime,
                                                         const size_t & tindex,
                                                         const int & seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*objectiveTimer);
  // assumes the params have been seeded elsewhere (solver, postprocess interfaces)
  Kokkos::View<AD**,AssemblyDevice> objective;
  auto cwts = wts;
  
  if (!(cellData->multiscale) || cellData->mortar_objective) {
    
    
    if (cellData->response_type == "pointwise") { // uses sensor data
    
      //Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(solvetime,tindex,seedwhat);
      Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(seedwhat);
      
      ScalarT TOL = 1.0e-6; // tolerance for comparing sensor times and simulation times
      objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,numSensors);
      
      if (numSensors > 0) { // if this element has any sensors
        for (size_t s=0; s<numSensors; s++) {
          bool foundtime = false;
          size_t ftime;
          
          for (size_type t2=0; t2<sensorData[s].extent(0); t2++) {
            ScalarT stime = sensorData[s](t2,0);
            if (abs(stime-solvetime) < TOL) {
              foundtime = true;
              ftime = t2;
            }
          }
          
          if (foundtime) {
            int ee = sensorElem[s];
            for (size_type r=0; r<responsevals.extent(1); r++) {
              AD rval = responsevals(ee,r,s);
              ScalarT sval = sensorData[s](ftime,r+1);
              if(cellData->compute_diff) {
                objective(ee,s) += 0.5*wkset->deltat*(rval-sval) * (rval-sval);
              }
              else {
                objective(ee,s) += wkset->deltat*rval;
              }
            }
          }
        }
      }
      
    }
    else if (cellData->response_type == "global") { // uses physicsmodules->target
      
      //Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(solvetime,tindex,seedwhat);
      Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(seedwhat);
      
      objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,ip.extent(1));
      Kokkos::deep_copy(wkset->ip,ip);
      Kokkos::View<AD***,AssemblyDevice> targ = computeTarget(solvetime);
      Kokkos::View<AD***,AssemblyDevice> weight = computeWeight(solvetime);
      
      for (size_type r=0; r<responsevals.extent(1); r++) {
        auto cresp = Kokkos::subview(responsevals,Kokkos::ALL(),r,Kokkos::ALL());
        auto ctarg = Kokkos::subview(targ,Kokkos::ALL(),r,Kokkos::ALL());
        auto cweight = Kokkos::subview(weight,Kokkos::ALL(),r,Kokkos::ALL());
        auto dt = wkset->deltat_KV;
        
        if(cellData->compute_diff) {
          parallel_for("cell objective",RangePolicy<AssemblyExec>(0,cresp.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type pt=0; pt<cresp.extent(1); pt++) {
              AD diff = cresp(elem,pt)-ctarg(elem,pt);
              objective(elem,pt) += 0.5*dt(0)*cweight(elem,pt)*(diff)*(diff)*cwts(elem,pt);
            }
          });
        }
        else {
          parallel_for("cell objective",RangePolicy<AssemblyExec>(0,cresp.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
            for (size_t pt=0; pt<cresp.extent(1); pt++) {
              objective(elem,pt) += dt(0)*cresp(elem,pt)*cwts(elem,pt);
            }
          });
        }
      }
    }
    
  }
  else {
    
    int sgindex = subgrid_model_index[tindex];
    Kokkos::View<AD*,AssemblyDevice> cobj = subgridModels[sgindex]->computeObjective(cellData->response_type,seedwhat,
                                                                                     solvetime,subgrid_usernum);
    
    objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,cobj.extent(0));
    parallel_for("cell objective",RangePolicy<AssemblyExec>(0,cobj.extent(0)), KOKKOS_LAMBDA (const size_type i ) {
      objective(0,i) += cobj(i); // TMW: tempory fix
    });
  }
  
  return objective;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the regularization over the domain given the domain discretized parameters
///////////////////////////////////////////////////////////////////////////////////////

AD cell::computeDomainRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                     const vector<int> reg_indices) {
  
  //AD reg;
  
  //bool seedParams = true;
  //int numip = wkset->numip;
  this->updateWorksetBasis();
  wkset->computeParamVolIP(param, 3);
  
  Kokkos::View<AD[1],AssemblyDevice> adscratch("scratch for AD");
  auto adscratch_host = Kokkos::create_mirror_view(adscratch);
  
  Kokkos::View<int[2],AssemblyDevice> iscratch("scratch for ints");
  auto iscratch_host = Kokkos::create_mirror_view(iscratch);
  
  Kokkos::View<ScalarT[2],AssemblyDevice> dscratch("scratch for ScalarT");
  auto dscratch_host = Kokkos::create_mirror_view(dscratch);
  
  auto cwts = wts;
  
  int numParams = reg_indices.size();
  ScalarT reg_offset = 1.0e-5;
  Kokkos::View<AD****,AssemblyDevice> par = wkset->local_param;
  Kokkos::View<AD****,AssemblyDevice> par_grad = wkset->local_param_grad;
  for (int i = 0; i < numParams; i++) {
    dscratch_host(0) = reg_constants[i];
    dscratch_host(1) = reg_offset;
    iscratch_host(0) = reg_types[i];
    iscratch_host(1) = reg_indices[i];
    Kokkos::deep_copy(dscratch,dscratch_host);
    Kokkos::deep_copy(iscratch,iscratch_host);
    parallel_for("cell domain reg",RangePolicy<AssemblyExec>(0,par.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      int pindex = iscratch(1);
      int rtype = iscratch(0);
      ScalarT reg_const = dscratch(0);
      ScalarT reg_off = dscratch(1);
      for (size_type k = 0; k < par.extent(2); k++) {
        AD p = par(e,pindex,k,0);
        // L2
        if (rtype == 0) {
          adscratch(0) += 0.5*reg_const*p*p*cwts(e,k);
        }
        else {
          AD dpdx = par_grad(e,pindex,k,0);
          AD dpdy = 0.0;
          AD dpdz = 0.0;
          if (par_grad.extent(3) > 1)
            dpdy = par_grad(e,pindex,k,1);
          if (par_grad.extent(3) > 2)
            dpdz = par_grad(e,pindex,k,2);
          // H1
          if (rtype == 1) {
            adscratch(0) += 0.5*reg_const*(dpdx*dpdx + dpdy*dpdy + dpdz*dpdz)*cwts(e,k);
          }
          // TV
          else if (rtype == 2) {
            adscratch(0) += reg_const*sqrt(dpdx*dpdx + dpdy*dpdy + dpdz*dpdz + reg_off*reg_off)*cwts(e,k);
          }
        }
      }
    });
  }
  Kokkos::deep_copy(adscratch_host,adscratch);
  return adscratch_host(0);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the target at the integration points given the solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeTarget(const ScalarT & solvetime) {
  return cellData->physics_RCP->target(cellData->myBlock, ip, solvetime, wkset);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the weighting function at the integration points given the solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeWeight(const ScalarT & solvetime) {
  return cellData->physics_RCP->weight(cellData->myBlock, ip, solvetime, wkset);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at the integration points given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

void cell::addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                      const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                      Teuchos::RCP<discretization> & disc,
                      const vector<basis_RCP> & basis_pointers,
                      const vector<basis_RCP> & param_basis_pointers) {
  
  
  // If we have sensors, then we set the response type to pointwise
  cellData->response_type = "pointwise";
  useSensors = true;
  bool useFineScale = true;
  if (!(cellData->multiscale) || cellData->mortar_objective) {
    useFineScale = false;
  }
  
  if (cellData->exodus_sensors) {
    // don't use sensor_points
    // set sensorData and sensorLocations from exodus file
    if (sensorLocations.size() > 0) {
      sensorPoints = DRV("sensorPoints",1,sensorLocations.size(),cellData->dimension);
      for (size_t i=0; i<sensorLocations.size(); i++) {
        for (size_t j=0; j<cellData->dimension; j++) {
          sensorPoints(0,i,j) = sensorLocations[i](0,j);
        }
        sensorElem.push_back(0);
      }
      DRV refsenspts_buffer("refsenspts_buffer",1,sensorLocations.size(),cellData->dimension);
      Intrepid2::CellTools<PHX::Device>::mapToReferenceFrame(refsenspts_buffer, sensorPoints, nodes, *(cellData->cellTopo));
      DRV refsenspts("refsenspts",sensorLocations.size(),cellData->dimension);
      Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
      
      vector<DRV> csensorBasis;
      vector<DRV> csensorBasisGrad;
      
      for (size_t b=0; b<basis_pointers.size(); b++) {
        csensorBasis.push_back(disc->evaluateBasis(basis_pointers[b], refsenspts, orientation));
        csensorBasisGrad.push_back(disc->evaluateBasisGrads(basis_pointers[b], nodes, refsenspts,
                                                            cellData->cellTopo, orientation));
      }
      
      sensorBasis.push_back(csensorBasis);
      sensorBasisGrad.push_back(csensorBasisGrad);
      
      
      vector<DRV> cpsensorBasis;
      vector<DRV> cpsensorBasisGrad;
      
      for (size_t b=0; b<param_basis_pointers.size(); b++) {
        cpsensorBasis.push_back(disc->evaluateBasis(param_basis_pointers[b], refsenspts, orientation));
        cpsensorBasisGrad.push_back(disc->evaluateBasisGrads(param_basis_pointers[b], nodes,
                                                             refsenspts, cellData->cellTopo, orientation));
      }
      
      param_sensorBasis.push_back(cpsensorBasis);
      param_sensorBasisGrad.push_back(cpsensorBasisGrad);
    }
    
  }
  else {
    if (useFineScale) {
      
      for (size_t i=0; i<subgridModels.size(); i++) {
        //if (subgrid_model_index[0] == i) {
        subgridModels[i]->addSensors(sensor_points,sensor_loc_tol,sensor_data,have_sensor_data,
                                     basis_pointers, subgrid_usernum);
        //}
      }
      
    }
    else {
      DRV phys_points("phys_points",1,sensor_points.extent(0),cellData->dimension);
      for (size_t i=0; i<sensor_points.extent(0); i++) {
        for (size_t j=0; j<cellData->dimension; j++) {
          phys_points(0,i,j) = sensor_points(i,j);
        }
      }
      
      if (!(cellData->loadSensorFiles)) {
        for (size_t e=0; e<numElem; e++) {
          
          DRV refpts("refpts", 1, sensor_points.extent(0), sensor_points.extent(1));
          Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell", 1, sensor_points.extent(0));
          DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
          for (size_type i=0; i<nodes.extent(1); i++) {
            for (size_type j=0; j<nodes.extent(2); j++) {
              cnodes(0,i,j) = nodes(e,i,j);
            }
          }
          CellTools::mapToReferenceFrame(refpts, phys_points, cnodes, *(cellData->cellTopo));
          CellTools::checkPointwiseInclusion(inRefCell, refpts, *(cellData->cellTopo), sensor_loc_tol);
          
          for (size_type i=0; i<sensor_points.extent(0); i++) {
            if (inRefCell(0,i) == 1) {
              
              Kokkos::View<ScalarT**,HostDevice> newsenspt("new sensor point",1,cellData->dimension);
              for (size_t j=0; j<cellData->dimension; j++) {
                newsenspt(0,j) = sensor_points(i,j);
              }
              sensorLocations.push_back(newsenspt);
              mySensorIDs.push_back(i);
              sensorElem.push_back(e);
              if (have_sensor_data) {
                sensorData.push_back(sensor_data[i]);
              }
              if (cellData->writeSensorFiles) {
                std::stringstream ss;
                ss << localElemID(e);
                string str = ss.str();
                string fname = "sdat." + str + ".dat";
                std::ofstream outfile(fname.c_str());
                outfile.precision(8);
                outfile << i << "  ";
                outfile << sensor_points(i,0) << "  " << sensor_points(i,1) << "  ";
                //outfile << sensor_data[i](0,0) << "  " << sensor_data[i](0,1) << "  " << sensor_data[i](0,2) << "  " ;
                outfile << endl;
                outfile.close();
              }
            }
          }
        }
      }
      
      if (cellData->loadSensorFiles) {
        for (size_t e=0; e<numElem; e++) {
          std::stringstream ss;
          ss << localElemID(e);
          string str = ss.str();
          std::ifstream sfile;
          sfile.open("sensorLocations/sdat." + str + ".dat");
          int cID;
          //ScalarT l1, l2, t1, d1, d2;
          ScalarT l1, l2;
          sfile >> cID;
          sfile >> l1;
          sfile >> l2;
          
          sfile.close();
          
          Kokkos::View<ScalarT**,HostDevice> newsenspt("sensor point",1,cellData->dimension);
          //FC newsensdat(1,3);
          newsenspt(0,0) = l1;
          newsenspt(0,1) = l2;
          sensorLocations.push_back(newsenspt);
          mySensorIDs.push_back(cID);
          sensorElem.push_back(e);
        }
      }
      
      numSensors = sensorLocations.size();
      
      // Evaluate the basis functions and derivatives at sensor points
      if (numSensors > 0) {
        sensorPoints = DRV("sensorPoints",numElem,numSensors,cellData->dimension);
        
        for (size_t i=0; i<numSensors; i++) {
          
          DRV csensorPoints("sensorPoints",1,1,cellData->dimension);
          DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
          for (size_t j=0; j<cellData->dimension; j++) {
            csensorPoints(0,0,j) = sensorLocations[i](0,j);
            sensorPoints(0,i,j) = sensorLocations[i](0,j);
            for (size_type k=0; k<nodes.extent(1); k++) {
              cnodes(0,k,j) = nodes(sensorElem[i],k,j);
            }
          }
          
          
          DRV refsenspts_buffer("refsenspts_buffer",1,1,cellData->dimension);
          DRV refsenspts("refsenspts",1,cellData->dimension);
          
          CellTools::mapToReferenceFrame(refsenspts_buffer, csensorPoints, cnodes, *(cellData->cellTopo));
          //CellTools<AssemblyDevice>::mapToReferenceFrame(refsenspts, csensorPoints, cnodes, *cellTopo);
          Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
          
          vector<DRV> csensorBasis;
          vector<DRV> csensorBasisGrad;
          
          for (size_t b=0; b<basis_pointers.size(); b++) {
            csensorBasis.push_back(disc->evaluateBasis(basis_pointers[b], refsenspts, orientation));
            csensorBasisGrad.push_back(disc->evaluateBasisGrads(basis_pointers[b], cnodes,
                                                                refsenspts, cellData->cellTopo, orientation));
          }
          sensorBasis.push_back(csensorBasis);
          sensorBasisGrad.push_back(csensorBasisGrad);
          
          
          vector<DRV> cpsensorBasis;
          vector<DRV> cpsensorBasisGrad;
          
          for (size_t b=0; b<param_basis_pointers.size(); b++) {
            cpsensorBasis.push_back(disc->evaluateBasis(param_basis_pointers[b], refsenspts, orientation));
            cpsensorBasisGrad.push_back(disc->evaluateBasisGrads(param_basis_pointers[b], nodes,
                                                                 refsenspts, cellData->cellTopo, orientation));
          }
          
          param_sensorBasis.push_back(cpsensorBasis);
          param_sensorBasisGrad.push_back(cpsensorBasisGrad);
        }
        
      }
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Subgrid Plotting
///////////////////////////////////////////////////////////////////////////////////////

void cell::writeSubgridSolution(const std::string & filename) {
  //if (multiscale) {
  //  subgridModel->writeSolution(filename, subgrid_usernum);
  //}
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the subgrid model
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models) {
  
  /*
   wkset->update(ip,jacobian);
   int newmodel = udfunc->getSubgridModel(nodes, wkset, models.size());
   if (newmodel != subgrid_model_index) {
   // then we need:
   // 1. To add the macro-element to the new model
   // 2. Project the most recent solutions onto the new model grid
   // 3. Update this cell to use the new model
   
   // Step 1:
   int newusernum = models[newmodel]->addMacro(nodes, sideinfo, sidenames,
   GIDs, index);
   
   // Step 2:
   
   // Step 3:
   subgridModel = models[newmodel];
   subgrid_model_index = newmodel;
   subgrid_usernum = newusernum;
   
   
   }*/
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateData() {
  
  // hard coded for what I need it for right now
  if (cellData->have_cell_phi) {
    wkset->have_rotation_phi = true;
    wkset->rotation_phi = cell_data;
  }
  else if (cellData->have_cell_rotation) {
    wkset->have_rotation = true;
    Kokkos::View<ScalarT***,AssemblyDevice> rot = wkset->rotation;
    parallel_for("cell update data", RangePolicy<AssemblyExec>(0,cell_data.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
      rot(e,0,0) = cell_data(e,0);
      rot(e,0,1) = cell_data(e,1);
      rot(e,0,2) = cell_data(e,2);
      rot(e,1,0) = cell_data(e,3);
      rot(e,1,1) = cell_data(e,4);
      rot(e,1,2) = cell_data(e,5);
      rot(e,2,0) = cell_data(e,6);
      rot(e,2,1) = cell_data(e,7);
      rot(e,2,2) = cell_data(e,8);
    });
    /*
     for (int e=0; e<numElem; e++) {
     rotmat(e,0,0) = cell_data(e,0);
     rotmat(e,0,1) = cell_data(e,1);
     rotmat(e,0,2) = cell_data(e,2);
     rotmat(e,1,0) = cell_data(e,3);
     rotmat(e,1,1) = cell_data(e,4);
     rotmat(e,1,2) = cell_data(e,5);
     rotmat(e,2,0) = cell_data(e,6);
     rotmat(e,2,1) = cell_data(e,7);
     rotmat(e,2,2) = cell_data(e,8);
     }*/
    //wkset->rotation = rotmat;
  }
  else if (cellData->have_extra_data) {
    wkset->extra_data = cell_data;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetAdjPrev(const ScalarT & val) {
  Kokkos::deep_copy(adj_prev,val);
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the solution at the nodes
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> cell::getSolutionAtNodes(const int & var) {
  
  Teuchos::TimeMonitor nodesoltimer(*computeNodeSolTimer);
  
  int bnum = wkset->usebasis[var];
  auto cbasis = basis_nodes[bnum];
  Kokkos::View<ScalarT***,AssemblyDevice> nodesol("solution at nodes",
                                                  cbasis.extent(0), cbasis.extent(2), cellData->dimension);
  auto uvals = Kokkos::subview(u,Kokkos::ALL(), var, Kokkos::ALL());
  parallel_for("cell node sol",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
      ScalarT uval = uvals(elem,dof);
      for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
        for (size_type s=0; s<cbasis.extent(3); s++ ) {
          nodesol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
        }
      }
    }
  });
  
  return nodesol;
  
}
