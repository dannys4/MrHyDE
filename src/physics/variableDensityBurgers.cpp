/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "variableDensityBurgers.hpp"

using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

template<class EvalT>
VariableDensityBurgers<EvalT>::VariableDensityBurgers(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  TEUCHOS_TEST_FOR_EXCEPTION(dimension_ != 1,std::runtime_error,"Error: variable density Burgers is only implemented for 1D problems.");
  std::cerr << "Constructing VDBurgers, " << dimension_ << "D" << std::endl;
  
  label = "VariableDensityBurgers";
  myvars.push_back("rhou");
  mybasistypes.push_back("HGRAD");
  myvars.push_back("rho");
  mybasistypes.push_back("HGRAD");
  myvars.push_back("F");
  mybasistypes.push_back("HGRAD");
  std::cerr << "VDBurgers constructed" << std::endl;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VariableDensityBurgers<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  std::cerr << "Defining functions for VDBurgers" << std::endl;
  functionManager = functionManager_;
  functionManager->addFunction("alpha",fs.get<string>("alpha","1e-5"),"ip");
  std::cerr << "Functions defined for VDBurgers" << std::endl;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VariableDensityBurgers<EvalT>::volumeResidual() {
  std::cerr << "VDBurgers: calculating volume residual" << std::endl;
  using namespace std;
  
  // Evaluate the functions we always need
  auto alpha = functionManager->evaluate("alpha","ip");
  
  auto rhou = wkset->getSolutionField("rhou");
  auto rho = wkset->getSolutionField("rho");
  auto F = wkset->getSolutionField("F");
  auto drhoudx = wkset->getSolutionField("grad(rhou)[x]");
  auto drhodx = wkset->getSolutionField("grad(rho)[x]");
  auto dFdx = wkset->getSolutionField("grad(F)[x]");
  auto dudt = wkset->getSolutionField("rhou_t");
  auto drhodt = wkset->getSolutionField("rho_t");

  { // Solves d/dt (rhou) + d/dx (rhou^2/rho + F) = 0
    // Get some information from the workset
    auto basis = wkset->getBasis("rhou");
    auto basis_grad = wkset->getBasisGrad("rhou");
    auto res = wkset->res;
    auto wts = wkset->wts;
    auto off = wkset->getOffsets("rhou");

    parallel_for("VariableDensityBurgers volume resid u",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT rhou_sq = rhou(elem,pt)*rhou(elem,pt)/rho(elem,pt);
        EvalT reg = F(elem,pt);
        EvalT f = dudt(elem,pt)*wts(elem,pt);
        EvalT Fx = -(rhou_sq + reg)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }

  { // Solves drhodt + drhodx u + rho dudx = 0
    // Get some information from the workset
    auto basis = wkset->getBasis("rho");
    auto basis_grad = wkset->getBasisGrad("rho");
    auto res = wkset->res;
    auto wts = wkset->wts;
    auto off = wkset->getOffsets("rho");

    parallel_for("VariableDensityBurgers volume resid rho",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT f = drhodt(elem,pt)*wts(elem,pt);
        EvalT Fx = -rhou(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }

  { // Solves rho^{-1} F - alpha*d/dx (rho^{-1} dFdx) - 2*alpha*(dudx)^2 = 0
    // Get some information from the workset
    auto basis = wkset->getBasis("F");
    auto basis_grad = wkset->getBasisGrad("F");
    auto res = wkset->res;
    auto wts = wkset->wts;
    auto off = wkset->getOffsets("F");

    parallel_for("VariableDensityBurgers volume resid F",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT dudx = (drhoudx(elem,pt) - rhou(elem,pt)*drhodx(elem,pt)/rho(elem,pt))/rho(elem,pt);
        EvalT state = F(elem,pt)/rho(elem,pt) - 2*alpha(elem,pt)*dudx*dudx;
        EvalT f = state*wts(elem,pt);
        EvalT Fx = alpha(elem,pt)*(dFdx(elem,pt)/rho(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }
  std::cerr << "VDBurgers: finished calculating volume residual" << std::endl;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VariableDensityBurgers<EvalT>::boundaryResidual() {
  
  // auto bcs = wkset->var_bcs;
  // int cside = wkset->currentside;
  
  //Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  // int dim = wkset->dimension;
  
  
  // Currently unimplemented
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::VariableDensityBurgers<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::VariableDensityBurgers<AD>;

// Standard built-in types
template class MrHyDE::VariableDensityBurgers<AD2>;
template class MrHyDE::VariableDensityBurgers<AD4>;
template class MrHyDE::VariableDensityBurgers<AD8>;
template class MrHyDE::VariableDensityBurgers<AD16>;
template class MrHyDE::VariableDensityBurgers<AD18>;
template class MrHyDE::VariableDensityBurgers<AD24>;
template class MrHyDE::VariableDensityBurgers<AD32>;
#endif
