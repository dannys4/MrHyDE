/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHDYE_VARIABLEDENSITYBURGERS_H
#define MRHDYE_VARIABLEDENSITYBURGERS_H

#include "managers/functionManager.hpp"
#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief VariableDensityBurgers' physics class.
   *
   * This class computes volumetric residuals for the physics described by the following strong form:
   * \f{eqnarray*}
   *   \frac{\partial u}{\partial t}
   *   +
   *   \frac{\partial}{\partial x} (\rho u^2 + F)
   *   &= 0 \\
   *  %
   *  \frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x} (\rho u) &= 0\\
   *  %
   *  \rho^{-1} F
   *  -
   *  \alpha \frac{\partial }{\partial x}(\rho^{-1}\frac{\partial F}{\partial x})
   *  -
   *  2\alpha(\frac{\partial}{\partial x} u)^2
   *  &= 0
   * \f}
   * Where the unknown \f$u\f$ is the fluid velocity, \f$\rho\f$ is the fluid density
   * and \f$F\f$ is an information geometric regularization term (Cao, Schaefer 2023).
   * The following functions may be specified in the input.yaml file:
   *   - "alpha" is the size of the regularization
   */

  template<class EvalT>
  class VariableDensityBurgers : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    VariableDensityBurgers() {} ;
    
    ~VariableDensityBurgers() {};
    
    // ========================================================================================
    // ========================================================================================
    
    VariableDensityBurgers(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    void boundaryResidual();
  
  private:
  
  };
  
}

#endif // MRHDYE_VARIABLEDENSITYBURGERS_H