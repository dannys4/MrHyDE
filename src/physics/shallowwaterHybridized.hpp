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

/** @file shallowwaterHybridized.hpp
 *
 * @brief Shallow water physics module, hybridized version
 *
 * Solves the shallow water equations with a hybridized formulation.
 * See Samii (J. Sci. Comp. 2019). 
 */

#ifndef MRHYDE_SHALLOWWATERHYBRIDIZED_H
#define MRHYDE_SHALLOWWATERHYBRIDIZED_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /** Shallow water physics module, hybridized version
   *
   * Solves the shallow water equations with a hybridized formulation.
   * See Samii (J. Sci. Comp. 2019). 
   * 
   * @note If the user intends for the bottom boundary \f$b(x,y)\f$ to vary spatially, this should
   * be implement through the sources in the \f$Hu_x\f$ and \f$Hu_y\f$ equations with terms
   * \f$-gH\frac{\partial b}{\partial x}\f$ and \f$-gH\frac{\partial b}{\partial y}\f$, respectively.
   */
  
  class shallowwaterHybridized : public physicsbase {
  public:

    shallowwaterHybridized() {};
    
    ~shallowwaterHybridized() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    shallowwaterHybridized(Teuchos::ParameterList & settings, const int & dimension);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void boundaryResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);

    /* @brief Update the fluxes for the residual calculation.
     *
     * @param[in] on_side  Bool indicating if we are on an element side or not
     *
     * @details When we are at an interface, the flux is evaluated using the trace variables.
     */

    void computeFluxVector(const bool & on_side);

    /* @brief Update the stabilization term for numerical flux at interfaces.
     *
     * @details When we are at an interface, the flux 
     * \f$\hat{F}(\mathbf{S}) \cdot \mathbf{n} = 
     * F(\hat{\mathbf{S}}) \cdot \mathbf{n} + Stab(\hat{\mathbf{S}},\mathbf{S}) 
     * \times (\hat{\mathbf{S}} - \mathbf{S})\f$ where \f$Stab(\hat{\mathbf{S}},\mathbf{S})\f$
     * is the local stabilization matrix. We store the vector result of applying the matrix to
     * the trace variable discrepancy \f$(\hat{\mathbf{S}} - \mathbf{S})\f$.
     */

    void computeStabilizationTerm();

    /* @brief Update the boundary flux at the domain boundary.
     *
     * @details When we are at a domain boundary, the flux \f$\hat{B}(\mathbf{S}) = B(\hat{\mathbf{S}})\f$
     * is used to weakly enforce the boundary condition in the computeFlux() routine.
     */

    void computeBoundaryTerm();

    /* @brief Computes the local eigenvalue decomposition for the stabilization and boundary terms.
     * This is the 1-D version.
     *
     * @param[inout] leftEV  Storage for the left eigenvectors
     * @param[inout] Lambda  Storage for the eigenvalues. A vector, not a matrix since it is diagonal.
     * @param[inout] rightEV  Storage for the right eigenvectors
     * @param[in] Hux  x-component of the depth-weighted velocity
     * @param[in] H  Depth
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void eigendecompFluxJacobian(View_AD2 leftEV, View_AD1 Lambda, View_AD2 rightEV, 
        const AD & Hux, const AD & H);

    /* @brief Computes the local eigenvalue decomposition for the stabilization and boundary terms.
     * This is the 2-D version.
     *
     * @param[inout] leftEV  Storage for the left eigenvectors
     * @param[inout] Lambda  Storage for the eigenvalues. A vector, not a matrix since it is diagonal.
     * @param[inout] rightEV  Storage for the right eigenvectors
     * @param[in] Hux  x-component of the depth-weighted velocity 
     * @param[in] Huy  y-component of the depth-weighted velocity 
     * @param[in] H  Density
     * @param[in] nx  x-component of the normal vector
     * @param[in] ny  y-component of the normal vector
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void eigendecompFluxJacobian(View_AD2 leftEV, View_AD1 Lambda, View_AD2 rightEV, 
        const AD & rhoux, const AD & rhouy, const AD & rho, const ScalarT & nx, const ScalarT & ny);

    /* @brief Computes y = Ax
     *
     * @param[in] A  Matrix
     * @param[in] x  Vector
     * @param[out] y  Result
     *
     */

    template<class V1, class V2, class V3>
    KOKKOS_FUNCTION void matVec(const V1 A, const V2 x, V3 y) {

      // TODO error checking for size
    
      size_type n = A.extent(0);  // should be square and x and y should be of length n
    
      for (size_type i=0; i<n; ++i) {
        y(i) = 0.; // zero out just in case
        for (size_type j=0; j<n; ++j) {
          y(i) += A(i,j) * x(j);
        }
      }
    }

// TODO This needs to be handled in a better way, temporary!
#ifndef MrHyDE_UNITTEST_HIDE_PRIVATE_VARS
  private:
#endif

    int spaceDim;
    
    int H_num, Hux_num, Huy_num;
    int auxH_num, auxHux_num, auxHuy_num;

    // indices to access the model params
    int gravity_mp_num = 0;

    bool maxEVstab,roestab; // Options for stabilization

    View_AD4 fluxes_vol, fluxes_side; // Storage for the fluxes
    View_AD3 stab_bound_side; // Storage for the stabilization term/boundary term

    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> fluxVectorFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeFluxVector() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxVectorFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeFluxVector() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> stabCompFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeStabilizationTerm() - function evaluation");
    Teuchos::RCP<Teuchos::Time> stabCompFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeStabilizationTerm() - evaluation of product");
    Teuchos::RCP<Teuchos::Time> boundCompFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeBoundaryTerm() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundCompFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::shallowwaterHybridized::computeBoundaryTerm() - evaluation of product");

  };
  
}

#endif
