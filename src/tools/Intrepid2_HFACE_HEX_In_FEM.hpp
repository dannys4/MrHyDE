// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid2_HFACE_HEX_In_FEM.hpp
 \brief  Header file for the Intrepid2::Basis_HFACE_HEX_In_FEM class.
 \author Created by T. Wildey based on implementation by R. Kirby and P. Bochev and D. Ridzal and K. Petrson.
 Kokkorized by Kyungjoo Kim
 */

#ifndef __INTREPID2_HFACE_HEX_IN_FEM_HPP__
#define __INTREPID2_HFACE_HEX_IN_FEM_HPP__

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"

namespace Intrepid2 {
  
  namespace Impl {
    
    /**
     \brief See Intrepid2::Basis_HFACE_HEX_In_FEM
     */
    class Basis_HFACE_HEX_In_FEM {
    public:
      typedef struct Quadrilateral<4> cell_topology_type;
      /**
       \brief See Intrepid2::Basis_HFACE_HEX_In_FEM
       */
      template<EOperator opType>
      struct Serial {
        template<typename outputValueViewType,
        typename inputPointViewType,
        typename workViewType,
        typename vinvViewType>
        KOKKOS_INLINE_FUNCTION
        static void
        getValues(       outputValueViewType outputValues,
                  const inputPointViewType  inputPoints,
                  workViewType        work,
                  const vinvViewType        vinvLine);
        
        KOKKOS_INLINE_FUNCTION
        static ordinal_type
        getWorkSizePerPoint(ordinal_type order) {
          return 4*getPnCardinality<1>(order);
        }
      };
      
      template<typename DeviceType, ordinal_type numPtsPerEval,
      typename outputValueValueType, class ...outputValueProperties,
      typename inputPointValueType,  class ...inputPointProperties,
      typename vinvValueType,        class ...vinvProperties>
      static void
      getValues(        Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                const Kokkos::DynRankView<vinvValueType,       vinvProperties...>        vinvLine,
                const EOperator operatorType );
      
      /**
       \brief See Intrepid2::Basis_HFACE_HEX_In_FEM
       */
      template<typename outputValueViewType,
      typename inputPointViewType,
      typename vinvViewType,
      typename workViewType,
      EOperator opType,
      ordinal_type numPtsEval>
      struct Functor {
        outputValueViewType _outputValues;
        const inputPointViewType  _inputPoints;
        const vinvViewType        _vinvLine;
        workViewType        _work;
        
        KOKKOS_INLINE_FUNCTION
        Functor( outputValueViewType outputValues_,
                inputPointViewType  inputPoints_,
                vinvViewType        vinvLine_,
                workViewType        work_)
        : _outputValues(outputValues_), _inputPoints(inputPoints_),
        _vinvLine(vinvLine_), _work(work_) {}
        
        KOKKOS_INLINE_FUNCTION
        void operator()(const size_type iter) const {
          const auto ptBegin = Util<ordinal_type>::min(iter*numPtsEval,    _inputPoints.extent(0));
          const auto ptEnd   = Util<ordinal_type>::min(ptBegin+numPtsEval, _inputPoints.extent(0));
          
          const auto ptRange = Kokkos::pair<ordinal_type,ordinal_type>(ptBegin, ptEnd);
          const auto input   = Kokkos::subview( _inputPoints, ptRange, Kokkos::ALL() );
          
          typename workViewType::pointer_type ptr = _work.data() + _work.extent(0)*ptBegin*get_dimension_scalar(_work);
          
          auto vcprop = Kokkos::common_view_alloc_prop(_work);
          workViewType  work(Kokkos::view_wrap(ptr,vcprop), (ptEnd-ptBegin)*_work.extent(0));
          
          switch (opType) {
            case OPERATOR_VALUE : {
              auto output = Kokkos::subview( _outputValues, Kokkos::ALL(), ptRange, Kokkos::ALL() );
              Serial<opType>::getValues( output, input, work, _vinvLine);
              break;
            }
            case OPERATOR_GRAD : {
              // probably won't be needed
            }
            default: {
              INTREPID2_TEST_FOR_ABORT( true,
                                       ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::Functor) operator is not supported");
              
            }
          }
        }
      };
    };
  }
  
  /** \class  Intrepid2::Basis_HFACE_HEX_In_FEM
   \brief  Implementation of the default H(face)-compatible FEM basis on Quadrilateral cell
   */
  
  template<typename DeviceType = void,
  typename outputValueType = double,
  typename pointValueType = double>
  class Basis_HFACE_HEX_In_FEM
  : public Basis<DeviceType,outputValueType,pointValueType> {
  public:
    using OrdinalTypeArray1DHost = typename Basis<DeviceType,outputValueType,pointValueType>::OrdinalTypeArray1DHost;
    using OrdinalTypeArray2DHost = typename Basis<DeviceType,outputValueType,pointValueType>::OrdinalTypeArray2DHost;
    using OrdinalTypeArray3DHost = typename Basis<DeviceType,outputValueType,pointValueType>::OrdinalTypeArray3DHost;
    
    /** \brief  Constructor.
     */
    Basis_HFACE_HEX_In_FEM(const ordinal_type order,
                           const EPointType   pointType = POINTTYPE_EQUISPACED);
    
    using OutputViewType = typename Basis<DeviceType,outputValueType,pointValueType>::OutputViewType;
    using PointViewType  = typename Basis<DeviceType,outputValueType,pointValueType>::PointViewType;
    using ScalarViewType = typename Basis<DeviceType,outputValueType,pointValueType>::ScalarViewType;
    
    using Basis<DeviceType,outputValueType,pointValueType>::getValues;
    
    virtual
    void
    getValues(       OutputViewType outputValues,
              const PointViewType  inputPoints,
              const EOperator operatorType = OPERATOR_VALUE ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      Intrepid2::getValues_HGRAD_Args(outputValues,
                                      inputPoints,
                                      operatorType,
                                      this->getBaseCellTopology(),
                                      this->getCardinality() );
#endif
      constexpr ordinal_type numPtsPerEval = Parameters::MaxNumPtsPerBasisEval;
      Impl::Basis_HFACE_HEX_In_FEM::
      getValues<DeviceType,numPtsPerEval>( outputValues,
                                          inputPoints,
                                          this->vinvLine_,
                                          operatorType );
    }
    virtual
    void
    getDofCoords( ScalarViewType dofCoords ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.rank() != 2, std::invalid_argument,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::getDofCoords) rank = 2 required for dofCoords array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }
    
    virtual
    void
    getDofCoeffs( ScalarViewType dofCoeffs ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.rank() != 2, std::invalid_argument,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::getDofCoeffs) rank = 2 required for dofCoeffs array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoeffs.extent(0)) != this->getCardinality(), std::invalid_argument,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::getDofCoeffs) mismatch in number of dof and 0th dimension of dofCoeffs array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                   ">>> ERROR: (Intrepid2::Basis_HFACE_HEX_In_FEM::getDofCoeffs) incorrect reference cell (1st) dimension in dofCoeffs array");
#endif
      Kokkos::deep_copy(dofCoeffs, this->dofCoeffs_);
    }
    
    virtual
    const char*
    getName() const override {
      return "Intrepid2_HFACE_HEX_In_FEM";
    }
    
    virtual
    bool
    requireOrientation() const override {
      return false; // TMW: update when fixed
    }
    
    //BasisPtr<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>
    //getHostBasis() const override{
    //  return Teuchos::rcp(new Basis_HFACE_HEX_In_FEM<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>(this->basisDegree_, pointType_));
    //}
    
  private:
    
    /** \brief inverse of Generalized Vandermonde matrix (isotropic order) */
    Kokkos::DynRankView<typename ScalarViewType::value_type,DeviceType> vinvLine_;
    
    /** \brief type of lattice used for creating the DoF coordinates  */
    EPointType pointType_;
    
  public:
    
    /** \brief returns the basis associated to a subCell.
     
     The bases of the subCell are the restriction to the subCell
     of the bases of the parent cell.
     \param [in] subCellDim - dimension of subCell
     \param [in] subCellOrd - position of the subCell among of the subCells having the same dimension
     \return pointer to the subCell basis of dimension subCellDim and position subCellOrd
     */
    BasisPtr<DeviceType,outputValueType,pointValueType>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override{
      if(subCellDim == 1) {
        return Teuchos::rcp(new
                            Basis_HGRAD_LINE_Cn_FEM<DeviceType,outputValueType,pointValueType>
                            (this->basisDegree_, POINTTYPE_EQUISPACED));
      } else if(subCellDim == 2) {
        return Teuchos::rcp(new
                            Basis_HGRAD_QUAD_Cn_FEM<DeviceType,outputValueType,pointValueType>
                            (this->basisDegree_, POINTTYPE_EQUISPACED));
      }
      INTREPID2_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Input parameters out of bounds");
    }
    
    BasisPtr<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>
    getHostBasis() const override{
      return Teuchos::rcp(new Basis_HFACE_HEX_In_FEM<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>(this->basisDegree_, POINTTYPE_EQUISPACED));
    }
    
  };
  
}// namespace Intrepid2

#include "Intrepid2_HFACE_HEX_In_FEMdef.hpp"

#endif
