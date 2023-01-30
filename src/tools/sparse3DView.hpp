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

#ifndef MRHYDE_SPARSE3DVIEW_H
#define MRHYDE_SPARSE3DVIEW_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to store sparse data
  // =================================================================
  
  class Sparse3DView {

  public:
    View_Sc3 values;
    Kokkos::View<size_type***,AssemblyDevice> columns, local_columns;
    Kokkos::View<size_type**,AssemblyDevice> nnz_row;
    ScalarT tol;
    size_type maxent;
    bool have_local_columns = false;

    Sparse3DView(View_Sc3 denseview, ScalarT & tol_)
    : tol(tol_)
    {
      ScalarT maxval = 0.0;
      size_type numelem = denseview.extent(0);
      size_type numrows = denseview.extent(1);

      auto denseview_host = create_mirror_view(denseview);
      deep_copy(denseview_host,denseview);

      // Determine the maximum value for compression
      // Could have also used Kokkos::parallel_reduce
      for (size_type elem=0; elem<denseview_host.extent(0); elem++ ) {
        for (size_type i=0; i<denseview_host.extent(1); i++ ) {
          for (size_type j=0; j<denseview_host.extent(2); j++ ) {
            if (std::abs(denseview_host(elem,i,j))>maxval) {
              maxval = std::abs(denseview_host(elem,i,j));
            }
          }
        }
      }

      // Figure out how many entries will be retained per row
      nnz_row = Kokkos::View<size_type**,AssemblyDevice>("num nonzeros per elem/row",numelem,numrows);
      auto nnz_host = create_mirror_view(nnz_row);
      maxent = 0;
      for (size_type elem=0; elem<denseview_host.extent(0); elem++ ) {
        for (size_type i=0; i<denseview_host.extent(1); i++ ) {
          size_type nnz = 0;
          for (size_type j=0; j<denseview_host.extent(2); j++ ) {
            if (std::abs(denseview_host(elem,i,j))/maxval>tol) {
              nnz++;
            }
          }
          nnz_host(elem,i) = nnz;
          maxent = std::max(maxent,nnz);
        }
      }
      deep_copy(nnz_host,nnz_row);

      // Allocate and fill in values and columns
      values = View_Sc3("values",numelem,numrows,maxent);
      columns = Kokkos::View<size_type***,AssemblyDevice>("columns",numelem,numrows,maxent);
      local_columns = Kokkos::View<size_type***,AssemblyDevice>("columns",numelem,numrows,maxent);

      auto values_host = create_mirror_view(values);
      auto columns_host = create_mirror_view(columns);

      for (size_type elem=0; elem<denseview_host.extent(0); elem++ ) {
        for (size_type i=0; i<denseview_host.extent(1); i++ ) {
          size_type prog = 0;
          for (size_type j=0; j<denseview_host.extent(2); j++ ) {
            if (std::abs(denseview_host(elem,i,j))/maxval>tol) {
              columns_host(elem,i,prog) = j;
              values_host(elem,i,prog) = denseview_host(elem,i,j);
              ++prog;
            }
          }
        }
      }
      deep_copy(values,values_host);
      deep_copy(columns,columns_host);
    }
      
    KOKKOS_INLINE_FUNCTION    
    Sparse3DView() {};
    
    KOKKOS_INLINE_FUNCTION
    ~Sparse3DView() {};
    
    KOKKOS_INLINE_FUNCTION
    size_type size() const {
      auto nnz_host = create_mirror_view(nnz_row);
      deep_copy(nnz_host,nnz_row);
      size_type total = 0;
      for (size_type e=0; e<nnz_host.extent(0); ++e) {
        for (size_type j=0; j<nnz_host.extent(1); ++j) {
          total += nnz_host(e,j);
        }
      }
      return total;//values.extent(0)*values.extent(1)*values.extent(2);
    }

    KOKKOS_INLINE_FUNCTION
    View_Sc3 getValues() const {
      return values;
    }
    
    KOKKOS_INLINE_FUNCTION
    bool getStatus() const {
      return have_local_columns;
    }
    
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<size_type***,AssemblyDevice> getColumns() const {
      return columns;
    }
    
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<size_type***,AssemblyDevice> getLocalColumns() const {
      return local_columns;
    }
    
    void setLocalColumns(Kokkos::View<int**,AssemblyDevice> offsets, Kokkos::View<int*,AssemblyDevice> numDOF) {
      parallel_for("get mass",
                    RangePolicy<AssemblyExec>(0,columns.extent(0)),
                    KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type var=0; var<numDOF.extent(0); var++) {
          for (int i=0; i<numDOF(var); i++ ) {
            LO localrow = offsets(var,i);
            for (int k=0; k<nnz_row(elem,localrow); ++k ) {
              for (int j=0; j<numDOF(var); j++ ) {                    
                LO localcol = offsets(var,j);
                if (columns(elem,localrow,k) == localcol) {
                  local_columns(elem,localrow,k) = j;
                }
              }
            }
          }
        }
      });   
      have_local_columns = true;  
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::View<size_type**,AssemblyDevice> getNNZPerRow() const {
      return nnz_row;
    }
    
  };
  
}

#endif

