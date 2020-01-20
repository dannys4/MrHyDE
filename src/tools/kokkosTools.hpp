/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef KKTOOLS_H
#define KKTOOLS_H

#include "trilinos.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Sacado.hpp"
#include "Shards_CellTopology.hpp"
#include "Intrepid2_Utils.hpp"

#include "Kokkos_Core.hpp"
#include "preferences.hpp"
#include "Teuchos_FancyOStream.hpp"

typedef Kokkos::DynRankView<ScalarT,AssemblyDevice> DRV;
typedef Kokkos::DynRankView<int,AssemblyDevice> DRVint;

class KokkosTools {
public:
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T*,AssemblyDevice> V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  value  " << endl;
    cout << "--------------------" << endl;
    
    for (unsigned int i=0; i<V.dimension(0); i++) {
      for (unsigned int j=0; j<V.dimension(1); j++) {
        cout << "  " << i << "  " << "  " << "  " << V(i,j) << "  " << endl;
      }
    }
    cout << "--------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T**,AssemblyDevice> V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  j  " << "  value  " << endl;
    cout << "-------------------------------" << endl;
    
    for (unsigned int i=0; i<V.dimension(0); i++) {
      for (unsigned int j=0; j<V.dimension(1); j++) {
        cout << "  " << i << "  " << "  " << j << "  " <<
        "  " << "  " << V(i,j) << "  " << endl;
      }
    }
    cout << "-------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(Teuchos::RCP<MpiComm> & Comm, vector_RCP & V, const string & message="") {
    auto V_kv = V->getLocalView<HostDevice>();
    
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V_kv.label() << endl;
    
    cout << " PID " << "  i  " << "  j  " << "  value  " << endl;
    cout << "------------------------------------------" << endl;
    
    for (unsigned int i=0; i<V_kv.dimension(0); i++) {
      for (unsigned int j=0; j<V_kv.dimension(1); j++) {
        cout << "  " << Comm->getRank() <<  "  " << i << "  " << "  " << j << "  " <<
        "  " << "  " << V_kv(i,j) << "  " << endl;
      }
    }
    cout << "------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(matrix_RCP & M, const string & message="") {
    cout << message << endl;
    Teuchos::EVerbosityLevel vl = Teuchos::VERB_EXTREME;
    auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
    M->describe(*out,vl);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(const vector_RCP & V, const string & message="") {
    cout << message << endl;
    Teuchos::EVerbosityLevel vl = Teuchos::VERB_EXTREME;
    auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
    V->describe(*out,vl);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(FDATA V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  j  " << "  value  " << endl;
    cout << "-------------------------------" << endl;
    
    for (unsigned int i=0; i<V.dimension(0); i++) {
      for (unsigned int j=0; j<V.dimension(1); j++) {
        cout << "  " << i << "  " << "  " << j << "  " <<
        "  " << "  " << V(i,j) << "  " << endl;
      }
    }
    cout << "-------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T***,AssemblyDevice> V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V.label() << endl;
    
    cout << "  i  " << "  j  " << "  k  " << "  value  " << endl;
    cout << "------------------------------------------" << endl;
    
    for (unsigned int i=0; i<V.dimension(0); i++) {
      for (unsigned int j=0; j<V.dimension(1); j++) {
        for (unsigned int k=0; k<V.dimension(2); k++) {
          cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << k << "  " << "  " << V(i,j,k) << "  " << endl;
        }
      }
    }
    cout << "------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T****,AssemblyDevice> V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V.label() << endl;
    cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << endl;
    cout << "-----------------------------------------------------" << endl;
    
    for (unsigned int i=0; i<V.dimension(0); i++) {
      for (unsigned int j=0; j<V.dimension(1); j++) {
        for (unsigned int k=0; k<V.dimension(2); k++) {
          for (unsigned int n=0; n<V.dimension(3); n++) {
            cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << endl;
          }
        }
      }
    }
    cout << "-----------------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  template<class T>
  static void print(Kokkos::View<T*****,AssemblyDevice> V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for View: " << V.label() << endl;
    cout << "  i  " << "  j  " << "  k  " << "  n  " << "  m  " << "  value  " << endl;
    cout << "----------------------------------------------------------------" << endl;
    
    for (unsigned int i=0; i<V.dimension(0); i++) {
      for (unsigned int j=0; j<V.dimension(1); j++) {
        for (unsigned int k=0; k<V.dimension(2); k++) {
          for (unsigned int n=0; n<V.dimension(3); n++) {
            for (unsigned int m=0; m<V.dimension(4); m++) {
              cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << m
              << "  " << "  " << V(i,j,k,n,m) << "  " << endl;
            }
          }
        }
      }
    }
    cout << "----------------------------------------------------------------" << endl;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(DRV V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for DynRankView: " << V.label() << endl;
  
    if (V.rank() == 2) {
      cout << "  i  " << "  j  " << "  value  " << endl;
      cout << "-------------------------------" << endl;
      
      for (unsigned int i=0; i<V.dimension(0); i++) {
        for (unsigned int j=0; j<V.dimension(1); j++) {
          cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V(i,j) << "  " << endl;
        }
      }
      cout << "-------------------------------" << endl;
      
    }
    else if (V.rank() == 3) {
      cout << "  i  " << "  j  " << "  k  " << "  value  " << endl;
      cout << "------------------------------------------" << endl;
      
      for (unsigned int i=0; i<V.dimension(0); i++) {
        for (unsigned int j=0; j<V.dimension(1); j++) {
          for (unsigned int k=0; k<V.dimension(2); k++) {
            cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << V(i,j,k) << "  " << endl;
          }
        }
      }
      cout << "------------------------------------------" << endl;
      
    }
    else if (V.rank() == 4) {
      cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << endl;
      cout << "-----------------------------------------------------" << endl;
      
      for (unsigned int i=0; i<V.dimension(0); i++) {
        for (unsigned int j=0; j<V.dimension(1); j++) {
          for (unsigned int k=0; k<V.dimension(2); k++) {
            for (unsigned int n=0; n<V.dimension(3); n++) {
              cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << endl;
            }
          }
        }
      }
      cout << "-----------------------------------------------------" << endl;
      
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void print(DRVint V, const string & message="") {
    cout << endl;
    cout << message << endl;
    cout << "Printing data for DynRankView: " << V.label() << endl;
    
    if (V.rank() == 2) {
      cout << "  i  " << "  j  " << "  value  " << endl;
      cout << "-------------------------------" << endl;
      
      for (unsigned int i=0; i<V.dimension(0); i++) {
        for (unsigned int j=0; j<V.dimension(1); j++) {
          cout << "  " << i << "  " << "  " << j << "  " <<
          "  " << "  " << V(i,j) << "  " << endl;
        }
      }
      cout << "-------------------------------" << endl;
      
    }
    else if (V.rank() == 3) {
      cout << "  i  " << "  j  " << "  k  " << "  value  " << endl;
      cout << "------------------------------------------" << endl;
      
      for (unsigned int i=0; i<V.dimension(0); i++) {
        for (unsigned int j=0; j<V.dimension(1); j++) {
          for (unsigned int k=0; k<V.dimension(2); k++) {
            cout << "  " << i << "  " << "  " << j << "  " <<
            "  " << k << "  " << "  " << V(i,j,k) << "  " << endl;
          }
        }
      }
      cout << "------------------------------------------" << endl;
      
    }
    else if (V.rank() == 4) {
      cout << "  i  " << "  j  " << "  k  " << "  n  " << "  value  " << endl;
      cout << "-----------------------------------------------------" << endl;
      
      for (unsigned int i=0; i<V.dimension(0); i++) {
        for (unsigned int j=0; j<V.dimension(1); j++) {
          for (unsigned int k=0; k<V.dimension(2); k++) {
            for (unsigned int n=0; n<V.dimension(3); n++) {
              cout << "  " << i << "  " << "  " << j << "  " <<
              "  " << k << "  " << "  " << n << "  " << "  " << V(i,j,k,n) << "  " << endl;
            }
          }
        }
      }
      cout << "-----------------------------------------------------" << endl;
      
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  template<class T>
  static void checkSizes(Kokkos::View<T*,AssemblyDevice> V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  template<class T>
  static void checkSizes(Kokkos::View<T**,AssemblyDevice> V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  template<class T>
  static void checkSizes(Kokkos::View<T***,AssemblyDevice> V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  template<class T>
  static void checkSizes(Kokkos::View<T****,AssemblyDevice> V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  template<class T>
  static void checkSizes(Kokkos::View<T*****,AssemblyDevice> V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void checkSizes(DRV V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void checkSizes(DRVint V, vector<int> & sizes, const string message="") {
    cout << endl;
    cout << message << endl;
    
    if (V.rank() != sizes.size()) {
      cout << "ERROR ---" << endl;
      cout << "Rank of View = " << V.rank() << "    Expected rank = " << sizes.size() << endl;
      cout << "---------" << endl;
    }
    else {
      for (size_t k=0; k<V.rank(); k++) {
        if (V.dimension(k) != sizes[k]) {
          cout << "ERROR ---" << endl;
          cout << "Size of dimension(" << k << ") = " << V.dimension(k) << "    Expected size = " << sizes[k] << endl;
          cout << "---------" << endl;
        }
      }
    }
  }
};
#endif

