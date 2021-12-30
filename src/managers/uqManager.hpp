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

#ifndef UQ_MANAGER_H
#define UQ_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include <random>
#include <time.h>

namespace MrHyDE {
  
  class UQManager {
    
  public:
    
    UQManager() {};
    
    ~UQManager() {};
  
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    UQManager(const Teuchos::RCP<MpiComm> Comm_, const Teuchos::ParameterList & uqsettings_,
              const std::vector<string> & param_types_,
              const std::vector<ScalarT> & param_means_, const std::vector<ScalarT> & param_variances_,
              const std::vector<ScalarT> & param_mins_, const std::vector<ScalarT> & param_maxs_);
    
    // ========================================================================================
    // ========================================================================================
    
    Kokkos::View<ScalarT**,HostDevice> generateSamples(const int & numsamples, int & seed);
    
    // ========================================================================================
    // ========================================================================================
    
    Kokkos::View<int*,HostDevice> generateIntegerSamples(const int & numsamples, int & seed);
    
    // ========================================================================================
    // ========================================================================================
    
    void generateSamples(const int & numsamples, int & seed,
                         Kokkos::View<ScalarT**,HostDevice> samplepts,
                         Kokkos::View<ScalarT*,HostDevice> samplewts);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeStatistics(const std::vector<ScalarT> & values);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values);
    
    // ========================================================================================
    // ========================================================================================
    
    Teuchos::RCP<MpiComm> Comm;
    std::string surrogate;
    std::vector<std::vector<ScalarT> > points;
    int evalprog, numstochparams;
    bool use_user_defined;
    Teuchos::ParameterList uqsettings;
    std::vector<string> param_types;
    std::vector<ScalarT> param_means, param_variances, param_mins, param_maxs;
    
    Kokkos::View<ScalarT**,HostDevice> samples;
  };
}
#endif
