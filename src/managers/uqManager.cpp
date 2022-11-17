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

#include "uqManager.hpp"
#include "data.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

UQManager::UQManager(const Teuchos::RCP<MpiComm> Comm_, const Teuchos::ParameterList & uqsettings_,
                     const std::vector<string> & param_types_,
                     const std::vector<ScalarT> & param_means_, const std::vector<ScalarT> & param_variances_,
                     const std::vector<ScalarT> & param_mins_, const std::vector<ScalarT> & param_maxs_) :
Comm(Comm_), uqsettings(uqsettings_), param_types(param_types_), param_means(param_means_),
param_variances(param_variances_), param_mins(param_mins_), param_maxs(param_maxs_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::UQManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  numstochparams = param_types.size();
  surrogate = uqsettings.get<std::string>("surrogate model","regression");
  evalprog = 0;
  
  use_user_defined = uqsettings.get<bool>("use user defined",false);
  if (use_user_defined) {
    Data sdata("Sample Points", numstochparams, uqsettings.get("source","samples.dat"));
    samples = sdata.getPoints();
  }
  
  if (surrogate == "regression") {
    
  }
  else if (surrogate == "sparse grid") {
  }
  else if (surrogate == "voronoi") {
  }
  else {
    // complain
  }
  
}

// ========================================================================================
// ========================================================================================

Kokkos::View<ScalarT**,HostDevice> UQManager::generateSamples(const int & numsamples, int & seed) {

  if (!use_user_defined) {
    if (seed == -1) {
      srand(time(NULL));
      seed = rand();
    }

    samples = Kokkos::View<ScalarT**,HostDevice>("samples",numsamples, numstochparams);
    std::default_random_engine generator(seed);
    for (int j=0; j<numstochparams; j++) {
      if (param_types[j] == "uniform") {
        std::uniform_real_distribution<ScalarT> distribution(param_mins[j],param_maxs[j]);
        for (int k=0; k<numsamples; k++) {
          ScalarT number = distribution(generator);
          samples(k,j) = number;
        }
      }
      else if (param_types[j] == "Gaussian") {
        std::normal_distribution<ScalarT> distribution(param_means[j],param_variances[j]);
        for (int k=0; k<numsamples; k++) {
          ScalarT number = distribution(generator);
          samples(k,j) = number;
        }
      }
    }
  }
  
  return samples;
}

// ========================================================================================
// ========================================================================================

Kokkos::View<int*,HostDevice> UQManager::generateIntegerSamples(const int & numsamples, int & seed) {
  if (seed == -1) {
    srand(time(NULL));
    seed = rand();
  }
  
  Kokkos::View<int*,HostDevice> isamples("samples",numsamples);
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(1,1000000);
  for (int k=0; k<numsamples; k++) {
    ScalarT number = distribution(generator);
    isamples(k) = number;
  }
  return isamples;
}

// ========================================================================================
// ========================================================================================

void UQManager::generateSamples(const int & numsamples, int & seed,
                                Kokkos::View<ScalarT**,HostDevice> samplepts,
                                Kokkos::View<ScalarT*,HostDevice> samplewts) {
  Kokkos::resize(samplepts,numsamples, numstochparams);
  Kokkos::resize(samplewts, numsamples);

  if (seed == -1) {
    srand(time(NULL));
    seed = rand();
  }
  
  std::default_random_engine generator(seed);
  for (int j=0; j<numstochparams; j++) {
    if (param_types[j] == "uniform") {
      std::uniform_real_distribution<ScalarT> distribution(param_mins[j],param_maxs[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samplepts(k,j) = number;
        samplewts(k) = 1.0/(ScalarT)numsamples;
      }
    }
    else if (param_types[j] == "Gaussian") {
      std::normal_distribution<ScalarT> distribution(param_means[j],param_variances[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samplepts(k,j) = number;
        samplewts(k) = 1.0/(ScalarT)numsamples;
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void UQManager::computeStatistics(const std::vector<ScalarT> & values) {
  int numvals = values.size();
  if (uqsettings.get<bool>("compute mean",true)) {
    ScalarT meanval = 0.0;
    for (int j=0; j<numvals; j++) {
      meanval += values[j];
    }
    meanval = meanval / numvals;
    if (Comm->getRank() == 0 )
    cout << "Mean value of the response: " << meanval << endl;
  }
  if (uqsettings.get<bool>("compute variance",true)) {
    ScalarT meanval = 0.0;
    for (int j=0; j<numvals; j++) {
      meanval += values[j];
    }
    meanval = meanval / numvals;
    ScalarT variance = 0.0;
    for (int j=0; j<numvals; j++) {
      variance += (values[j]-meanval)*(values[j]-meanval);
    }
    variance = variance / numvals;
    if (Comm->getRank() == 0 )
    cout << "Variance of the response: " << variance << endl;
  }
  if (uqsettings.isSublist("Probability levels")) {
    Teuchos::ParameterList plevels = uqsettings.sublist("Probability levels");
    Teuchos::ParameterList::ConstIterator pl_itr = plevels.begin();
    while (pl_itr != plevels.end()) {
      ScalarT currplevel = plevels.get<ScalarT>(pl_itr->first);
      int count = 0;
      for (int j=0; j<numvals; j++) {
        if (values[j] <= currplevel)
        count += 1;
      }
      ScalarT currprob = (ScalarT)count / (ScalarT)numvals;
      if (Comm->getRank() == 0 )
      cout << "Probability the response is less than " << currplevel << " = " << currprob << endl;
      pl_itr++;
    }
  }
  
  
}

// ========================================================================================
// ========================================================================================

void UQManager::computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values) {
  int numvals = values.size();
  // assumes that values[i] is a rank-3 FC
  int dim0 = values[0].extent(0);
  int dim1 = values[0].extent(1);
  int dim2 = values[0].extent(2);
  
  if (uqsettings.get<bool>("compute mean",true)) {
    
    Kokkos::View<ScalarT***,HostDevice> meanval("mean values",dim0,dim1,dim2);
    for (int j=0; j<numvals; j++) {
      for (int d0=0; d0<dim0; d0++) {
        for (int d1=0; d1<dim1; d1++) {
          for (int d2=0; d2<dim2; d2++) {
            meanval(d0,d1,d2) += values[j](d0,d1,d2)/numvals;
          }
        }
      }
    }
    if (Comm->getRank() == 0 )
    cout << "Mean value of the response: " << endl;
    // KokkosTools::print(meanval); // GH: commenting this out for now; it can't tell DRV from DRVint
  }
  /*if (uqsettings.get<bool>("Compute variance",true)) {
   ScalarT meanval = 0.0;
   for (int j=0; j<numvals; j++) {
   meanval += values[j];
   }
   meanval = meanval / numvals;
   ScalarT variance = 0.0;
   for (int j=0; j<numvals; j++) {
   variance += (values[j]-meanval)*(values[j]-meanval);
   }
   variance = variance / numvals;
   if (Comm.getRank() == 0 )
   cout << "Variance of the response: " << variance << endl;
   }
   if (uqsettings.isSublist("Probability levels")) {
   Teuchos::ParameterList plevels = uqsettings.sublist("Probability levels");
   Teuchos::ParameterList::ConstIterator pl_itr = plevels.begin();
   while (pl_itr != plevels.end()) {
   ScalarT currplevel = plevels.get<ScalarT>(pl_itr->first);
   int count = 0;
   for (int j=0; j<numvals; j++) {
   if (values[j] <= currplevel)
   count += 1;
   }
   ScalarT currprob = (ScalarT)count / (ScalarT)numvals;
   if (Comm.getRank() == 0 )
   cout << "Probability the response is less than " << currplevel << " = " << currprob << endl;
   pl_itr++;
   }
   }*/
  
  
}

// ========================================================================================
// ========================================================================================

View_Sc1 UQManager::KDE(View_Sc2 seedpts, View_Sc2 evalpts) {

  // TMW: note that this won't really work on a GPU - easy to fix though

  // This is mostly translated from a matlab implementation
  size_type Ne = evalpts.extent(0);	// number of evaluation points (not seed points)
  size_type Ns = seedpts.extent(0);	// number of seed points
  size_type dim = evalpts.extent(1); // dimension of the space

  View_Sc1 density("KDE",Ne);

  View_Sc1 variance = this->computeVariance(seedpts);

  ScalarT Nsc = static_cast<ScalarT>(Ns);

  vector<ScalarT> sigma(dim,0.0), coeff(dim,0.0);
  for (size_type d=0; d<dim; ++d) {
    sigma[d] = 1.06*std::sqrt(variance(d))*std::pow(Ns,-1.0/5.0);
    coeff[d] = 1.0/(std::sqrt(2.0*PI*sigma[d]*sigma[d]));
  }

  // kernel density estimation
  for (size_type i=0; i<Ne; ++i) {
    ScalarT val = 0.0;
    for (size_type k=0; k<Ns; ++k) {
      ScalarT cval = 1.0;
      for (size_type d=0; d<dim; ++d) {
        ScalarT diff = evalpts(i,d) - seedpts(k,d);
        cval *= coeff[d]*std::exp(-1.0*diff*diff/(2.0*sigma[d]*sigma[d]));
      }
      val += cval;
    }
    density(i) = val/Nsc;
  }

  return density;
    
}

// ========================================================================================
// ========================================================================================

View_Sc1 UQManager::computeVariance(View_Sc2 pts) {
  size_type N = pts.extent(0);
  size_type dim = pts.extent(1);
  View_Sc1 vars("variance",dim);

  ScalarT Nsc = static_cast<ScalarT>(N);
  auto pts_host = create_mirror_view(pts);
  deep_copy(pts_host,pts);

  for (size_type i=0; i<dim; ++i) {
    ScalarT mean = 0.0;
    for (size_type k=0; k<N; ++k) {
      mean += pts_host(k,i)/Nsc;
    }
    ScalarT var = 0.0;
    for (size_type k=0; k<N; ++k) {
      var += (pts_host(k,i)-mean)*(pts_host(k,i)-mean)/Nsc;
    }
    vars(i) = var;
  }
  return vars;

}

// ========================================================================================
// ========================================================================================

Kokkos::View<bool*,HostDevice> UQManager::rejectionSampling(View_Sc1 ratios, const int & seed) {

  ScalarT C = 0.0;
  for (size_type k=0; k<ratios.extent(0); ++k) {
    if (std::abs(ratios(k)) > C) {
      C = std::abs(ratios(k));
    }
  }
  Kokkos::View<bool*,HostDevice> accept("acceptance",ratios.extent(0));
  deep_copy(accept,false);

  int cseed = seed;
  if (cseed == -1) {
    srand(time(NULL));
    cseed = rand();
  }
  
  std::default_random_engine generator(cseed);
  std::uniform_real_distribution<ScalarT> distribution(0.0,1.0);
        
  for (size_type k=0; k<ratios.extent(0); ++k) {
    ScalarT check = distribution(generator);
    if (std::abs(ratios(k))/C >= check) {
      accept(k) = true;
    }
  }
  return accept;
}