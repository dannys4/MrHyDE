
#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  //typedef Kokkos::LayoutContiguous<Kokkos::LayoutLeft> Layout;
  
  const int numDerivs = 24;
  typedef Sacado::Fad::SFad<ScalarT,numDerivs> EvalT;
  //typedef double EvalT;
  typedef Kokkos::View<EvalT*,AssemblyDevice> View1;
  typedef Kokkos::View<EvalT**,AssemblyDevice> View2;
  typedef Kokkos::View<EvalT***,AssemblyDevice> View3;
  typedef Kokkos::View<EvalT****,AssemblyDevice> View4;
  typedef Kokkos::View<EvalT**,HostDevice> View2_host;
  typedef Kokkos::View<EvalT****,HostDevice> View4_host;
  
  
  
  {
    int numElem = 20000;
    int numip = 8;
    int numvars = 3;
    int dimension = 3;
    int numrepeats = 1;
    int numdof = 8;
    
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////
    Kokkos::Timer timer;
 
    View4 sol_ip("solution at ip",numElem,numvars,numip,dimension);
    Kokkos::View<ScalarT****,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
    Kokkos::View<EvalT***,AssemblyDevice> sol_dof("sol at dof",numElem,numvars,numdof);
    
    Kokkos::deep_copy(basis,1.0);
    Kokkos::deep_copy(sol_dof,1.0);
    
    {
      timer.reset();
      
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int var=0; var<sol_dof.extent(1); var++) {
          for (int dof=0; dof<basis.extent(1); dof++) {
            EvalT uval = sol_dof(elem,var,dof);
            for (size_t pt=0; pt<basis.extent(2); pt++ ) {
              for (int s=0; s<basis.extent(3); s++ ) {
                sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
              }
            }
          }
        }
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("Baseline GPU time:   %e \n", sol_time1);
      
    }
    
    {
      typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
      const int vector_size = 1;
      const int team_size = 1;//256;
      
      timer.reset();
      
      //parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      parallel_for(TeamPolicy(basis.extent(0),team_size,vector_size), KOKKOS_LAMBDA (const typename TeamPolicy::member_type& team) {
        const size_t elem = team.league_rank();
        for (int var=0; var<sol_dof.extent(1); var++) {
          
          //for (int dof=team_index; dof<basis.extent(1); dof+=team_size) {
          for (int dof=0; dof<basis.extent(1); dof++) {
            EvalT uval = sol_dof(elem,var,dof);
            for (size_t pt=0; pt<basis.extent(2); pt++ ) {
              for (int s=0; s<basis.extent(3); s++ ) {
                sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
              }
            }

          }
        }
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("GPU time (basic team):   %e \n", sol_time1);
      
    }
    
    /*
    {
      typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
      const int vector_size = 1;
      const int team_size = 1;//256;
      using Kokkos::TeamThreadRange;
      using Kokkos::parallel_for;
      using Kokkos::parallel_reduce;
      
      timer.reset();
      
      //parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      parallel_for(TeamPolicy(basis.extent(0),team_size,vector_size), KOKKOS_LAMBDA (const typename TeamPolicy::member_type& team) {
        const size_t elem = team.league_rank();
        const int team_index = team.team_rank();
        
        for (int var=0; var<sol_dof.extent(1); var++) {
          auto csol = Kokkos::subview(sol_dof,elem,var,Kokkos::ALL());
          auto cbasis = Kokkos::subview(basis,elem,Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
          auto csolip = Kokkos::subview(sol_ip,elem,var,Kokkos::ALL(),Kokkos::ALL());
          parallel_for(TeamThreadRange(team, basis.extent(1)), KOKKOS_LAMBDA (const int dof) {
            for (size_t pt=0; pt<basis.extent(2); pt++ ) {
              for (int s=0; s<basis.extent(3); s++ ) {
                csolip(pt,s) += csol(dof)*cbasis(dof,pt,s);
              }
            }
          });
        }
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("GPU time (nested):   %e \n", sol_time1);
      
    }
     */
    
    /*
    {
      View4_host sol_ip("solution at ip",numElem,numvars,numip,dimension);
      Kokkos::View<ScalarT****,HostDevice> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<EvalT***,HostDevice> sol_dof("sol at dof",numElem,numvars,numdof);
      
      parallel_for(RangePolicy<HostExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int dof=0; dof<basis.extent(1); dof++) {
          for (int pt=0; pt<basis.extent(2); pt++) {
            for (int dim=0; dim<basis.extent(3); dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      });
      parallel_for(RangePolicy<HostExec>(0,sol_dof.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int var=0; var<sol_dof.extent(1); var++) {
          for (int dof=0; dof<sol_dof.extent(2); dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      });
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        
        parallel_for(RangePolicy<HostExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int var=0; var<sol_dof.extent(1); var++) {
            for (int dof=0; dof<sol_dof.extent(2); dof++) {
              EvalT uval = sol_dof(elem,var,dof);
              
              if (dof == 0) {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,var,pt,s) = uval*basis(elem,dof,pt,s);
                  }
                }
              }
              else {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
                  }
                }
              }
            }
          }
        });
        
      }
      Kokkos::fence();
      double sol_time = timer.seconds();
      printf("Host time 1:   %e \n", sol_time);
    }
    */
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


