/** \file   observationInterface.hpp
 \brief  Contains the observation interface to MrHyDE.
 \author Created by D. Sharp
 */

#ifndef MRHYDE_OBSERVATIONINTERFACE_H
#define MRHYDE_OBSERVATIONINTERFACE_H

#include "trilinos.hpp"
#include "managers/postprocessManager.hpp"
#include "managers/parameterManager.hpp"
#include "managers/analysisManager.hpp"
#include "interfaces/meshInterface.hpp"
#include "interfaces/physicsInterface.hpp"
#include "interfaces/userInterface.hpp"

/**
 * Create an interface constructed from a filename for an input
 * file and then create some kind of function (and derivative)
 * that simulates observing a PDE at a certain parameter value
 * at the points given in the input file.
 */
namespace MrHyDE{
    class ObservationInterface {
        public:
            ObservationInterface(std::string filename, Teuchos::RCP<MpiComm> & Comm_, std::vector<std::string>& which_params);
            double observe(const std::vector<double> &parameters);
            double observeDerivative(const std::vector<double> &parameters, std::vector<double> &gradient_out);

        private:
            
            Teuchos::RCP<AnalysisManager> SetupAnalysis();
            void ResetParameters(const std::vector<double> &parameters);

            Teuchos::RCP<ParameterManager<SolverNode> > params;
            Teuchos::RCP<Teuchos::ParameterList> settings;
            Teuchos::RCP<MeshInterface> mesh;
            Teuchos::RCP<PhysicsInterface> physics;
            Teuchos::RCP<DiscretizationInterface> disc;
            std::vector<int> param_indices; // indices of the parameters we vary

            const Teuchos::RCP<MpiComm> Comm;
            
            std::string filename;
    };
}

#endif // MRHYDE_OBSERVATIONINTERFACE_H