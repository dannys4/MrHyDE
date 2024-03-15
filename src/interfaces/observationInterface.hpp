/**
 * Create an interface constructed from a filename for an input
 * file and then create some kind of function (and derivative)
 * that simulates observing a PDE at a certain parameter value
 * at the points given in the input file.
 */
namespace MrHyDE{
    class ObservationInterface {
        public:
            ObservationInterface(std::string filename);
            double observe(const std::vector<double> &parameters);
            void observeDerivative(const std::vector<double> &parameters, std::vector<double> &derivative);

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