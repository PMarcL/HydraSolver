#include "Solver.h"
#include "Model.h"
#include "Variable.h"

using namespace std;
using namespace chrono;

namespace hydra {

	Solver::Solver(Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()), nbOfBacktracks(0) {
	}

	Solution Solver::findSolution() {
		auto tic = high_resolution_clock::now();
		auto solution = solve();
		auto toc = high_resolution_clock::now();

		solution.setComputingtime(duration_cast<milliseconds>(toc - tic).count());
		solution.setNumberOfBacktracks(nbOfBacktracks);

		return solution;
	}

	Solution Solver::solve() {
		Solution solution;
		do {
			auto result = propagator.propagate();
			model->pushEnvironment();

			if (result == INCONSISTENT_STATE) {
				model->popEnvironment();
				return Solution({}, false, model);
			}

			if (model->allVariablesAreInstantiated()) {
				return Solution(model->getVariables(), true, model);
			}

			auto instantiatedVariable = variableSelector.instantiateVariable(model->getVariables());
			auto v = instantiatedVariable->getInstantiatedValue();
			solution = solve();
			if (!solution.isConsistent()) {
				nbOfBacktracks++;
				// if filtering v empties out the domain of the instantiated variable we return an empty solution
				if (instantiatedVariable->cardinality() == 1) {
					model->popEnvironment();
					return Solution({}, false, model);
				}
				instantiatedVariable->filterValue(v);
			}
		} while (!solution.isConsistent());
		return solution;
	}

} // namespace hydra
