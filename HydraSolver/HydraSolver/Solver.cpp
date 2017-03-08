#include "Solver.h"
#include "Model.h"
#include "Variable.h"

namespace hydra {

	Solver::Solver(Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()) {
	}

	Solution Solver::findSolution() {
		Solution solution;
		do {
			auto result = propagator.propagate();
			model->pushEnvironment();

			if (result == INCONSISTENT_STATE) {
				model->popEnvironment();
				return Solution({}, false);
			}

			if (model->allVariablesAreInstantiated()) {
				return Solution(model->getVariables(), true);
			}

			auto instantiatedVariable = variableSelector.instantiateVariable(model->getVariables());
			auto v = instantiatedVariable->getInstantiatedValue();
			solution = findSolution();
			if (!solution.isConsistent()) {
				// if filtering v empties out the domain of the instantiated variable we return an empty solution
				if (instantiatedVariable->cardinality() == 1) {
					model->popEnvironment();
					return Solution({}, false);
				}
				instantiatedVariable->filterValue(v);
			}
		} while (!solution.isConsistent());
		return solution;
	}


} // namespace hydra
