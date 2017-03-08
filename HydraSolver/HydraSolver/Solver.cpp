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
			model->getEnvironment().push();

			if (result == INCONSISTENT_STATE) {
				model->getEnvironment().pop();
				return Solution({}, false);
			}

			if (model->getEnvironment().allVariablesAreInstantiated()) {
				return Solution(model->getEnvironment().getVariables(), true);
			}

			auto instantiatedVariable = variableSelector.instantiateVariable(model->getEnvironment().getVariables());
			auto v = instantiatedVariable->getInstantiatedValue();
			solution = findSolution();
			if (!solution.isConsistent()) {
				// if filtering v empties out the domain of the instantiated variable we return an empty solution
				if (instantiatedVariable->cardinality() == 1) {
					model->getEnvironment().pop();
					return Solution({}, false);
				}
				instantiatedVariable->filterValue(v);
			}
		} while (!solution.isConsistent());
		return solution;
	}


} // namespace hydra
