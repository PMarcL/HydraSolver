#include "Solver.h"
#include "Model.h"
#include "Variable.h"
#include <cmath>

using namespace std;
using namespace chrono;

namespace hydra {

	Solver::Solver(Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()), solution(Solution({}, false, model)),
		nbOfBacktracks(0), nbOfRestarts(0), maxNbOfBacktracks(1) {
	}

	Solution Solver::findSolution() {
		auto tic = high_resolution_clock::now();
		auto i = 0;
		while (solve() == RESTART) {
			nbOfBacktracks = 0;
			maxNbOfBacktracks = pow(BASE_RESTART_CONST, i);
			nbOfRestarts++;
			i++;
		}
		auto toc = high_resolution_clock::now();

		solution.setComputingtime(duration_cast<milliseconds>(toc - tic).count());
		solution.setNumberOfBacktracks(nbOfBacktracks);
		solution.setNumberOfRestarts(nbOfRestarts);

		return solution;
	}

	void Solver::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		propagator.setLocalConsistencyConfig(config);
	}

	Solver::SolverState Solver::solve() {
		auto nbOfEnvPush = 0;
		do {
			auto result = propagator.propagate();
			model->pushEnvironment();
			nbOfEnvPush++;

			if (result == INCONSISTENT_STATE) {
				model->popEnvironmentNTimes(nbOfEnvPush);
				return INCONSISTENT_SOLUTION;
			}

			if (model->allVariablesAreInstantiated()) {
				solution = Solution(model->getVariables(), true, model);
				return SOLUTION_FOUND;
			}

			auto instantiatedVariable = variableSelector.instantiateVariable(model->getVariables());
			auto v = instantiatedVariable->getInstantiatedValue();
			auto solveResult = solve();

			if (solveResult == INCONSISTENT_SOLUTION) {
				nbOfBacktracks++;
				if (nbOfBacktracks >= maxNbOfBacktracks) {
					model->popEnvironmentNTimes(nbOfEnvPush);
					return RESTART;
				}
				// if filtering v empties out the domain of the instantiated variable we return an empty solution
				if (instantiatedVariable->cardinality() == 1) {
					model->popEnvironmentNTimes(nbOfEnvPush);
					return INCONSISTENT_SOLUTION;
				}
				instantiatedVariable->filterValue(v);
			} else if (solveResult == RESTART) {
				model->popEnvironmentNTimes(nbOfEnvPush);
				return solveResult;
			}
		} while (!solution.isConsistent());
		return SOLUTION_FOUND;
	}
	/*
	Solution Solver::parallelize() {
		Model modeln(*model);

	}
	*/

} // namespace hydra
