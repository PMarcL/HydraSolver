#pragma once

#include "VariableSelector.h"
#include "Solution.h"
#include "Propagator.h"

namespace hydra {

	class Solver;

	class MultiAgentSolver {
	public:
		MultiAgentSolver(int numberOfSolvers, Model* model, Heuristic heuristic = SMALLEST_DOMAIN, Heuristic tieBreaker = RANDOM);
		~MultiAgentSolver();

		Solution findSolution();
		void setLocalConsistencyConfig(LocalConsistencyConfig config);

	private:
		std::vector<Solver*> solvers;
	};
}
