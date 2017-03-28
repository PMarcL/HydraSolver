#pragma once
#include "VariableSelector.h"
#include "Solution.h"
#include "Propagator.h"
#include "Solver.h"

namespace hydra {


	class pSolver
	{
	public:

		~pSolver();
		explicit pSolver(int numberOfSolvers, Model* model, Heuristic heuristic = SMALLEST_DOMAIN, Heuristic tieBreaker = RANDOM);

		Solution findSolution();
		void setLocalConsistencyConfig(LocalConsistencyConfig config);

	private:
		std::vector<Solver> solvers;
		Model* model;
		VariableSelector variableSelector;
		Propagator propagator;
		int nbOfBacktracks;

	};
}
