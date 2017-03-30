#pragma once

#include "VariableSelector.h"
#include "Solution.h"
#include "Propagator.h"

namespace hydra {

	class Model;

	class Solver {
	public:
		explicit Solver(Model* model, Heuristic heuristic = SMALLEST_DOMAIN, Heuristic tieBreaker = RANDOM);

		Solution findSolution();
		void setLocalConsistencyConfig(LocalConsistencyConfig config);
		void setOtherSolverHasFinished(bool boolean);

	private:
		enum SolverState {
			SOLUTION_FOUND,
			INCONSISTENT_SOLUTION,
			RESTART
		};

		SolverState solve();

		static const int BASE_RESTART_CONST = 2;

		Model* model;
		VariableSelector variableSelector;
		Propagator propagator;
		Solution solution;
		int nbOfBacktracks;
		int nbOfRestarts;
		int maxNbOfBacktracks;
		bool otherSolverHasFinished = false;
	};

} // namespace hydra
