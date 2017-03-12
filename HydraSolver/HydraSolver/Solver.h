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

	private:
		Solution solve();

		Model* model;
		VariableSelector variableSelector;
		Propagator propagator;
		int nbOfBacktracks;
	};

} // namespace hydra
