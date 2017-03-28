#pragma once
#include "VariableSelector.h"
#include "Solution.h"
#include "Propagator.h"
namespace hydra {


	class pSolver
	{
	public:

		~pSolver();
		explicit pSolver(Model* model, Heuristic heuristic = SMALLEST_DOMAIN, Heuristic tieBreaker = RANDOM);
		Solution findnsolutions();
	private:
		Model* model;
		VariableSelector variableSelector;
		Propagator propagator;
		int nbOfBacktracks;

	};
}
