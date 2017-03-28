#include "pSolver.h"
#include "Model.h"
#include "Variable.h"
#include "Solver.h"

namespace hydra {



	pSolver::pSolver(Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()), nbOfBacktracks(0) {
	}


	pSolver::~pSolver()
	{
	}

	Solution pSolver::findnsolutions() {
		Solution sol;
		//int position = 0;

		Solver solver(model, hydra::RANDOM);
		sol = solver.findSolution();
		return sol;
		std::vector<Solution> vsols;
		/*
#pragma omp parallel
		{
			Model modeln(*model);
			Solver solver(&modeln);
			sol = solver.findSolution();
#pragma omp critical
			{
				vsols[position] = sol;
				position += 1;
			}
		}
		return vsols[0];*/
	}
}
