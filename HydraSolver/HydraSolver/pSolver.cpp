#include "pSolver.h"
#include "Model.h"
#include "Variable.h"
#include "Solver.h"

namespace hydra {



	pSolver::pSolver(int numberOfSolvers, Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()), nbOfBacktracks(0) {
		solvers = std::vector<Solver>();
		for (int i = 0; i < numberOfSolvers; i++)
		{
			Solver solver(model, heuristic);
			solvers.push_back(solver);
		}
	}


	pSolver::~pSolver()
	{
	}

	Solution pSolver::findSolution() {
		Solution sol;
		//int position = 0;


		// Valeur temporaire à zéro, on teste pour 1 seul solver
		sol = solvers[0].findSolution();
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

	void pSolver::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		for (Solver solver : solvers) {
			solver.setLocalConsistencyConfig(config);
		}
	}

}
