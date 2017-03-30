#include "pSolver.h"
#include "Model.h"
#include "Variable.h"
#include "Solver.h"
#include <iostream>
#include <string>

namespace hydra {



	pSolver::pSolver(int numberOfSolvers, Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()), nbOfBacktracks(0) {
		solvers = std::vector<Solver>();
		for (int i = 0; i < numberOfSolvers; i++)
		{
			Model *newModel = new Model();
			*newModel = *model;
			models.push_back(newModel);
			Solver solver(newModel, hydra::RANDOM);
			solvers.push_back(solver);
		}
	}


	pSolver::~pSolver()
	{
		//		for (Model* mod : models) {
		//			delete[]  mod;
		//		}
	}

	Solution pSolver::findSolution() {
		Solution sol;
		int position = 0;


		// Valeur temporaire à zéro, on teste pour 1 seul solver
		// Il va faire un findSolution sur chaque solver du vector dans différent thread
		//sol = solvers[0].findSolution();
		//return sol;
		size_t sizevsols = solvers.size();
		std::vector<Solution> vsols(sizevsols);
		std::vector<Solution> vsolsf(sizevsols);

#pragma omp parallel for
		for (int i = 0; i < solvers.size(); i++)
		{

			vsols[i] = solvers[i].findSolution();
#pragma omp critical
			{
				vsolsf[position] = vsols[i];
				position += 1;
			}

		}
		std::cout << vsolsf[0].getFormattedSolution() << std::endl;
		std::cout << vsolsf[1].getFormattedSolution() << std::endl;
		return vsolsf[2];
	}

	void pSolver::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		for (Solver& solver : solvers) {
			solver.setLocalConsistencyConfig(config);
		}
	}

}
