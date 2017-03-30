#include "pSolver.h"
#include "Model.h"
#include "Solver.h"
#include <iostream>
#include <string>

namespace hydra {



	pSolver::pSolver(int numberOfSolvers, Model* model, Heuristic heuristic, Heuristic tieBreaker)
		: model(model), variableSelector(heuristic, tieBreaker), propagator(model->getConstraints()), nbOfBacktracks(0) {
		solvers = std::vector<Solver>();
		for (int i = 0; i < numberOfSolvers; i++)
		{
			Model *newModel = new Model(*model);
			Solver solver(newModel, hydra::RANDOM);
			solvers.push_back(solver);
		}
	}


	pSolver::~pSolver()
	{
	}

	Solution pSolver::findSolution() {

		//Pour tester la solution
		return solvers[0].findSolution();


		//		Solution sol;
		//		int position = 0;

		//		size_t sizevsols = solvers.size();
		//		std::vector<Solution> vsols(sizevsols);
		//		std::vector<Solution> vsolsf(sizevsols);

		//#pragma omp parallel for
		//		for (int i = 0; i < solvers.size(); i++)
		//		{
		//
		//			vsols[i] = solvers[i].findSolution();
		//#pragma omp critical
		//			{
		//				vsolsf[position] = vsols[i];
		//				position += 1;
		//				for (Solver& solver : solvers) {
		//					solver.setOtherSolverHasFinished(true);
		//				}
		//			}

		//		}
		//		std::cout << vsolsf[0].getFormattedSolution() << std::endl;
		//		std::cout << vsolsf[1].getFormattedSolution() << std::endl;
		//		std::cout << vsolsf[2].getFormattedSolution() << std::endl;
		//		std::cout << vsolsf[3].getFormattedSolution() << std::endl;
		//		return vsolsf[4];
	}

	void pSolver::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		for (Solver& solver : solvers) {
			solver.setLocalConsistencyConfig(config);
		}
	}

}
