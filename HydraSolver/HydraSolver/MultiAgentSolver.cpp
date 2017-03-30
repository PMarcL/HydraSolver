#include "MultiAgentSolver.h"
#include "Model.h"
#include "Solver.h"
#include <iostream>
#include <string>

using namespace std;

namespace hydra {

	MultiAgentSolver::MultiAgentSolver(int numberOfSolvers, Model* model, Heuristic heuristic, Heuristic tieBreaker) : solvers(vector<Solver*>()) {
		for (auto i = 0; i < numberOfSolvers; i++) {
			auto newModel = new Model(*model);
			solvers.push_back(new Solver(newModel, RANDOM));
		}
	}

	MultiAgentSolver::~MultiAgentSolver() {
	}

	Solution MultiAgentSolver::findSolution() {
		//Pour tester la solution
		return solvers[0]->findSolution();


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

	void MultiAgentSolver::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		for (auto solver : solvers) {
			solver->setLocalConsistencyConfig(config);
		}
	}

}
