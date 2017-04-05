#include "MultiAgentSolver.h"
#include "Model.h"
#include "Solver.h"
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
		for (auto solver : solvers) {
			delete solver;
		}
		solvers.clear();
	}

	Solution MultiAgentSolver::findSolution() {
		auto position = 0;
		auto sizevsols = solvers.size();
		vector<Solution> vsols(sizevsols);
		vector<Solution> vsolsf(sizevsols);

#pragma omp parallel for
		for (int i = 0; i < solvers.size(); i++) {
			vsols[i] = solvers[i]->findSolution();
#pragma omp critical
			{
				vsolsf[position] = vsols[i];
				position += 1;
				Solver::setOtherSolverHasFinished(true);
			}
		}
		Solution solutionFound({}, false, nullptr);
		for (auto solution : vsolsf) {
			if (solution.isConsistent()) {
				solutionFound = solution;
				break;
			}
		}
		return solutionFound;
	}

	void MultiAgentSolver::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		for (auto solver : solvers) {
			solver->setLocalConsistencyConfig(config);
		}
	}

}
