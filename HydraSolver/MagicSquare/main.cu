#include "Model.h"
#include "MultiAgentSolver.h"
#include "ConstraintUtils.h"
#include <iostream>
#include <string>

const int N = 4;
const int SUM = N * (N * N + 1) / 2;

int main() {

	hydra::Model model("Magic Square");
	auto lines = model.createIntVarMatrix("x", N, N, 1, N*N);

	std::vector<hydra::Variable*> allVars;
	for (size_t i = 0; i < lines.size(); i++) {
		allVars.insert(allVars.end(), lines[i].begin(), lines[i].end());
	}

	model.postConstraint(CreateAllDifferentConstraint(allVars));

	std::vector<std::vector<hydra::Variable*> > columns(N);
	for (size_t i = 0; i < lines.size(); i++) {
		for (size_t j = 0; j < lines[i].size(); j++) {
			columns[i].push_back(lines[j][i]);
		}
	}

	std::vector<hydra::Variable*> diagonal1;
	std::vector<hydra::Variable*> diagonal2;
	for (auto i = 0; i < N; i++) {
		model.postConstraint(CreateSumConstraint(lines[i], SUM));
		model.postConstraint(CreateSumConstraint(columns[i], SUM));
		diagonal1.push_back(lines[i][i]);
		diagonal2.push_back(lines[N - i - 1][i]);
	}
	model.postConstraint(CreateSumConstraint(diagonal1, SUM));
	model.postConstraint(CreateSumConstraint(diagonal2, SUM));


//	auto solver = hydra::Solver(&model, hydra::RANDOM);
//	solver.setLocalConsistencyConfig(hydra::BOUND_CONSISTENCY);
//	auto solution = solver.findSolution();


	auto psolver = hydra::MultiAgentSolver(1, &model, hydra::RANDOM);
	psolver.setLocalConsistencyConfig(hydra::BOUND_CONSISTENCY);
	auto solution = psolver.findSolution();

	std::cout << solution.getFormattedSolution() << std::endl;

	return 0;
}
