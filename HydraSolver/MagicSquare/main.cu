#include "Model.h"
#include "MultiAgentSolver.h"
#include "ConstraintUtils.h"
#include <iostream>
#include <string>

const int N = 5;
const bool USE_GPU = false;
const int SUM = N * (N * N + 1) / 2;

int main() {
	std::cout << "Solving magic square of order " << N << ". The sum should be " << SUM << ". " << std::endl;
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
		model.postConstraint(CreateSumConstraint(lines[i], SUM, USE_GPU));
		model.postConstraint(CreateSumConstraint(columns[i], SUM, USE_GPU));
		diagonal1.push_back(lines[i][i]);
		diagonal2.push_back(lines[N - i - 1][i]);
	}
	model.postConstraint(CreateSumConstraint(diagonal1, SUM, USE_GPU));
	model.postConstraint(CreateSumConstraint(diagonal2, SUM, USE_GPU));

	auto psolver = hydra::MultiAgentSolver(8, &model, hydra::RANDOM);
	psolver.setLocalConsistencyConfig(hydra::INTERVAL_CONSISTENCY);
	auto solution = psolver.findSolution();

	std::cout << solution.getFormattedSolution() << std::endl;

	std::cout << solution.isConsistent() << std::endl;
	return 0;
}
