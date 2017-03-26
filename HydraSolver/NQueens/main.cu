#include <iostream>
#include <string>
#include "Model.h"
#include "ConstraintUtils.h"
#include "Solver.h"

const int N = 10;

int main() {
	hydra::Model model(std::to_string(N) + "-Queens");
	auto queens = model.createIntVarArray("Q", N, 1, N);

	model.postConstraint(CreateAllDifferentConstraint(queens));

	for (auto i = 0; i < N - 1; i++) {
		for (auto j = i + 1; j < N; j++) {
			auto constraint = CreateBinaryArithmeticConstraint(queens[i], queens[j], (i + 1) - (j + 1), hydra::MINUS, hydra::NEQ);
			constraint->setGPUFilteringActive();
			model.postConstraint(constraint);
			model.postConstraint(CreateBinaryArithmeticConstraint(queens[i], queens[j], (j + 1) - (i + 1), hydra::MINUS, hydra::NEQ));
		}
	}

	auto solver = hydra::Solver(&model, hydra::RANDOM);
	solver.setLocalConsistencyConfig(hydra::BOUND_CONSISTENCY);

	auto solution = solver.findSolution();
	std::cout << solution.getFormattedSolution() << std::endl;

	return 0;
}
