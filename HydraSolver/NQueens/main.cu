#include <iostream>
#include <string>
#include "Model.h"
#include "ConstraintUtils.h"
#include "Solver.h"

const int N = 10;
const bool USE_GPU = true;

int main() {
	hydra::Model model(std::to_string(N) + "-Queens");
	auto queens = model.createIntVarArray("Q", N, 1, N);

	for (auto i = 0; i < N - 1; i++) {
		for (auto j = i + 1; j < N; j++) {
			auto constraint = CreateBinaryArithmeticConstraint(queens[i], queens[j], (i + 1) - (j + 1), hydra::MINUS, hydra::NEQ, USE_GPU);
			model.postConstraint(constraint);

			constraint = CreateBinaryArithmeticConstraint(queens[i], queens[j], (j + 1) - (i + 1), hydra::MINUS, hydra::NEQ, USE_GPU);
			model.postConstraint(constraint);

			constraint = CreateBinaryArithmeticConstraint(queens[i], queens[j], 0, hydra::MINUS, hydra::NEQ, USE_GPU);
			model.postConstraint(constraint);
		}
	}

	auto solver = hydra::Solver(&model, hydra::RANDOM);
	solver.setLocalConsistencyConfig(hydra::DOMAIN_CONSISTENCY);

	auto solution = solver.findSolution();
	std::cout << solution.getFormattedSolution() << std::endl;

	return 0;
}
