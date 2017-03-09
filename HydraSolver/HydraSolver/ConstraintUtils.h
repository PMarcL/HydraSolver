#pragma once

#include <vector>

namespace hydra {

	class SumConstraint;
	class AllDifferent;
	class Variable;

	SumConstraint* CreateSumConstraint(const std::vector<Variable*>& variables, int sum);
	AllDifferent* CreateAllDifferentConstraint(const std::vector<Variable*>& variables);
}
