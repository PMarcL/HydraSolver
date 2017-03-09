#pragma once

#include <vector>

namespace hydra {

	class Constraint;
	class Variable;

	Constraint* CreateSumConstraint(const std::vector<Variable*>& variables, int sum);
	Constraint* CreateAllDifferentConstraint(const std::vector<Variable*>& variables);
}
