#pragma once

#include <vector>
#include "BinaryArithmeticConstraint.h"

namespace hydra {

	class Constraint;
	class Variable;

	Constraint* CreateSumConstraint(const std::vector<Variable*>& variables, int sum, bool useGPU = false);
	Constraint* CreateAllDifferentConstraint(const std::vector<Variable*>& variables);
	Constraint* CreateBinaryArithmeticConstraint(Variable* v1, Variable* v2, int rhs, Operator op, RelationalOperator relop, bool useGPU = false);
}
