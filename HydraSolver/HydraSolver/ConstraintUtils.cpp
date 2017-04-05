#include "ConstraintUtils.h"
#include "SumConstraint.h"
#include "AllDifferent.h"

using namespace std;

namespace hydra {

	Constraint* CreateSumConstraint(const vector<Variable*>& variables, int sum, bool useGPU) {
		auto constraint = new SumConstraint(variables, sum);
		if (useGPU) {
			constraint->setGPUFilteringActive();
		}
		return constraint;
	}

	Constraint* CreateAllDifferentConstraint(const vector<Variable*>& variables) {
		return new AllDifferent(variables);
	}

	Constraint* CreateBinaryArithmeticConstraint(Variable* v1, Variable* v2, int rhs, Operator op, RelationalOperator relop, bool useGPU) {
		auto constraint = new BinaryArithmeticConstraint(v1, v2, rhs, op, relop);
		if (useGPU) {
			constraint->setGPUFilteringActive();
		}
		return constraint;
	}

} // namespace hydra
