#include "ConstraintUtils.h"
#include "SumConstraint.h"
#include "AllDifferent.h"

using namespace std;

namespace hydra {

	Constraint* CreateSumConstraint(const vector<Variable*>& variables, int sum) {
		return new SumConstraint(variables, sum);
	}

	Constraint* CreateAllDifferentConstraint(const vector<Variable*>& variables) {
		return new AllDifferent(variables);
	}

} // namespace hydra
