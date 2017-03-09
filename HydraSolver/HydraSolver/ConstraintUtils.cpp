#include "ConstraintUtils.h"
#include "SumConstraint.h"
#include "AllDifferent.h"

using namespace std;

namespace hydra {

	SumConstraint* CreateSumConstraint(const vector<Variable*>& variables, int sum) {
		return new SumConstraint(variables, sum);
	}

	AllDifferent* CreateAllDifferentConstraint(const vector<Variable*>& variables) {
		return new AllDifferent(variables);
	}

} // namespace hydra
