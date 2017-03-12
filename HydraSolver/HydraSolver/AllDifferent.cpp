#include "AllDifferent.h"
#include "AllDifferentUtils.h"
#include "Variable.h"

using namespace std;

namespace hydra {

	AllDifferent::AllDifferent(const vector<Variable*>& variables) : variables(variables) {
	}

	AllDifferent::~AllDifferent() {
	}

	bool AllDifferent::containsVariable(Variable* variable) const {
		for (auto var : variables) {
			if (var == variable) {
				return true;
			}
		}
		return false;
	}

	vector<Variable*> AllDifferent::filter() {
		unordered_set<Variable*> filteredVariables;
		satisfied = ReginAlgorithm(variables, filteredVariables);
		return vector<Variable*>(filteredVariables.begin(), filteredVariables.end());
	}

	vector<Variable*> AllDifferent::filterDomains() {
		unordered_set<Variable*> filteredVariables;
		satisfied = ReginAlgorithm(variables, filteredVariables);
		return vector<Variable*>(filteredVariables.begin(), filteredVariables.end());
	}

	vector<Variable*> AllDifferent::filterBounds() {
		unordered_set<Variable*> filteredVariables;
		satisfied = ReginAlgorithm(variables, filteredVariables);
		return vector<Variable*>(filteredVariables.begin(), filteredVariables.end());
	}

	bool AllDifferent::isSatisfied() const {
		return satisfied;
	}

} // namespace hydra
