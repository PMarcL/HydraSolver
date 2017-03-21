#include "AllDifferent.h"
#include "AllDifferentUtils.h"
#include "AllDiffBoundsFilter.h"
#include "Variable.h"

using namespace std;

namespace hydra {

	AllDifferent::AllDifferent(const vector<Variable*>& variables) : variables(variables), boundsFilter(new AllDiffBoundsFilter(variables)) {
	}

	AllDifferent::~AllDifferent() {
		delete boundsFilter;
	}

	bool AllDifferent::containsVariable(Variable* variable) const {
		return find(variables.begin(), variables.end(), variable) != variables.end();
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
		vector<Variable*> filteredVariables;
		satisfied = boundsFilter->filter(filteredVariables);
		return filteredVariables;
	}

	bool AllDifferent::isSatisfied() const {
		return satisfied;
	}

	void AllDifferent::replaceVariable(Variable* varToReplace, Variable* replacement) {
		for (size_t i = 0; i < variables.size(); i++) {
			if (variables[i] == varToReplace) {
				variables[i] = replacement;
				break;
			}
		}
	}

	Constraint* AllDifferent::clone() const {
		return new AllDifferent(variables);
	}

} // namespace hydra
