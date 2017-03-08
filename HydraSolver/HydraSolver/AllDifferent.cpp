#include "AllDifferent.h"
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
		return vector<Variable*>();
	}

	vector<Variable*> AllDifferent::filterDomains() {
		return vector<Variable*>();
	}

	vector<Variable*> AllDifferent::filterBounds() {
		return vector<Variable*>();
	}

	bool AllDifferent::isSatisfied() const {
		return false;
	}

} // namespace hydra
