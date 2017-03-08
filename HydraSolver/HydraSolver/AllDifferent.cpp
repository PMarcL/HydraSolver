#include "AllDifferent.h"
#include "IntVariable.h"

using namespace std;

namespace hydra {

	AllDifferent::AllDifferent(const vector<IntVariable*>& variables) : variables(variables) {
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

	void AllDifferent::filter() {
	}

	void AllDifferent::filterDomains() {
	}

	void AllDifferent::filterBounds() {
	}

} // namespace hydra
