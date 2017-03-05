#include "FixedIntVariable.h"

using namespace std;

namespace hydra {

	FixedIntVariable::FixedIntVariable(const string& name, int value) : IntVariable(name), value(value) {
	}


	FixedIntVariable::~FixedIntVariable() {
	}

	void FixedIntVariable::pushCurrentState() {
		// fixed value doesn't need to push its state
		return;
	}

	void FixedIntVariable::popState() {
		// fixed value doesn't need to pop an earlier state
		return;
	}

	void FixedIntVariable::filterValue(int value) {
		return;
	}

	void FixedIntVariable::filterLowerBound(int newLowerBound) {
		return;
	}

	void FixedIntVariable::filterUpperBound(int newUpperBound) {
		return;
	}

	int FixedIntVariable::getLowerBound() const {
		return value;
	}

	int FixedIntVariable::getUpperBound() const {
		return value;
	}

	bool FixedIntVariable::containsValue(int value) const {
		return this->value == value;
	}


} // namespace hydra
