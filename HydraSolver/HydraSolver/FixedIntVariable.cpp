#include "FixedIntVariable.h"
#include "IllegalVariableOperationException.h"

using namespace std;

namespace hydra {

	FixedIntVariable::FixedIntVariable(const string& name, int value) : IntVariable(name), value(value) {
	}


	FixedIntVariable::~FixedIntVariable() {
	}

	void FixedIntVariable::pushCurrentState() {
		// fixed values don't need to push their state
	}

	void FixedIntVariable::popState() {
		// fixed values don't need to pop an earlier state
	}

	void FixedIntVariable::filterValue(int value) {
		IllegalVariableOperationException e;
		e.setDescription(getErrorDescriptionForMehtod("filterValue"));

		throw e;
	}

	void FixedIntVariable::filterLowerBound(int newLowerBound) {
		IllegalVariableOperationException e;
		e.setDescription(getErrorDescriptionForMehtod("filterLowerBound"));

		throw e;
	}

	void FixedIntVariable::filterUpperBound(int newUpperBound) {
		IllegalVariableOperationException e;
		e.setDescription(getErrorDescriptionForMehtod("filterUpperBound"));

		throw e;
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

	std::string FixedIntVariable::getErrorDescriptionForMehtod(const std::string& methodName) {
		return methodName + " was called on a FixedIntVariable.";
	}

} // namespace hydra
