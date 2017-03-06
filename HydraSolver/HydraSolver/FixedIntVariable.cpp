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

	int FixedIntVariable::cardinality() const {
		return 1;
	}

	void FixedIntVariable::filterValue(int) {
		IllegalVariableOperationException e;
		e.setDescription(getErrorDescriptionForMehtod("filterValue"));

		throw e;
	}

	void FixedIntVariable::filterLowerBound(int) {
		IllegalVariableOperationException e;
		e.setDescription(getErrorDescriptionForMehtod("filterLowerBound"));

		throw e;
	}

	void FixedIntVariable::filterUpperBound(int) {
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

	IntVariableIterator* FixedIntVariable::iterator() {
		return new FixedIntIterator(value);
	}

	string FixedIntVariable::getErrorDescriptionForMehtod(const std::string& methodName) const {
		return methodName + " was called on a FixedIntVariable (" + name + ").";
	}

} // namespace hydra
