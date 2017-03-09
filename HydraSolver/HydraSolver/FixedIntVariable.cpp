#include "FixedIntVariable.h"
#include "IllegalVariableOperationException.h"
#include <string>

using namespace std;

namespace hydra {

	FixedIntVariable::FixedIntVariable(const string& name, int value) : Variable(name), value(value), valueFiltered(false) {
	}

	FixedIntVariable::~FixedIntVariable() {
	}

	string FixedIntVariable::getFormattedDomain() const {
		return "{ " + to_string(value) + " }";
	}

	void FixedIntVariable::pushCurrentState() {
		// fixed values don't need to push their state
	}

	void FixedIntVariable::popState() {
		if (valueFiltered) {
			valueFiltered = false;
		}
	}

	int FixedIntVariable::cardinality() const {
		return valueFiltered ? 0 : 1;
	}

	void FixedIntVariable::instantiate() {
		// fixed values don't need to be instantiated
	}

	int FixedIntVariable::getInstantiatedValue() const {
		return value;
	}

	void FixedIntVariable::filterValue(int valueToFilter) {
		if (valueToFilter != value) {
			IllegalVariableOperationException e;
			e.setDescription(getErrorDescriptionForMehtod("filterValue") + "with value not equal to current value.");

			throw e;
		}

		valueFiltered = true;
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
