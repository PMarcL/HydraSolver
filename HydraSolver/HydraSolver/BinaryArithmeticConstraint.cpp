#include "BinaryArithmeticConstraint.h"
#include "Variable.h"
#include <unordered_set>

using namespace std;

namespace hydra {

	BinaryArithmeticConstraint::BinaryArithmeticConstraint(Variable* var1, Variable* var2, int result, Operator op, RelationalOperator relop) : var1(var1),
		var2(var2), rhs(result), op(op), relop(relop) {
	}

	BinaryArithmeticConstraint::~BinaryArithmeticConstraint() {
	}

	bool BinaryArithmeticConstraint::containsVariable(Variable* var) const {
		return var == var1 || var == var2;
	}

	vector<Variable*> BinaryArithmeticConstraint::filter() {
		return filterDomains();
	}

	vector<Variable*> BinaryArithmeticConstraint::filterBounds() {
		satisfied = true;
		vector<Variable*> filteredVariables;

		if (filterVariableBounds(var1, var2)) {
			filteredVariables.push_back(var1);
		}

		if (var1->cardinality() != 0 && filterVariableBounds(var2, var1)) {
			filteredVariables.push_back(var2);
		}

		if (var1->cardinality() == 0 || var2->cardinality() == 0) {
			satisfied = false;
		}

		return filteredVariables;
	}

	bool BinaryArithmeticConstraint::filterVariableBounds(Variable* varToFilter, Variable* otherVar) {
		auto operation = getOperation();
		auto variableWasFiltered = false;

		auto iterator = varToFilter->iterator();
		while (iterator->hasNextValue()) {
			auto currentValue = iterator->next();
			if (!operation(currentValue, otherVar->getLowerBound()) && !operation(currentValue, otherVar->getUpperBound())) {
				varToFilter->filterValue(currentValue);
				variableWasFiltered = true;
			}
		}
		return variableWasFiltered;
	}


	vector<Variable*> BinaryArithmeticConstraint::filterDomains() {
		satisfied = true;
		vector<Variable*> filteredVariables;

		if (filterVariableDomain(var1, var2)) {
			filteredVariables.push_back(var1);
		}

		if (var1->cardinality() != 0 && filterVariableDomain(var2, var1)) {
			filteredVariables.push_back(var2);
		}

		if (var1->cardinality() == 0 || var2->cardinality() == 0) {
			satisfied = false;
		}

		return filteredVariables;
	}

	bool BinaryArithmeticConstraint::filterVariableDomain(Variable* varToFilter, Variable* otherVar) {
		auto variableWasFiltered = false;
		auto operation = getOperation();

		auto iteratorV1 = varToFilter->iterator();
		while (iteratorV1->hasNextValue()) {
			auto currentV1Value = iteratorV1->next();
			auto valueHasSupport = false;

			auto iteratorV2 = otherVar->iterator();
			auto v2Size = otherVar->cardinality();
			while (iteratorV2->hasNextValue()) {
				auto currentV2Value = iteratorV2->next();
				if (operation(currentV1Value, currentV2Value)) {
					valueHasSupport = true;
					break;
				}
			}
			delete iteratorV2;
			if (!valueHasSupport) {
				varToFilter->filterValue(currentV1Value);
				variableWasFiltered = true;
			}
		}
		delete iteratorV1;
		return variableWasFiltered;
	}


	function<bool(int, int)> BinaryArithmeticConstraint::getOperation() {
		function<int(int, int)> lhsResult;
		switch (op) {
		case PLUS:
			lhsResult = [](int v1, int v2) { return v1 + v2; };
			break;
		case MINUS:
			lhsResult = [](int v1, int v2) { return v1 - v2; };
			break;
		case MULTIPLIES:
			lhsResult = [](int v1, int v2) { return v1 * v2; };
			break;
		case DIVIDES:
			lhsResult = [](int v1, int v2) { return v1 / v2; };
			break;
		}

		function<bool(int, int)> operation;
		switch (relop) {
		case EQ:
			operation = [lhsResult, this](int v1, int v2) { return lhsResult(v1, v2) == rhs; };
			break;
		case NEQ:
			operation = [lhsResult, this](int v1, int v2) { return lhsResult(v1, v2) != rhs; };
			break;
		case GEQ:
			operation = [lhsResult, this](int v1, int v2) { return lhsResult(v1, v2) >= rhs; };
			break;
		case GT:
			operation = [lhsResult, this](int v1, int v2) { return lhsResult(v1, v2) > rhs; };
			break;
		case LEQ:
			operation = [lhsResult, this](int v1, int v2) { return lhsResult(v1, v2) <= rhs; };
			break;
		case LT:
			operation = [lhsResult, this](int v1, int v2) { return lhsResult(v1, v2) < rhs; };
			break;
		}

		return operation;
	}


	bool BinaryArithmeticConstraint::isSatisfied() const {
		return satisfied;
	}

	void BinaryArithmeticConstraint::replaceVariable(Variable* varToReplace, Variable* replacement) {
		if (var1 == varToReplace) {
			var1 = replacement;
		} else if (var2 == varToReplace) {
			var2 = replacement;
		}
	}

	Constraint* BinaryArithmeticConstraint::clone() const {
		return new BinaryArithmeticConstraint(var1, var2, rhs, op, relop);
	}


} // namespace hydra