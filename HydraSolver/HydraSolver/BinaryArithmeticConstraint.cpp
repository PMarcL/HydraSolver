#include "BinaryArithmeticConstraint.h"
#include "BitsetIntVariable.h"
#include "Variable.h"

using namespace std;

namespace hydra {

	BinaryArithmeticConstraint::BinaryArithmeticConstraint(Variable* var1, Variable* var2, int result, Operator op, RelationalOperator relop) :
		Constraint({ var1, var2 }), var1(var1), var2(var2), rhs(result), op(op), relop(relop), operation(getOperation(op, relop)), gpuFilter(nullptr),
		timelogger("BinaryArithmeticConstraint") {
		gpuFilter = new BinaryArithmeticIncrementalGPUFilter(static_cast<BitsetIntVariable*>(var1), static_cast<BitsetIntVariable*>(var2), op, relop, rhs);
	}

	BinaryArithmeticConstraint::~BinaryArithmeticConstraint() {
		delete gpuFilter;
	}

	vector<Variable*> BinaryArithmeticConstraint::filter() {
		return filterDomains();
	}

	vector<Variable*> BinaryArithmeticConstraint::filterBounds() {
		satisfied = true;
		vector<Variable*> filteredVariables;
		timelogger.tic();
		if (useGPU) {
			filteredVariables = gpuFilter->filterBoundsGPU();
		} else {
			auto var1WasFiltered = false;

			auto iterator = var1->iterator();
			while (iterator->hasNextValue()) {
				auto currentValue = iterator->next();
				if (!operation(currentValue, var2->getLowerBound()) && !operation(currentValue, var2->getUpperBound())) {
					var1->filterValue(currentValue);
					var1WasFiltered = true;
				}
			}
			delete iterator;
			if (var1WasFiltered) {
				filteredVariables.push_back(var1);
			}

			auto var2WasFiltered = false;

			iterator = var2->iterator();
			while (iterator->hasNextValue()) {
				auto currentValue = iterator->next();
				if (!operation(var1->getLowerBound(), currentValue) && !operation(var1->getUpperBound(), currentValue)) {
					var2->filterValue(currentValue);
					var2WasFiltered = true;
				}
			}
			delete iterator;
			if (var2WasFiltered) {
				filteredVariables.push_back(var2);
			}
		}

		if (var1->cardinality() == 0 || var2->cardinality() == 0) {
			satisfied = false;
		}
		timelogger.toc();
		return filteredVariables;
	}

	vector<Variable*> BinaryArithmeticConstraint::filterDomains() {
		satisfied = true;
		vector<Variable*> filteredVariables;

		if (useGPU) {
			filteredVariables = gpuFilter->filterDomainGPU();
		} else {
			// domain filtering for var1
			auto var1WasFiltered = false;
			auto iteratorV1 = var1->iterator();
			while (iteratorV1->hasNextValue()) {
				auto currentV1Value = iteratorV1->next();
				auto valueHasSupport = false;

				auto iteratorV2 = var2->iterator();
				while (iteratorV2->hasNextValue()) {
					auto currentV2Value = iteratorV2->next();
					if (operation(currentV1Value, currentV2Value)) {
						valueHasSupport = true;
						break;
					}
				}
				delete iteratorV2;
				if (!valueHasSupport) {
					var1->filterValue(currentV1Value);
					var1WasFiltered = true;
				}
			}
			delete iteratorV1;

			if (var1WasFiltered) {
				filteredVariables.push_back(var1);
			}

			if (var1->cardinality() == 0) {
				satisfied = false;
				return filteredVariables;
			}
			// domain filtering for var2
			auto var2WasFiltered = false;
			auto iteratorV2 = var2->iterator();
			while (iteratorV2->hasNextValue()) {
				auto currentV2Value = iteratorV2->next();
				auto valueHasSupport = false;

				iteratorV1 = var1->iterator();
				while (iteratorV1->hasNextValue()) {
					auto currentV1Value = iteratorV1->next();
					if (operation(currentV1Value, currentV2Value)) {
						valueHasSupport = true;
						break;
					}
				}
				delete iteratorV1;
				if (!valueHasSupport) {
					var2->filterValue(currentV2Value);
					var2WasFiltered = true;
				}
			}
			delete iteratorV2;

			if (var2WasFiltered) {
				filteredVariables.push_back(var2);
			}
		}

		if (var1->cardinality() == 0 || var2->cardinality() == 0) {
			satisfied = false;
		}

		return filteredVariables;
	}

	bool BinaryArithmeticConstraint::filterVariableDomain(Variable* varToFilter, Variable* otherVar) const {
		auto variableWasFiltered = false;

		auto iteratorV1 = varToFilter->iterator();
		while (iteratorV1->hasNextValue()) {
			auto currentV1Value = iteratorV1->next();
			auto valueHasSupport = false;

			auto iteratorV2 = otherVar->iterator();
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

	function<bool(int, int)> BinaryArithmeticConstraint::getOperation(Operator op, RelationalOperator relop) {
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
		Constraint::replaceVariable(varToReplace, replacement);
		if (var1 == varToReplace) {
			var1 = replacement;
		} else if (var2 == varToReplace) {
			var2 = replacement;
		}
	}

	Constraint* BinaryArithmeticConstraint::clone() const {
		auto clone = new BinaryArithmeticConstraint(var1, var2, rhs, MINUS, EQ);
		clone->operation = operation;
		return clone;
	}

} // namespace hydra