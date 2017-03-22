#pragma once
#include "Constraint.h"
#include <functional>

namespace hydra {

	class Variable;

	enum Operator {
		MINUS,
		PLUS,
		MULTIPLIES,
		DIVIDES
	};

	enum RelationalOperator {
		EQ,
		NEQ,
		GT,
		GEQ,
		LT,
		LEQ
	};

	class BinaryArithmeticConstraint : public Constraint {
	public:
		BinaryArithmeticConstraint(Variable* var1, Variable* var2, int result, Operator op, RelationalOperator relop);
		~BinaryArithmeticConstraint();

		bool containsVariable(Variable*) const override;
		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;
		bool isSatisfied() const override;
		void replaceVariable(Variable* varToReplace, Variable* replacement) override;
		Constraint* clone() const override;

	private:
		std::function<bool(int, int)> getOperation(Operator op, RelationalOperator relop);
		bool filterVariableDomain(Variable* varToFilter, Variable* otherVar) const;
		bool filterVariableBounds(Variable* varToFilter, Variable* otherVar) const;

		Variable* var1;
		Variable* var2;
		int rhs;
		std::function<bool(int, int)> operation;
	};

} // namespace hydra
