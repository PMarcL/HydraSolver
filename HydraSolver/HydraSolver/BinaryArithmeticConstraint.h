#pragma once

#include "Constraint.h"
#include "BinaryArithmeticConstraintUtils.cuh"
#include <functional>

namespace hydra {

	class BinaryArithmeticConstraint : public Constraint {
	public:
		BinaryArithmeticConstraint(Variable* var1, Variable* var2, int result, Operator op, RelationalOperator relop);
		~BinaryArithmeticConstraint();

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
		Operator op;
		RelationalOperator relop;
		std::function<bool(int, int)> operation;
		BinaryArithmeticIncrementalGPUFilter *gpuFilter;
	};

} // namespace hydra
