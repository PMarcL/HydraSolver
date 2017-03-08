#pragma once

#include "Constraint.h"

namespace hydra {

	class IntVariable;

	class SumConstraint :
		public Constraint {
	public:
		SumConstraint(const std::vector<IntVariable*>& var, int sum);
		~SumConstraint();

		bool containsVariable(Variable* variable) const override;
		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;

	private:
		std::vector<Variable*> CPUDomainFilteringAlgorithm();
		std::vector<Variable*> CPUBoundsFilteringAlgorithm();

		std::vector<IntVariable*> variables;
		int sum;
	};

} // namespace hydra
