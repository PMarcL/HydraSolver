#pragma once

#include "Constraint.h"

namespace hydra {

	class Variable;

	class SumConstraint :
		public Constraint {
	public:
		SumConstraint(const std::vector<Variable*>& var, int sum, bool pUseGPU = false);
		~SumConstraint();

		bool containsVariable(Variable* variable) const override;
		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;
		bool isSatisfied() const override;

	private:
		std::vector<Variable*> CPUDomainFilteringAlgorithm();
		std::vector<Variable*> CPUBoundsFilteringAlgorithm();
		std::vector<Variable*> GPUBoundsFilteringAlgorithm();

		std::vector<Variable*> variables;
		int sum;
	};

} // namespace hydra
