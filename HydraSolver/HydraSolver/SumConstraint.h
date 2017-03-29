#pragma once

#include "Constraint.h"
#include "BitsetIntVariable.h"

namespace hydra {

	class Variable;

	class SumConstraint :
		public Constraint {
	public:
		SumConstraint(const std::vector<Variable*>& var, int sum, bool pUseGPU = false);
		~SumConstraint();

		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;
		bool isSatisfied() const override;
		void replaceVariable(Variable* varToReplace, Variable* replacement) override;
		Constraint* clone() const override;

	private:
		std::vector<Variable*> CPUDomainFilteringAlgorithm();
		std::vector<Variable*> CPUBoundsFilteringAlgorithm();
		std::vector<BitsetIntVariable*> GPUBoundsFilteringAlgorithm();

		std::vector<Variable*> variables;
		int sum;
	};

} // namespace hydra
