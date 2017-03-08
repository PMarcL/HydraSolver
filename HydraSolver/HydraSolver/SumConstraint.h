#pragma once

#include "Constraint.h"
#include <vector>

namespace hydra {

	class IntVariable;

	class SumConstraint :
		public Constraint {
	public:
		SumConstraint(const std::vector<IntVariable*>& var, int sum);
		~SumConstraint();

		bool containsVariable(Variable* variable) const;
		void filter() override;
		void filterDomains() override;
		void filterBounds() override;

	private:
		void CPUDomainFilteringAlgorithm();
		void CPUBoundsFilteringAlgorithm();

		std::vector<IntVariable*> variables;
		int sum;
	};

} // namespace hydra
