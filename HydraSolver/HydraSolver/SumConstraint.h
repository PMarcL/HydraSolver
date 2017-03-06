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

		void filter() override;
		void filterDomains() override;
		void filterBounds() override;

	private:
		void CPUdomainFilteringAlgorithm();

		std::vector<IntVariable*> variables;
		int sum;
	};

} // namespace hydra
