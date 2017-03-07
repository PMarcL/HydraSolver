#pragma once

#include "Constraint.h"
#include <vector>

namespace hydra {

	class IntVariable;

	class AllDifferent : public Constraint {
	public:
		explicit AllDifferent(const std::vector<IntVariable*>& variables);
		~AllDifferent();

		void filter() override;
		void filterDomains() override;
		void filterBounds() override;

	private:
		std::vector<IntVariable*> variables;
	};

} // namespace hydra

