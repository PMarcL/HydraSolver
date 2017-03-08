#pragma once

#include "Constraint.h"
#include <vector>

namespace hydra {

	class IntVariable;

	class AllDifferent : public Constraint {
	public:
		explicit AllDifferent(const std::vector<IntVariable*>& variables);
		~AllDifferent();

		bool containsVariable(Variable* variable) const override;
		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;
		bool isSatisfied() const override;

	private:
		std::vector<IntVariable*> variables;
	};

} // namespace hydra

