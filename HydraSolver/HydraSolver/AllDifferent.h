#pragma once

#include "Constraint.h"
#include <vector>

namespace hydra {

	class Variable;

	class AllDifferent : public Constraint {
	public:
		explicit AllDifferent(const std::vector<Variable*>& variables);
		~AllDifferent();

		bool containsVariable(Variable* variable) const override;
		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;
		bool isSatisfied() const override;

	private:
		std::vector<Variable*> variables;
	};

} // namespace hydra

