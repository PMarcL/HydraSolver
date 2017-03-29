#pragma once

#include "Constraint.h"
#include <vector>

namespace hydra {

	class Variable;
	class AllDiffBoundsFilter;

	class AllDifferent : public Constraint {
	public:
		explicit AllDifferent(const std::vector<Variable*>& variables);
		~AllDifferent();

		std::vector<Variable*> filter() override;
		std::vector<Variable*> filterDomains() override;
		std::vector<Variable*> filterBounds() override;
		bool isSatisfied() const override;
		void replaceVariable(Variable* varToReplace, Variable* replacement) override;
		Constraint* clone() const override;

	private:
		std::vector<Variable*> variables;
		AllDiffBoundsFilter* boundsFilter;
	};

} // namespace hydra

