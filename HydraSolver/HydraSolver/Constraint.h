#pragma once

#include <vector>

namespace hydra {

	class Variable;

	class Constraint {
	public:
		Constraint();
		~Constraint();

		virtual std::vector<Variable*> getVariables() const = 0;
		virtual void filter() = 0;
		virtual void filterDomains() = 0;
		virtual void filterBounds() = 0;
	};

} // namespace hydra
