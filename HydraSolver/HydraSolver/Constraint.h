#pragma once

#include <vector>

namespace hydra {

	class Variable;

	class Constraint {
	public:
		Constraint();
		virtual ~Constraint();

		virtual bool containsVariable(Variable*) const = 0;
		virtual std::vector<Variable*> filter() = 0;
		virtual std::vector<Variable*> filterDomains() = 0;
		virtual std::vector<Variable*> filterBounds() = 0;

	protected:
		bool useGPU;
	};

} // namespace hydra
