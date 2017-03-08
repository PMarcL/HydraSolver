#pragma once

namespace hydra {

	class Variable;

	class Constraint {
	public:
		Constraint();
		virtual ~Constraint();

		virtual bool containsVariable(Variable*) const = 0;
		virtual void filter() = 0;
		virtual void filterDomains() = 0;
		virtual void filterBounds() = 0;

	protected:
		bool useGPU;
	};

} // namespace hydra
