#pragma once

#include <vector>
#include <unordered_set>

namespace hydra {

	class Variable;

	class Constraint {
	public:
		Constraint();
		virtual ~Constraint();

		bool containsVariable(Variable*) const;
		virtual std::vector<Variable*> filter() = 0;
		virtual std::vector<Variable*> filterDomains() = 0;
		virtual std::vector<Variable*> filterBounds() = 0;
		virtual bool isSatisfied() const = 0;
		virtual void replaceVariable(Variable* varToReplace, Variable* replacement) = 0;
		virtual Constraint* clone() const = 0;

	protected:
		Constraint(const std::vector<Variable*>& vars);

		bool useGPU;
		bool satisfied;

	private:
		std::unordered_set<Variable*> variablesSet;
	};

} // namespace hydra
