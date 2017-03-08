#pragma once

#include <vector>

namespace hydra {

	class Variable;

	enum Heuristic {
		SMALLEST_DOMAIN,
		RANDOM
	};

	class VariableSelector {
	public:
		explicit VariableSelector(Heuristic heuristic = SMALLEST_DOMAIN, Heuristic tieBreaker = RANDOM);

		Variable* instantiateVariable(const std::vector<Variable*>& variables) const;

	private:
		Variable* useHeuristic(const std::vector<Variable*>& variables, Heuristic heuristic) const;
		Variable* smallestDomain(const std::vector<Variable*>& variables) const;
		static Variable* randomSelection(const std::vector<Variable*>& variables);

		Heuristic heuristic;
		Heuristic tieBreaker;
	};

} // namespace hydra
