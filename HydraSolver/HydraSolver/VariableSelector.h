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

		void instantiateVariable(const std::vector<Variable*>& variables) const;

	private:
		void useHeuristic(const std::vector<Variable*>& variables, Heuristic heuristic) const;
		void smallestDomain(const std::vector<Variable*>& variables) const;
		static void randomSelection(const std::vector<Variable*>& variables);

		Heuristic heuristic;
		Heuristic tieBreaker;
	};

} // namespace hydra
