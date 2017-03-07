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

		void instantiateVariable(const std::vector<Variable*>) const;

	private:
		void smallestDomain(const std::vector<Variable*>) const;
		void randomSelection(const std::vector<Variable*>) const;

		Heuristic heuristic;
		Heuristic tieBreaker;
	};

} // namespace hydra
