#include "VariableSelector.h"

namespace hydra {

	VariableSelector::VariableSelector(Heuristic heuristic, Heuristic tieBreaker) : heuristic(heuristic), tieBreaker(tieBreaker) {
	}

	void VariableSelector::instantiateVariable(const std::vector<Variable*>) const {

	}

	void VariableSelector::smallestDomain(const std::vector<Variable*>) const {

	}

	void VariableSelector::randomSelection(const std::vector<Variable*>) const {

	}


} // namespace hydra