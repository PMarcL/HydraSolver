#include "VariableSelector.h"
#include "Variable.h"
#include <list>
#include <random>

using namespace std;

namespace hydra {

	VariableSelector::VariableSelector(Heuristic heuristic, Heuristic tieBreaker) : heuristic(heuristic), tieBreaker(tieBreaker) {
	}

	void VariableSelector::instantiateVariable(const std::vector<Variable*>& variables) const {
		useHeuristic(variables, heuristic);
	}

	void VariableSelector::useHeuristic(const std::vector<Variable*>& variables, Heuristic heuristic) const {
		switch (heuristic) {
		case SMALLEST_DOMAIN:
			smallestDomain(variables);
			break;
		case RANDOM:
			randomSelection(variables);
			break;
		default:
			randomSelection(variables);
		}
	}


	void VariableSelector::smallestDomain(const std::vector<Variable*>& variables) const {
		list<Variable*> smallestVariables;
		auto minCardinality = 0;

		for (auto variable : variables) {
			// ensure that the first and possibly only variable in the list is not instantiated
			if (variable->cardinality() > 1 && smallestVariables.empty()) {
				smallestVariables.push_back(variable);
				minCardinality = variable->cardinality();
			} else if (variable->cardinality() == minCardinality) {
				smallestVariables.push_back(variable);
			} else if (variable->cardinality() > 1 && variable->cardinality() < minCardinality) {
				smallestVariables.erase(smallestVariables.begin(), smallestVariables.end());
				smallestVariables.push_back(variable);
			}
		}

		if (smallestVariables.size() > 1) {
			useHeuristic(variables, tieBreaker);
		} else {
			smallestVariables.front()->instantiate();
		}
	}

	void VariableSelector::randomSelection(const std::vector<Variable*>& variables) {
		default_random_engine generator;
		uniform_int_distribution<size_t> distribution(0, variables.size() - 1);
		size_t index = distribution(generator);

		while (variables[index]->cardinality() == 1) {
			index = distribution(generator);
		}

		variables[index]->instantiate();
	}


} // namespace hydra