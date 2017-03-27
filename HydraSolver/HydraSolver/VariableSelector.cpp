#include "VariableSelector.h"
#include "Variable.h"
#include <random>
#include <chrono>

using namespace std;
using namespace chrono;

namespace hydra {

	VariableSelector::VariableSelector(Heuristic heuristic, Heuristic tieBreaker) : heuristic(heuristic), tieBreaker(tieBreaker) {
	}

	Variable* VariableSelector::instantiateVariable(const std::vector<Variable*>& variables) const {
		return useHeuristic(variables, heuristic);
	}

	Variable* VariableSelector::useHeuristic(const std::vector<Variable*>& variables, Heuristic heuristic) const {
		switch (heuristic) {
		case SMALLEST_DOMAIN:
			return smallestDomain(variables);
		case RANDOM:
			return randomSelection(variables);
		default:
			return randomSelection(variables);
		}
	}

	Variable* VariableSelector::smallestDomain(const std::vector<Variable*>& variables) const {
		vector<Variable*> smallestVariables;
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
			return useHeuristic(smallestVariables, tieBreaker);
		}
		smallestVariables.front()->instantiate();
		return smallestVariables.front();
	}

	Variable* VariableSelector::randomSelection(const vector<Variable*>& variables) {
		auto seed = system_clock::now().time_since_epoch().count();
		default_random_engine generator(seed);
		uniform_int_distribution<size_t> distribution(0, variables.size() - 1);
		auto index = distribution(generator);

		while (variables[index]->cardinality() == 1) {
			index = distribution(generator);
		}

		variables[index]->instantiate();
		return variables[index];
	}


} // namespace hydra