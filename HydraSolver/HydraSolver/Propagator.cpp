#include "Propagator.h"
#include "Constraint.h"
#include "Variable.h"
#include <list>
#include <algorithm>

using namespace std;

namespace hydra {

	Propagator::Propagator(const vector<Constraint*>& constraints, LocalConsistencyConfig config)
		: consistencyConfig(config), constraints(constraints) {
	}

	Propagator::~Propagator() {
	}

	PropagationResult Propagator::propagate() {
		list<Constraint*> constraintToFilter;
		list<Constraint*> filteredConstraints;
		constraintToFilter.insert(constraintToFilter.begin(), constraints.begin(), constraints.end());

		while (!constraintToFilter.empty()) {
			auto currentConstraint = constraintToFilter.front();
			constraintToFilter.pop_front();

			auto modifiedVariables = filterConstraint(currentConstraint);

			if (!currentConstraint->isSatisfied()) {
				return INCONSISTENT_STATE;
			}

			if (!modifiedVariables.empty()) {
				for (auto it = filteredConstraints.begin(); it != filteredConstraints.end(); ) {
					if (any_of(modifiedVariables.begin(), modifiedVariables.end(), [it](auto var) { return (*it)->containsVariable(var); })) {
						constraintToFilter.push_back(*it);
						filteredConstraints.erase(it++);
					} else {
						++it;
					}
				}
			}

			filteredConstraints.push_back(currentConstraint);
		}
		return LOCAL_CONSISTENCY;
	}

	void Propagator::setLocalConsistencyConfig(LocalConsistencyConfig config) {
		consistencyConfig = config;
	}

	vector<Variable*> Propagator::filterConstraint(Constraint* constraint) const {
		switch (consistencyConfig) {
		case DEFAULT_FILTERING_ALGO:
			return constraint->filter();
		case DOMAIN_CONSISTENCY:
			return constraint->filterDomains();
		case INTERVAL_CONSISTENCY:
			return constraint->filterBounds();
		default:
			return constraint->filter();
		}
	}


} // namespace hydra
