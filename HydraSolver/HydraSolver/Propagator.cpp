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
		constraintToFilter.insert(constraintToFilter.begin(), constraints.begin(), constraints.end());
		while (!constraintToFilter.empty()) {
			auto currentConstraint = constraintToFilter.front();
			constraintToFilter.pop_front();

			auto modifiedVariables = filterConstraint(currentConstraint);

			if (!currentConstraint->isSatisfied()) {
				return INCONSISTENT_STATE;
			}

			for (auto constraint : constraints) {
				if (constraint != currentConstraint && any_of(modifiedVariables.begin(), modifiedVariables.end(),
					[constraint](auto var) { return constraint->containsVariable(var); })) {
					constraintToFilter.push_back(constraint);
				}
			}
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
		case BOUND_CONSISTENCY:
			return constraint->filterBounds();
		default:
			return constraint->filter();
		}
	}


} // namespace hydra
