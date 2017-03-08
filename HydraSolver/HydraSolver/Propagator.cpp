#include "Propagator.h"

using namespace std;

namespace hydra {

	Propagator::Propagator(const vector<Constraint*>& constraints, LocalConsistencyConfig config)
		: consistencyConfig(config), constraints(constraints) {
	}

	Propagator::~Propagator() {
	}

	PropagationResult Propagator::propagate() {
		return INCONSISTENT_STATE;
	}


} // namespace hydra
