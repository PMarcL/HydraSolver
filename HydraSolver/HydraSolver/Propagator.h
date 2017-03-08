#pragma once
#include <vector>

namespace hydra {

	class Constraint;

	enum LocalConsistencyConfig {
		BOUND_CONSISTENCY,
		DOMAIN_CONSISTENCY
	};

	enum PropagationResult {
		LOCAL_CONSISTENCY,
		INCONSISTENT_STATE
	};

	class Propagator {
	public:
		explicit Propagator(const std::vector<Constraint*>& constraints, LocalConsistencyConfig config = DOMAIN_CONSISTENCY);
		~Propagator();

		PropagationResult propagate();

	private:
		LocalConsistencyConfig consistencyConfig;
		std::vector<Constraint*> constraints;
	};

} // namespace hydra
