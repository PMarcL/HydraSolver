#pragma once
#include <vector>

namespace hydra {

	class Constraint;
	class Variable;

	enum LocalConsistencyConfig {
		DEFAULT_FILTERING_ALGO,
		BOUND_CONSISTENCY_ALGO,
		DOMAIN_CONSISTENCY_ALGO
	};

	enum PropagationResult {
		LOCAL_CONSISTENCY,
		INCONSISTENT_STATE
	};

	class Propagator {
	public:
		explicit Propagator(const std::vector<Constraint*>& constraints, LocalConsistencyConfig config = DEFAULT_FILTERING_ALGO);
		~Propagator();

		PropagationResult propagate();

	private:
		std::vector<Variable*> filterConstraint(Constraint* constraint) const;

		LocalConsistencyConfig consistencyConfig;
		std::vector<Constraint*> constraints;
	};

} // namespace hydra
