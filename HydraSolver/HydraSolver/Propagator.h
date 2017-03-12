#pragma once
#include <vector>

namespace hydra {

	class Constraint;
	class Variable;

	enum LocalConsistencyConfig {
		DEFAULT_FILTERING_ALGO,
		BOUND_CONSISTENCY,
		DOMAIN_CONSISTENCY
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
		void setLocalConsistencyConfig(LocalConsistencyConfig config);

	private:
		std::vector<Variable*> filterConstraint(Constraint* constraint) const;

		LocalConsistencyConfig consistencyConfig;
		std::vector<Constraint*> constraints;
	};

} // namespace hydra
