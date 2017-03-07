#pragma once

namespace hydra {

	class VariableObserver {
	public:
		VariableObserver();
		virtual ~VariableObserver();

		virtual void domainChanged() = 0;
	};

} // namespace hydra
