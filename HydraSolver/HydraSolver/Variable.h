#pragma once

#include <vector>

namespace hydra {

	class VariableObserver;

	class Variable {
	public:
		explicit Variable(const std::string& name = "Var-");
		virtual ~Variable();

		std::string getName() const;
		void notifyDomainChanged() const;

		virtual void pushCurrentState() = 0;
		virtual void popState() = 0;
		virtual int cardinality() const = 0;
		virtual void instantiate() = 0;

	protected:
		std::string name;

	private:
		std::vector<VariableObserver*> observers;
	};

} // namespace hydra
