#pragma once

#include <vector>

namespace hydra {

	class Variable {
	public:
		explicit Variable(const std::string& name = "Var-");
		virtual ~Variable();

		std::string getName() const;

		virtual std::string getFormattedDomain() const = 0;
		virtual void pushCurrentState() = 0;
		virtual void popState() = 0;
		virtual int cardinality() const = 0;
		virtual void instantiate() = 0;

	protected:
		std::string name;
	};

} // namespace hydra
