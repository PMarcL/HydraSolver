#pragma once

#include <string>

namespace hydra {

	class Variable {
	public:
		explicit Variable(const std::string& name = "Var-");
		virtual ~Variable();

		std::string getName() const;

		virtual void pushCurrentState() = 0;
		virtual void popState() = 0;

	private:
		std::string name;
	};

} // namespace hydra
