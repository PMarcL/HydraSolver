#pragma once

#include <vector>

namespace hydra {

	class Variable;

	class VariableEnvironment {
	public:
		VariableEnvironment();
		~VariableEnvironment();

		void addVariable(Variable* var);
		void addVariableArray(const std::vector<Variable*>& vars);
		const std::vector<Variable*>& getVariables() const;
		void push() const;
		void pop() const;

	private:
		std::vector<Variable*> variables;
	};

} // namespace hydra
