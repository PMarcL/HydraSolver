#pragma once

#include "VariableEnvironment.h"
#include <vector>

namespace hydra {

	class Constraint;
	class Variable;

	class Model {
	public:
		explicit Model(const std::string& = "Model-");
		~Model();

		void postConstraint(Constraint*);
		void postConstraints(const std::vector<Constraint*>&);
		std::vector<Constraint*> getConstraints() const;
		void addVariable(Variable*);
		std::vector<Variable*> getVariables() const;
		std::string getName() const;

		Model(const Model&) = delete;
		Model& operator=(const Model&) = delete;
	private:

		std::string name;
		std::vector<Constraint*> constraints;
		VariableEnvironment variableEnvironment;
	};

} // namespace hydra


