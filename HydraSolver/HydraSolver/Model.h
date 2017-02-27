#pragma once

#include <vector>
#include <string>

namespace hydra {

	class Constraint;
	class Variable;

	class Model {
	public:
		explicit Model(std::string = "Model-");
		~Model();

		void postConstraint(Constraint*);
		std::vector<Constraint*> getConstraints() const;
		void addVariable(Variable*);
		std::vector<Variable*> getVariables() const;
		std::string getName() const;

		Model(const Model&) = delete;
		Model& operator=(const Model&) = delete;
	private:

		std::string name;
		std::vector<Constraint*> constraints;
		std::vector<Variable*> variables;
	};

} // namespace hydra


