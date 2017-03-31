#pragma once

#include "VariableEnvironment.h"
#include <vector>

namespace hydra {

	class Constraint;
	class Variable;
	class Solution;

	class Model {
	public:
		Model(const Model&);
		explicit Model(const std::string& = "Model-");
		~Model();

		void postConstraint(Constraint*);
		void postConstraints(const std::vector<Constraint*>&);
		std::vector<Constraint*> getConstraints() const;
		size_t getNumberOfConstraints() const;

		void addVariable(Variable*);
		void addVariableArray(const std::vector<Variable*>& vars);
		Variable* createIntVar(const std::string& name, int value);
		Variable* createIntVar(const std::string& name, int lb, int ub);
		std::vector<Variable*> createIntVarArray(const std::string& name, size_t size, int lb, int ub);
		std::vector<std::vector<Variable*> > createIntVarMatrix(const std::string& name, size_t row, size_t col, int lb, int ub);
		void pushEnvironment() const;
		void popEnvironmentNTimes(unsigned int n) const;
		size_t getNumberOfVariables() const;
		bool allVariablesAreInstantiated() const;
		std::vector<Variable*> getVariables() const;
		std::string getName() const;

		Model& operator=(const Model&);
	private:
		void updateAttributesWithModel(const Model&);

		std::string name;
		std::vector<Constraint*> constraints;
		VariableEnvironment variableEnvironment;
	};

} // namespace hydra


