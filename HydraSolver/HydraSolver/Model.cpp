#include "Model.h"
#include "Constraint.h"
#include "Variable.h"
#include "VariableUtils.h"

using namespace std;

namespace hydra {

	Model::Model(const string& name) : name(name) {
	}

	Model::~Model() {
		for (auto c : constraints) {
			delete c;
		}
	}

	void Model::postConstraint(Constraint* constraint) {
		constraints.push_back(constraint);
	}

	void Model::postConstraints(const std::vector<Constraint*>& constraintsList) {
		constraints.insert(constraints.end(), constraintsList.begin(), constraintsList.end());
	}

	vector<Constraint*> Model::getConstraints() const {
		return constraints;
	}

	size_t Model::getNumberOfConstraints() const {
		return constraints.size();
	}

	void Model::addVariable(Variable* variable) {
		variableEnvironment.addVariable(variable);
	}

	void Model::addVariableArray(const vector<Variable*>& vars) {
		variableEnvironment.addVariableArray(vars);
	}

	Variable* Model::createIntVar(const string& name, int value) {
		auto var = CreateIntVar(name, value);
		addVariable(var);
		return var;
	}

	Variable* Model::createIntVar(const string& name, int lb, int ub) {
		auto var = CreateIntVar(name, lb, ub);
		addVariable(var);
		return var;
	}

	vector<Variable*> Model::createIntVarArray(const string& name, size_t size, int lb, int ub) {
		auto vars = CreateIntVarArray(name, size, lb, ub);
		addVariableArray(vars);
		return vars;
	}

	vector<vector<Variable*>> Model::createIntVarMatrix(const string& name, size_t row, size_t col, int lb, int ub) {
		auto vars = CreateIntVarMatrix(name, row, col, lb, ub);
		for (auto currentRow : vars) {
			addVariableArray(currentRow);
		}
		return vars;
	}

	size_t Model::getNumberOfVariables() const {
		return variableEnvironment.getVariables().size();
	}

	void Model::pushEnvironment() const {
		variableEnvironment.push();
	}

	void Model::popEnvironment() const {
		variableEnvironment.pop();
	}

	bool Model::allVariablesAreInstantiated() const {
		return variableEnvironment.allVariablesAreInstantiated();
	}

	vector<Variable*> Model::getVariables() const {
		return variableEnvironment.getVariables();
	}

	VariableEnvironment Model::getVariableEnvironnement() const
	{
		return variableEnvironment;
	}


	string Model::getName() const {
		return name;
	}

	Model::Model(const Model& model) {
		name = model.getName();
		variableEnvironment = model.getVariableEnvironnement();
		constraints = model.getConstraints();
	}

} // namespace hydra
