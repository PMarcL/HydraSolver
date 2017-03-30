#include "Model.h"
#include "Constraint.h"
#include "Variable.h"
#include "VariableUtils.h"
#include "Solver.h"
#include "omp.h"

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

	void Model::popEnvironmentNTimes(unsigned int n) const {
		for (auto i = 0; i < n; i++) {
			variableEnvironment.pop();
		}
	}

	bool Model::allVariablesAreInstantiated() const {
		return variableEnvironment.allVariablesAreInstantiated();
	}

	vector<Variable*> Model::getVariables() const {
		return variableEnvironment.getVariables();
	}

	string Model::getName() const {
		return name;
	}

	Model::Model(const Model& model) {
		updateAttributesWithModel(model);
	}

	Model& Model::operator=(const Model& model) {
		updateAttributesWithModel(model);
		return *this;
	}

	void Model::updateAttributesWithModel(const Model& model) {
		name = model.getName();

		auto originalModelVariables = model.getVariables();
		vector<Variable*> variablesCopy;

		for (auto var : originalModelVariables) {
			variablesCopy.push_back(var->clone());
		}

		variableEnvironment.addVariableArray(variablesCopy);

		for (auto constraint : model.getConstraints()) {
			constraints.push_back(constraint->clone());
		}

		for (auto constraint : constraints) {
			for (size_t i = 0; i < originalModelVariables.size(); i++) {
				if (constraint->containsVariable(originalModelVariables[i])) {
					constraint->replaceVariable(originalModelVariables[i], variablesCopy[i]);
				}
			}
		}
	}
} // namespace hydra
