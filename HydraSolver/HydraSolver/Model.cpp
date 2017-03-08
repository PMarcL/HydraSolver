#include "Model.h"
#include "Constraint.h"
#include "Variable.h"

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

	void Model::addVariable(Variable* variable) {
		variableEnvironment.addVariable(variable);
	}

	void Model::addVariableArray(const std::vector<Variable*>& vars) {
		variableEnvironment.addVariableArray(vars);
	}

	const VariableEnvironment& Model::getEnvironment() const {
		return variableEnvironment;
	}

	string Model::getName() const {
		return name;
	}

} // namespace hydra
