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

	size_t Model::getNumberOfConstraints() const {
		return constraints.size();
	}

	void Model::addVariable(Variable* variable) {
		variableEnvironment.addVariable(variable);
	}

	void Model::addVariableArray(const std::vector<Variable*>& vars) {
		variableEnvironment.addVariableArray(vars);
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


	string Model::getName() const {
		return name;
	}

} // namespace hydra
