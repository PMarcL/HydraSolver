#include "Model.h"
#include "Constraint.h"
#include "Variable.h"

using namespace std;

namespace hydra {

	Model::Model(string name) : name(name) {
	}

	Model::~Model() {
		for (auto c : constraints) {
			delete c;
		}

		for (auto v : variables) {
			delete v;
		}
	}

	void Model::postConstraint(Constraint* constraint) {
		constraints.push_back(constraint);
	}

	vector<Constraint*> Model::getConstraints() const {
		return constraints;
	}

	void Model::addVariable(Variable* variable) {
		variables.push_back(variable);
	}

	vector<Variable*> Model::getVariables() const {
		return variables;
	}

	string Model::getName() const {
		return name;
	}

} // namespace hydra