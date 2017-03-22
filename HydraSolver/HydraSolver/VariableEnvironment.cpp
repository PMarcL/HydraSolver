#include "VariableEnvironment.h"
#include "Variable.h"

namespace hydra {

	VariableEnvironment::VariableEnvironment() {
	}

	VariableEnvironment::~VariableEnvironment() {
		for (auto v : variables) {
			delete v;
		}
	}

	void VariableEnvironment::addVariable(Variable* var) {
		variables.push_back(var);
	}

	void VariableEnvironment::addVariableArray(const std::vector<Variable*>& vars) {
		variables.insert(variables.end(), vars.begin(), vars.end());
	}

	const std::vector<Variable*>& VariableEnvironment::getVariables() const {
		return variables;
	}

	void VariableEnvironment::push() const {
		for (auto variable : variables) {
			variable->pushCurrentState();
		}
	}

	void VariableEnvironment::pop() const {
		for (auto variable : variables) {
			variable->popState();
		}
	}

	bool VariableEnvironment::allVariablesAreInstantiated() const {
		for (auto variable : variables) {
			if (variable->cardinality() != 1) {
				return false;
			}
		}
		return true;
	}

} // namespace hydra
