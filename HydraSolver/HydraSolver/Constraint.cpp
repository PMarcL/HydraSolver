#include "Constraint.h"

using namespace std;

namespace hydra {

	Constraint::Constraint() : useGPU(false), satisfied(false) {
	}

	Constraint::Constraint(const std::vector<Variable*>& vars) : useGPU(false), satisfied(false),
		variablesSet(unordered_set<Variable*>(vars.begin(), vars.end())) {

	}

	Constraint::~Constraint() {
	}

	void Constraint::setGPUFilteringActive() {
		useGPU = true;
	}

	bool Constraint::containsVariable(Variable* var) const {
		return variablesSet.find(var) != variablesSet.end();
	}

	void Constraint::replaceVariable(Variable* varToReplace, Variable* replacement) {
		variablesSet.erase(varToReplace);
		variablesSet.insert(replacement);
	}

} // namespace hydra
