#include "VariableUtils.h"
#include "BitsetIntVariable.h"
#include "FixedIntVariable.h"
#include <string>

using namespace std;

namespace hydra {

	Variable* CreateIntVar(const string& name, int value) {
		return new FixedIntVariable(name, value);
	}

	Variable* CreateIntVar(const string& name, int lb, int ub) {
		if (lb == ub) {
			return CreateIntVar(name, lb);
		}
		return new BitsetIntVariable(name, lb, ub);
	}

	vector<Variable*> CreateIntVarArray(const string& name, size_t size, int lb, int ub) {
		vector<Variable*> variables;
		for (size_t i = 0; i < size; i++) {
			auto currentName = name + to_string(i + 1);
			variables.push_back(new BitsetIntVariable(currentName, lb, ub));
		}
		return variables;
	}

	vector<vector<Variable*> > CreateIntVarMatrix(const string& name, size_t row, size_t col, int lb, int ub) {
		vector<vector<Variable*> > variables;
		for (size_t r = 0; r < row; r++) {
			auto rowName = name + to_string(r);
			variables.push_back(CreateIntVarArray(rowName, col, lb, ub));
		}
		return variables;
	}
}