#pragma once

#include <vector>

namespace hydra {

	class Variable;

	Variable* CreateIntVar(const std::string& name, int value);
	Variable* CreateIntVar(const std::string& name, int lb, int ub);
	std::vector<Variable*> CreateIntVarArray(const std::string& name, size_t size, int lb, int ub);
	std::vector<std::vector<Variable*> > CreateIntVarMatrix(const std::string& name, size_t row, size_t col, int lb, int ub);

} // namespace hydra
