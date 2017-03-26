#pragma once

#include <vector>

namespace hydra {

	class BitsetIntVariable;
	class Variable;

	std::vector<Variable*> filterBoundsGPU(BitsetIntVariable* var1, BitsetIntVariable* var2, Operator op, RelationalOperator relop, int rhs);
}