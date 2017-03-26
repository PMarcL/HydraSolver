#include "BinaryArithmeticConstraint.h"
#include "Variable.h"
#include "BitsetIntVariable.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <functional>

using namespace std;

namespace hydra {

	function<bool(int, int)> getOperation(Operator op, RelationalOperator relop, int rhs) {
		function<int(int, int)> lhsResult;
		switch (op) {
		case PLUS:
			lhsResult = [] __host__ __device__(int v1, int v2) { return v1 + v2; };
			break;
		case MINUS:
			lhsResult = [] __host__ __device__(int v1, int v2) { return v1 - v2; };
			break;
		case MULTIPLIES:
			lhsResult = [] __host__ __device__(int v1, int v2) { return v1 * v2; };
			break;
		case DIVIDES:
			lhsResult = [] __host__ __device__(int v1, int v2) { return v1 / v2; };
			break;
		}

		function<bool(int, int)> operation;
		switch (relop) {
		case EQ:
			operation = [lhsResult, rhs] __host__ __device__(int v1, int v2) { return lhsResult(v1, v2) == rhs; };
			break;
		case NEQ:
			operation = [lhsResult, rhs] __host__ __device__(int v1, int v2) { return lhsResult(v1, v2) != rhs; };
			break;
		case GEQ:
			operation = [lhsResult, rhs] __host__ __device__(int v1, int v2) { return lhsResult(v1, v2) >= rhs; };
			break;
		case GT:
			operation = [lhsResult, rhs] __host__ __device__(int v1, int v2) { return lhsResult(v1, v2) > rhs; };
			break;
		case LEQ:
			operation = [lhsResult, rhs] __host__ __device__(int v1, int v2) { return lhsResult(v1, v2) <= rhs; };
			break;
		case LT:
			operation = [lhsResult, rhs] __host__ __device__(int v1, int v2) { return lhsResult(v1, v2) < rhs; };
			break;
		}

		return operation;
	}

	vector<Variable*> filterBoundsGPU(BitsetIntVariable* var1, BitsetIntVariable* var2, Operator op, RelationalOperator relop, int rhs) {
		auto operation = getOperation(op, relop, rhs);
		return vector<Variable*>();
	}

}