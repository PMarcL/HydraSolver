#include "BinaryArithmeticConstraint.h"
#include "Variable.h"
#include "BitsetIntVariable.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>

using namespace std;

namespace hydra {

	nvstd::function<bool(int, int)> getOperation(Operator op, RelationalOperator relop, int rhs) {
		nvstd::function<int(int, int)> lhsResult;
		switch (op) {
		case PLUS:
			lhsResult = [] __device__(int v1, int v2) { return v1 + v2; };
			break;
		case MINUS:
			lhsResult = []  __device__(int v1, int v2) { return v1 - v2; };
			break;
		case MULTIPLIES:
			lhsResult = []  __device__(int v1, int v2) { return v1 * v2; };
			break;
		case DIVIDES:
			lhsResult = []  __device__(int v1, int v2) { return v1 / v2; };
			break;
		}

		nvstd::function<bool(int, int)> operation;
		switch (relop) {
		case EQ:
			operation = [lhsResult, rhs]  __device__(int v1, int v2) { return lhsResult(v1, v2) == rhs; };
			break;
		case NEQ:
			operation = [lhsResult, rhs]  __device__(int v1, int v2) { return lhsResult(v1, v2) != rhs; };
			break;
		case GEQ:
			operation = [lhsResult, rhs]  __device__(int v1, int v2) { return lhsResult(v1, v2) >= rhs; };
			break;
		case GT:
			operation = [lhsResult, rhs]  __device__(int v1, int v2) { return lhsResult(v1, v2) > rhs; };
			break;
		case LEQ:
			operation = [lhsResult, rhs]  __device__(int v1, int v2) { return lhsResult(v1, v2) <= rhs; };
			break;
		case LT:
			operation = [lhsResult, rhs]  __device__(int v1, int v2) { return lhsResult(v1, v2) < rhs; };
			break;
		}

		return operation;
	}

	bool filterVariableBounds(BitsetIntVariable* varToFilter, BitsetIntVariable* otherVar, nvstd::function<bool(int, int)> operation);

	vector<Variable*> filterBoundsGPU(BitsetIntVariable* var1, BitsetIntVariable* var2, Operator op, RelationalOperator relop, int rhs) {
		auto operation = getOperation(op, relop, rhs);
		vector<Variable*> filteredVariables;

		if (filterVariableBounds(var1, var2, operation)) {
			filteredVariables.push_back(var1);
		}

		if (var1->cardinality() == 0) {
			return filteredVariables;
		}

		if (filterVariableBounds(var2, var1, operation)) {
			filteredVariables.push_back(var2);
		}
		
		return filteredVariables;
	}

	__constant__ int lb;
	__constant__ int ub;
	__constant__ int originalLowerBound;

	template <typename Operation>
	__global__ void filterBoundDevice(Operation op, uint8_t *result) {
		int value = originalLowerBound + threadIdx.x;
		result[threadIdx.x] = op(value, lb) || op(value, ub);
	}

	bool filterVariableBounds(BitsetIntVariable* varToFilter, BitsetIntVariable* otherVar, nvstd::function<bool(int, int)> operation) {
		auto lowerbound = otherVar->getUpperBound();
		cudaMemcpyToSymbol("lb", &lowerbound, sizeof(int));

		auto upperbound = otherVar->getUpperBound();
		cudaMemcpyToSymbol("ub", &upperbound, sizeof(int));

		auto originalLowerBound = varToFilter->getOriginalLowerBound();
		cudaMemcpyToSymbol("originalLowerBound", &originalLowerBound, sizeof(int));

		uint8_t *bitset_device, *bitset_host;
		auto size = varToFilter->getOriginalSize();
		bitset_host = (uint8_t *)malloc(size);
		cudaMalloc((void **)&bitset_device, size);

		filterBoundDevice<<<1, size>>>(operation, bitset_device);

		cudaMemcpy(bitset_host, bitset_device, size, cudaMemcpyDeviceToHost);

		return varToFilter->mergeBitset(bitset_host);
	}
}
