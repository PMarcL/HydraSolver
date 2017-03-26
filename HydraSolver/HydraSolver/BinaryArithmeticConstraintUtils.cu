#include "BinaryArithmeticConstraint.h"
#include "Variable.h"
#include "BitsetIntVariable.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvfunctional>
#include <stdio.h>

using namespace std;

namespace hydra {

	nvstd::function<bool(int, int)> getOperation(Operator op, RelationalOperator relop, int rhs) {
		switch (relop) {
		case EQ:
			switch (op) {
			case PLUS:
				return[rhs]  __device__(int v1, int v2) { return v1 + v2 == rhs; };
			case MINUS:
				return[rhs]  __device__(int v1, int v2) { return v1 - v2 == rhs; };
			case MULTIPLIES:
				return[rhs]  __device__(int v1, int v2) { return v1 * v2 == rhs; };
			case DIVIDES:
				return[rhs]  __device__(int v1, int v2) { return v1 / v2 == rhs; };
			}
		case NEQ:													   
			switch (op) {
			case PLUS:
				return[rhs]  __device__(int v1, int v2) { return v1 + v2 != rhs; };
			case MINUS:
				return[rhs]  __device__(int v1, int v2) { return v1 - v2 != rhs; };
			case MULTIPLIES:
				return[rhs]  __device__(int v1, int v2) { return v1 * v2 != rhs; };
			case DIVIDES:
				return[rhs]  __device__(int v1, int v2) { return v1 / v2 != rhs; };
			}
		case GEQ:													   
			switch (op) {
			case PLUS:
				return[rhs]  __device__(int v1, int v2) { return v1 + v2 >= rhs; };
			case MINUS:
				return[rhs]  __device__(int v1, int v2) { return v1 - v2 >= rhs; };
			case MULTIPLIES:
				return[rhs]  __device__(int v1, int v2) { return v1 * v2 >= rhs; };
			case DIVIDES:
				return[rhs]  __device__(int v1, int v2) { return v1 / v2 >= rhs; };
			}
		case GT:													   
			switch (op) {
			case PLUS:
				return[rhs]  __device__(int v1, int v2) { return v1 + v2 > rhs; };
			case MINUS:
				return[rhs]  __device__(int v1, int v2) { return v1 - v2 > rhs; };
			case MULTIPLIES:
				return[rhs]  __device__(int v1, int v2) { return v1 * v2 > rhs; };
			case DIVIDES:
				return[rhs]  __device__(int v1, int v2) { return v1 / v2 > rhs; };
			}
		case LEQ:													   
			switch (op) {
			case PLUS:
				return[rhs]  __device__(int v1, int v2) { return v1 + v2 <= rhs; };
			case MINUS:
				return[rhs]  __device__(int v1, int v2) { return v1 - v2 <= rhs; };
			case MULTIPLIES:
				return[rhs]  __device__(int v1, int v2) { return v1 * v2 <= rhs; };
			case DIVIDES:
				return[rhs]  __device__(int v1, int v2) { return v1 / v2 <= rhs; };
			}
		case LT:													   
			switch (op) {
			case PLUS:
				return[rhs]  __device__(int v1, int v2) { return v1 + v2 < rhs; };
			case MINUS:
				return[rhs]  __device__(int v1, int v2) { return v1 - v2 < rhs; };
			case MULTIPLIES:
				return[rhs]  __device__(int v1, int v2) { return v1 * v2 < rhs; };
			case DIVIDES:
				return[rhs]  __device__(int v1, int v2) { return v1 / v2 < rhs; };
			}
		}
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

	template <typename Operation>
	__global__ void filterBoundDevice(Operation op, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		printf("Thread id : %d\n", threadIdx.x);
		printf("Value : %d\n", value);
		result[threadIdx.x] = op(value, *lb) || op(value, *ub);
	}

	bool filterVariableBounds(BitsetIntVariable* varToFilter, BitsetIntVariable* otherVar, nvstd::function<bool(int, int)> operation) {
		auto lb = otherVar->getLowerBound();
		int *device_lb;
		cudaMalloc((void**)&device_lb, sizeof(int));
		cudaMemcpy(device_lb, &lb, sizeof(int), cudaMemcpyHostToDevice);

		auto ub = otherVar->getUpperBound();
		int *device_ub;
		cudaMalloc((void**)&device_ub, sizeof(int));
		cudaMemcpy(device_ub, &ub, sizeof(int), cudaMemcpyHostToDevice);

		auto originalLowerBound = varToFilter->getOriginalLowerBound();
		int *device_originallb;
		cudaMalloc((void**)&device_originallb, sizeof(int));
		cudaMemcpy(device_originallb, &originalLowerBound, sizeof(int), cudaMemcpyHostToDevice);

		uint8_t *bitset_device, *bitset_host;
		auto size = varToFilter->getOriginalSize();
		bitset_host = (uint8_t *)malloc(size);
		cudaMalloc((void **)&bitset_device, size);

		filterBoundDevice<<<1, size>>>(operation, device_lb, device_ub, device_originallb, bitset_device);

		cudaMemcpy(bitset_host, bitset_device, size, cudaMemcpyDeviceToHost);

		return varToFilter->mergeBitset(bitset_host);
	}
}
