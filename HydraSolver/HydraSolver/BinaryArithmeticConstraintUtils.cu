#include "BinaryArithmeticConstraint.h"
#include "Variable.h"
#include "BitsetIntVariable.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

using namespace std;

namespace hydra {

	bool filterVariableBounds(BitsetIntVariable* varToFilter, BitsetIntVariable* otherVar, Operator op, RelationalOperator relop, int rhs);

	vector<Variable*> filterBoundsGPU(BitsetIntVariable* var1, BitsetIntVariable* var2, Operator op, RelationalOperator relop, int rhs) {
		vector<Variable*> filteredVariables;

		if (filterVariableBounds(var1, var2, op, relop, rhs)) {
			filteredVariables.push_back(var1);
		}

		if (var1->cardinality() == 0) {
			return filteredVariables;
		}

		if (filterVariableBounds(var2, var1, op, relop, rhs)) {
			filteredVariables.push_back(var2);
		}
		
		return filteredVariables;
	}

	__global__ void filterBoundPLUS_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value + *lb == *rhs) || (value + *ub == *rhs);
	}

	__global__ void filterBoundPLUS_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value + *lb != *rhs) || (value + *ub != *rhs);
	}

	__global__ void filterBoundPLUS_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value + *lb >= *rhs) || (value + *ub >= *rhs);
	}

	__global__ void filterBoundPLUS_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value + *lb > *rhs) || (value + *ub > *rhs);
	}

	__global__ void filterBoundPLUS_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value + *lb <= *rhs) || (value + *ub <= *rhs);
	}

	__global__ void filterBoundPLUS_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value + *lb < *rhs) || (value + *ub < *rhs);
	}

	__global__ void filterBoundMINUS_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value - *lb == *rhs) || (value - *ub == *rhs);
	}

	__global__ void filterBoundMINUS_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value - *lb != *rhs) || (value - *ub != *rhs);
	}

	__global__ void filterBoundMINUS_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value - *lb >= *rhs) || (value - *ub >= *rhs);
	}

	__global__ void filterBoundMINUS_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value - *lb > *rhs) || (value - *ub > *rhs);
	}

	__global__ void filterBoundMINUS_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value - *lb <= *rhs) || (value - *ub <= *rhs);
	}

	__global__ void filterBoundMINUS_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value - *lb < *rhs) || (value - *ub < *rhs);
	}

	__global__ void filterBoundMULTIPLIES_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value * *lb == *rhs) || (value * *ub == *rhs);
	}

	__global__ void filterBoundMULTIPLIES_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value * *lb != *rhs) || (value * *ub != *rhs);
	}

	__global__ void filterBoundMULTIPLIES_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value * *lb >= *rhs) || (value * *ub >= *rhs);
	}

	__global__ void filterBoundMULTIPLIES_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value * *lb > *rhs) || (value * *ub > *rhs);
	}

	__global__ void filterBoundMULTIPLIES_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value * *lb <= *rhs) || (value * *ub <= *rhs);
	}

	__global__ void filterBoundMULTIPLIES_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value * *lb < *rhs) || (value * *ub < *rhs);
	}

	__global__ void filterBoundDIVIDES_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value / *lb == *rhs) || (value / *ub == *rhs);
	}

	__global__ void filterBoundDIVIDES_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value / *lb != *rhs) || (value / *ub != *rhs);
	}

	__global__ void filterBoundDIVIDES_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value / *lb >= *rhs) || (value / *ub >= *rhs);
	}

	__global__ void filterBoundDIVIDES_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value / *lb > *rhs) || (value / *ub > *rhs);
	}

	__global__ void filterBoundDIVIDES_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value / *lb <= *rhs) || (value / *ub <= *rhs);
	}

	__global__ void filterBoundDIVIDES_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
		int value = *originalLowerBound + threadIdx.x;
		result[threadIdx.x] = (value / *lb < *rhs) || (value / *ub < *rhs);
	}

	bool filterVariableBounds(BitsetIntVariable* varToFilter, BitsetIntVariable* otherVar, Operator op, RelationalOperator relop, int rhs) {
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

		int *device_rhs;
		cudaMalloc((void**)&device_rhs, sizeof(int));
		cudaMemcpy(device_rhs, &rhs, sizeof(int), cudaMemcpyHostToDevice);

		uint8_t *bitset_device, *bitset_host;
		auto size = varToFilter->getOriginalSize();
		bitset_host = (uint8_t *)malloc(size);
		cudaMalloc((void **)&bitset_device, size);

		switch (op) {
		case PLUS:
			switch (relop) {
			case EQ:
				filterBoundPLUS_EQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case NEQ:
				filterBoundPLUS_NEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GEQ:
				filterBoundPLUS_GEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GT:
				filterBoundPLUS_GT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LEQ:
				filterBoundPLUS_LEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LT:
				filterBoundPLUS_LT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			}
			break;
		case MINUS:
			switch (relop) {
			case EQ:
				filterBoundMINUS_EQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case NEQ:
				filterBoundMINUS_NEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GEQ:
				filterBoundMINUS_GEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GT:
				filterBoundMINUS_GT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LEQ:
				filterBoundMINUS_LEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LT:
				filterBoundMINUS_LT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			}
			break;
		case MULTIPLIES:
			switch (relop) {
			case EQ:
				filterBoundMULTIPLIES_EQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case NEQ:
				filterBoundMULTIPLIES_NEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GEQ:
				filterBoundMULTIPLIES_GEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GT:
				filterBoundMULTIPLIES_GT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LEQ:
				filterBoundMULTIPLIES_LEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LT:
				filterBoundMULTIPLIES_LT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			}
			break;
		case DIVIDES:
			switch (relop) {
			case EQ:
				filterBoundDIVIDES_EQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case NEQ:
				filterBoundDIVIDES_NEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GEQ:
				filterBoundDIVIDES_GEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case GT:
				filterBoundDIVIDES_GT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LEQ:
				filterBoundDIVIDES_LEQ << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			case LT:
				filterBoundDIVIDES_LT << <1, size >> > (device_rhs, device_lb, device_ub, device_originallb, bitset_device);
				break;
			}
			break;
		}

		cudaMemcpy(bitset_host, bitset_device, size, cudaMemcpyDeviceToHost);

		return varToFilter->mergeBitset(bitset_host);
	}
}
