#include "BinaryArithmeticConstraintUtils.cuh"
#include "BinaryArithmeticConstraint.h"
#include "Variable.h"
#include "BitsetIntVariable.h"
#include "cuda_runtime.h"
#include "BinaryArithmeticKernels.cuh"

using namespace std;

namespace hydra {

	BinaryArithmeticIncrementalGPUFilter::BinaryArithmeticIncrementalGPUFilter(BitsetIntVariable* var1, BitsetIntVariable* var2,
		Operator op, RelationalOperator relop, int rhs) : var1(var1), var2(var2), op(op), relop(relop), rhs(rhs), var1_lb(var1->getLowerBound()),
		var1_ub(var1->getUpperBound()), var2_lb(var2->getLowerBound()), var2_ub(var2->getUpperBound()) {
		cudaMalloc((void**)&device_rhs, sizeof(int));
		cudaMemcpy(device_rhs, &rhs, sizeof(int), cudaMemcpyHostToDevice);

		auto temp = var1->getOriginalLowerBound();
		cudaMalloc((void**)&deviceVar1Original_lb, sizeof(int));
		cudaMemcpy(deviceVar1Original_lb, &temp, sizeof(int), cudaMemcpyHostToDevice);

		temp = var2->getOriginalLowerBound();
		cudaMalloc((void**)&deviceVar2Original_lb, sizeof(int));
		cudaMemcpy(deviceVar2Original_lb, &temp, sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&deviceVar1_lb, sizeof(int));
		cudaMemcpy(deviceVar1_lb, &var1_lb, sizeof(int), cudaMemcpyHostToDevice);
		
		cudaMalloc((void**)&deviceVar2_lb, sizeof(int));
		cudaMemcpy(deviceVar2_lb, &var2_lb, sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&deviceVar1_ub, sizeof(int));
		cudaMemcpy(deviceVar1_ub, &var1_ub, sizeof(int), cudaMemcpyHostToDevice);
		
		cudaMalloc((void**)&deviceVar2_ub, sizeof(int));
		cudaMemcpy(deviceVar2_ub, &var2_ub, sizeof(int), cudaMemcpyHostToDevice);

		auto size = var1->getOriginalSize();
		bitset_host_var1 = (uint8_t *)malloc(size);
		cudaMalloc((void **)&bitset_device_var1, size);

		size = var2->getOriginalSize();
		bitset_host_var2 = (uint8_t *)malloc(size);
		cudaMalloc((void **)&bitset_device_var2, size);
	}

	BinaryArithmeticIncrementalGPUFilter::~BinaryArithmeticIncrementalGPUFilter() {
		cudaFree(deviceVar1_lb);
		cudaFree(deviceVar2_lb);
		cudaFree(deviceVar1_ub);
		cudaFree(deviceVar2_ub);
		cudaFree(device_rhs);
		cudaFree(deviceVar1Original_lb);
		cudaFree(deviceVar2Original_lb);
		cudaFree(bitset_device_var1);
		cudaFree(bitset_device_var2);
		free(bitset_host_var1);
		free(bitset_host_var2);
	}

	vector<Variable*> BinaryArithmeticIncrementalGPUFilter::filterBoundsGPU() {
		updateVar2DeviceAttributes();
		vector<Variable*> filteredVariables;

		if (filterVariableBounds(var1, deviceVar2_lb, deviceVar2_ub, deviceVar1Original_lb, bitset_device_var1, bitset_host_var1)) {
			filteredVariables.push_back(var1);
		}

		if (var1->cardinality() == 0) {
			return filteredVariables;
		}

		updateVar1DeviceAttributes();
		if (filterVariableBounds(var2, deviceVar1_lb, deviceVar1_ub, deviceVar2Original_lb, bitset_device_var2, bitset_host_var2)) {
			filteredVariables.push_back(var2);
		}

		return filteredVariables;
	}

	void BinaryArithmeticIncrementalGPUFilter::updateVar1DeviceAttributes() {
		if (var1_lb != var1->getLowerBound()) {
			var1_lb = var1->getLowerBound();
			cudaMemcpy(deviceVar1_lb, &var1_lb, sizeof(int), cudaMemcpyHostToDevice);
		}
		if (var1_ub != var1->getUpperBound()) {
			var1_ub = var1->getUpperBound();
			cudaMemcpy(deviceVar1_ub, &var1_ub, sizeof(int), cudaMemcpyHostToDevice);
		}
	}
	
	void BinaryArithmeticIncrementalGPUFilter::updateVar2DeviceAttributes() {
		if (var2_lb != var2->getLowerBound()) {
			var2_lb = var2->getLowerBound();
			cudaMemcpy(deviceVar2_lb, &var2_lb, sizeof(int), cudaMemcpyHostToDevice);
		}
		if (var2_ub != var2->getUpperBound()) {
			var2_ub = var2->getUpperBound();
			cudaMemcpy(deviceVar2_ub, &var2_ub, sizeof(int), cudaMemcpyHostToDevice);
		}
	}

	bool BinaryArithmeticIncrementalGPUFilter::filterVariableBounds(BitsetIntVariable* var, int *lb, int *ub, int *originalLowerBound, uint8_t *bitset_device,
		uint8_t *bitset_host) const {
		unsigned int size = var->getOriginalSize();
		switch (op) {
		case PLUS:
			switch (relop) {
			case EQ:
				filterBoundPLUS_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case NEQ:
				filterBoundPLUS_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GEQ:
				filterBoundPLUS_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GT:
				filterBoundPLUS_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LEQ:
				filterBoundPLUS_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LT:
				filterBoundPLUS_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			}
			break;
		case MINUS:
			switch (relop) {
			case EQ:
				filterBoundMINUS_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case NEQ:
				filterBoundMINUS_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GEQ:
				filterBoundMINUS_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GT:
				filterBoundMINUS_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LEQ:
				filterBoundMINUS_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LT:
				filterBoundMINUS_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			}
			break;
		case MULTIPLIES:
			switch (relop) {
			case EQ:
				filterBoundMULTIPLIES_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case NEQ:
				filterBoundMULTIPLIES_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GEQ:
				filterBoundMULTIPLIES_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GT:
				filterBoundMULTIPLIES_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LEQ:
				filterBoundMULTIPLIES_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LT:
				filterBoundMULTIPLIES_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			}
			break;
		case DIVIDES:
			switch (relop) {
			case EQ:
				filterBoundDIVIDES_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case NEQ:
				filterBoundDIVIDES_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GEQ:
				filterBoundDIVIDES_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case GT:
				filterBoundDIVIDES_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LEQ:
				filterBoundDIVIDES_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			case LT:
				filterBoundDIVIDES_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device);
				break;
			}
			break;
		}

		cudaMemcpy(bitset_host, bitset_device, size, cudaMemcpyDeviceToHost);

		return var->mergeBitset(bitset_host);
	}
}
