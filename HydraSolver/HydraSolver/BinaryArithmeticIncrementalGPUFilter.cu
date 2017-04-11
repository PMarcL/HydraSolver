#include "BinaryArithmeticIncrementalGPUFilter.cuh"
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

		unsigned int size = var1->getOriginalSize();
		cudaMalloc((void**)&deviceVar1_size, sizeof(unsigned int));
		cudaMemcpy(deviceVar1_size, &size, sizeof(unsigned int), cudaMemcpyHostToDevice);

		size = var2->getOriginalSize();
		cudaMalloc((void**)&deviceVar2_size, sizeof(unsigned int));
		cudaMemcpy(deviceVar2_size, &size, sizeof(unsigned int), cudaMemcpyHostToDevice);

		auto sizeVar1 = var1->getOriginalSize();
		bitset_host_var1 = (uint8_t *)malloc(sizeVar1);
		cudaMalloc((void **)&bitset_device_var1, sizeVar1);

		auto sizeVar2 = var2->getOriginalSize();
		bitset_host_var2 = (uint8_t *)malloc(sizeVar2);
		cudaMalloc((void **)&bitset_device_var2, sizeVar2);

		cudaMalloc((void**)&bitset_matrix, sizeVar1 * sizeVar2 * sizeof(uint8_t));
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
		cudaFree(bitset_matrix);
		cudaFree(deviceVar1_size);
		cudaFree(deviceVar2_size);
		free(bitset_host_var1);
		free(bitset_host_var2);
	}

	vector<Variable*> BinaryArithmeticIncrementalGPUFilter::filterBoundsGPU() {
		updateVar2DeviceAttributes();
		vector<Variable*> filteredVariables;

		auto valueIsFirst_host = true;
		bool *valueIsFirst;
		cudaMalloc((void**)&valueIsFirst, sizeof(bool));
		cudaMemcpy(valueIsFirst, &valueIsFirst_host, sizeof(bool), cudaMemcpyHostToDevice);
		if (filterVariableBounds(var1, deviceVar2_lb, deviceVar2_ub, deviceVar1Original_lb, bitset_device_var1, bitset_host_var1, valueIsFirst)) {
			filteredVariables.push_back(var1);
		}

		if (var1->cardinality() == 0) {
			return filteredVariables;
		}

		updateVar1DeviceAttributes();
		valueIsFirst_host = false;
		cudaMemcpy(valueIsFirst, &valueIsFirst_host, sizeof(bool), cudaMemcpyHostToDevice);
		if (filterVariableBounds(var2, deviceVar1_lb, deviceVar1_ub, deviceVar2Original_lb, bitset_device_var2, bitset_host_var2, valueIsFirst)) {
			filteredVariables.push_back(var2);
		}

		cudaFree(valueIsFirst);
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
		uint8_t *bitset_host, bool *valueIsFirst) const {
		unsigned int size = var->getOriginalSize();
		
		switch (op) {
		case PLUS:
			switch (relop) {
			case EQ:
				filterBoundPLUS_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case NEQ:
				filterBoundPLUS_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GEQ:
				filterBoundPLUS_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GT:
				filterBoundPLUS_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LEQ:
				filterBoundPLUS_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LT:
				filterBoundPLUS_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			}
			break;
		case MINUS:
			switch (relop) {
			case EQ:
				filterBoundMINUS_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case NEQ:
				filterBoundMINUS_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GEQ:
				filterBoundMINUS_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GT:
				filterBoundMINUS_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LEQ:
				filterBoundMINUS_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LT:
				filterBoundMINUS_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			}
			break;
		case MULTIPLIES:
			switch (relop) {
			case EQ:
				filterBoundMULTIPLIES_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case NEQ:
				filterBoundMULTIPLIES_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GEQ:
				filterBoundMULTIPLIES_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GT:
				filterBoundMULTIPLIES_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LEQ:
				filterBoundMULTIPLIES_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LT:
				filterBoundMULTIPLIES_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			}
			break;
		case DIVIDES:
			switch (relop) {
			case EQ:
				filterBoundDIVIDES_EQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case NEQ:
				filterBoundDIVIDES_NEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GEQ:
				filterBoundDIVIDES_GEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case GT:
				filterBoundDIVIDES_GT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LEQ:
				filterBoundDIVIDES_LEQ << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			case LT:
				filterBoundDIVIDES_LT << <1, size >> > (device_rhs, lb, ub, originalLowerBound, bitset_device, valueIsFirst);
				break;
			}
			break;
		}

		cudaMemcpy(bitset_host, bitset_device, size, cudaMemcpyDeviceToHost);

		return var->mergeBitset(bitset_host);
	}

	vector<Variable*> BinaryArithmeticIncrementalGPUFilter::filterDomainGPU() const {
		vector<Variable*> filteredVariables;

		cudaMemcpy(bitset_device_var2, var2->getBitset().data(), var2->getOriginalSize() * sizeof(uint8_t), cudaMemcpyHostToDevice);
		cudaMemcpy(bitset_device_var1, var1->getBitset().data(), var1->getOriginalSize() * sizeof(uint8_t), cudaMemcpyHostToDevice);
		this->filterVariableDomain();
		
		if (var1->mergeBitset(bitset_host_var1)) {
			filteredVariables.push_back(var1);
		}

		if (var2->mergeBitset(bitset_host_var2)) {
			filteredVariables.push_back(var2);
		}

		return filteredVariables;
	}


	void BinaryArithmeticIncrementalGPUFilter::filterVariableDomain() const {
		unsigned int sizeVar1 = var1->getOriginalSize();
		unsigned int sizeVar2 = var2->getOriginalSize();
		dim3 dimBlock(sizeVar2, sizeVar1);

		switch (op) {
		case PLUS:
			switch (relop) {
			case EQ:
				filterDomainPLUS_EQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case NEQ:
				filterDomainPLUS_NEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GEQ:
				filterDomainPLUS_GEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GT:
				filterDomainPLUS_GT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LEQ:
				filterDomainPLUS_LEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LT:
				filterDomainPLUS_LT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			}
			break;
		case MINUS:
			switch (relop) {
			case EQ:
				filterDomainMINUS_EQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case NEQ:
				filterDomainMINUS_NEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GEQ:
				filterDomainMINUS_GEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GT:
				filterDomainMINUS_GT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LEQ:
				filterDomainMINUS_LEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LT:
				filterDomainMINUS_LT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			}
			break;
		case MULTIPLIES:
			switch (relop) {
			case EQ:
				filterDomainMULTIPLIES_EQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case NEQ:
				filterDomainMULTIPLIES_NEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GEQ:
				filterDomainMULTIPLIES_GEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GT:
				filterDomainMULTIPLIES_GT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LEQ:
				filterDomainMULTIPLIES_LEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LT:
				filterDomainMULTIPLIES_LT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			}
			break;
		case DIVIDES:
			switch (relop) {
			case EQ:
				filterDomainDIVIDES_EQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case NEQ:
				filterDomainDIVIDES_NEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GEQ:
				filterDomainDIVIDES_GEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case GT:
				filterDomainDIVIDES_GT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LEQ:
				filterDomainDIVIDES_LEQ << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			case LT:
				filterDomainDIVIDES_LT << <1, dimBlock >> > (device_rhs, deviceVar1Original_lb, deviceVar2Original_lb, bitset_device_var1, bitset_device_var2, bitset_matrix);
				break;
			}
			break;
		}
		
		sumMatrixRows << <1, sizeVar1 >> > (bitset_matrix, sizeVar2, bitset_device_var1);
		sumMatrixCols << <1, sizeVar2 >> > (bitset_matrix, sizeVar2, sizeVar1, bitset_device_var2);

		cudaMemcpy(bitset_host_var1, bitset_device_var1, sizeVar1, cudaMemcpyDeviceToHost);
		cudaMemcpy(bitset_host_var2, bitset_device_var2, sizeVar2, cudaMemcpyDeviceToHost);
	}

}
