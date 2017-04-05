#include "SumConstraintUtils.cuh"
#include "SumConstraintKernels.cuh"
#include "cuda_runtime.h"

void launchFilteringKernels(int nKernel, int sum, int lowerBoundSum, int upperBoundSum, int originalLowerBound, std::vector<uint8_t>* bitSetPtr) {
	uint8_t* deviceBitSetPtr;
	cudaMalloc((void**)&deviceBitSetPtr, bitSetPtr->size() * sizeof(uint8_t));

	filterVariableKernel << < 1, nKernel >> > (sum, lowerBoundSum, upperBoundSum, originalLowerBound, deviceBitSetPtr);

	cudaMemcpy(bitSetPtr->data(), deviceBitSetPtr, bitSetPtr->size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}
