#include "SumConstraintUtils.cuh"
#include "SumConstraintKernels.cuh"
#include "cuda_runtime.h"
#include "HydraException.h"
#include <iostream>

void launchFilteringKernels(int nKernel, int sum, int lowerBoundSum, int upperBoundSum, int originalLowerBound, std::vector<uint8_t>* bitSetPtr) {
	uint8_t* deviceBitSetPtr = nullptr;
	while (nKernel % 1024 != 0) nKernel++;

	auto error = cudaMalloc((void**)&deviceBitSetPtr, nKernel * sizeof(uint8_t));
	if (error != cudaSuccess) {
		throw hydra::HydraException("Cuda memory allocation error.");
	}
	auto nBlocks = nKernel / 1024;

	filterVariableKernel << < nBlocks, 1024 >> > (sum, lowerBoundSum, upperBoundSum, originalLowerBound, deviceBitSetPtr);
	error = cudaMemcpy(bitSetPtr->data(), deviceBitSetPtr, bitSetPtr->size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		throw hydra::HydraException("Cuda memory copy error");
	}
	error = cudaFree(deviceBitSetPtr);
	if (error != cudaSuccess) {
		throw hydra::HydraException("Cuda free memory error");
	}
}
