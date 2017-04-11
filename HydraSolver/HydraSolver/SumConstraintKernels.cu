#include "SumConstraintKernels.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void filterVariableKernel(int sum, int lowerBoundSum, int upperBoundSum, int originalLowerBound, uint8_t* pBitset) {
	int value = threadIdx.x + 1024 * blockIdx.x + originalLowerBound;
	lowerBoundSum += value;
	upperBoundSum += value;
	auto hasSupport = sum >= lowerBoundSum && sum <= upperBoundSum;
	pBitset[threadIdx.x + 1024 * blockIdx.x] = uint8_t(hasSupport);
}