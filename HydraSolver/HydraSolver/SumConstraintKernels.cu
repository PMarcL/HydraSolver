#include "SumConstraintKernels.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>

__global__ void filterVariableKernel(
	int sum,
	int lowerBoundSum,
	int upperBoundSum,
	int offset,
	uint8_t* pBitset
) {
	int value = threadIdx.x + offset;
	lowerBoundSum += value;
	upperBoundSum += value;
	auto hasSupport = !(sum < lowerBoundSum || sum > upperBoundSum);
	// printf("On filtre %i , la lowerBoundSum est %i, et uperBoundSum est %i \n", value, lowerBoundSum, upperBoundSum);
	pBitset[threadIdx.x] = uint8_t(bool(pBitset[threadIdx.x]) && hasSupport);
	return;
}