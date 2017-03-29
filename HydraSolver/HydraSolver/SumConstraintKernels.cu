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
	bool* pBitset
) {
	int value = threadIdx.x + offset;
	lowerBoundSum += value;
	upperBoundSum += value;
	pBitset[threadIdx.x] = pBitset[threadIdx.x] && !(sum < lowerBoundSum || sum > upperBoundSum);
	return;
}