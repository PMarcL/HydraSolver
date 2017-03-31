#include "SumConstraintUtils.cuh"
#include "SumConstraintKernels.cuh"
#include "SumConstraint.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

void launchFilteringKernels(
	int nKernel,
	int sum,
	int lowerBoundSum,
	int upperBoundSum,
	int originalLowerBound,
	std::vector<bool>* bitSetPtr) {

	uint8_t* deviceBitSetPtr;
	cudaMalloc((void**)&deviceBitSetPtr, bitSetPtr->size());
	cudaMemcpy(deviceBitSetPtr, bitSetPtr, bitSetPtr->size(), cudaMemcpyHostToDevice);

	std::cout << "sum: " << sum << " lowerBoundSum: " << lowerBoundSum << " upperBoundSum: " << upperBoundSum << std::endl;
	filterVariableKernel << < 1, nKernel >> > (
		sum,
		lowerBoundSum,
		upperBoundSum,
		originalLowerBound,
		deviceBitSetPtr
		);

	cudaMemcpy(bitSetPtr, deviceBitSetPtr, bitSetPtr->size(), cudaMemcpyDeviceToHost);

}
