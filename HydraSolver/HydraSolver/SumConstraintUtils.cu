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
	std::vector<uint8_t>* bitSetPtr) {

	uint8_t* deviceBitSetPtr;
	cudaMalloc((void**)&deviceBitSetPtr, bitSetPtr->size() * sizeof(uint8_t));
	cudaMemcpy(deviceBitSetPtr, bitSetPtr, bitSetPtr->size(), cudaMemcpyHostToDevice);

	std::cout << "sum: " << sum << " lowerBoundSum: " << lowerBoundSum << " upperBoundSum: " << upperBoundSum << std::endl;
	filterVariableKernel << < 1, nKernel >> > (
		sum,
		lowerBoundSum,
		upperBoundSum,
		originalLowerBound,
		deviceBitSetPtr
		);

	cudaMemcpy(bitSetPtr, deviceBitSetPtr, bitSetPtr->size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

}
