#include "SumConstraintUtils.cuh"
#include "SumConstraintKernels.cuh"
#include "SumConstraint.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void launchFilteringKernels(
	int nKernel,
	int sum,
	int lowerBoundSum,
	int upperBoundSum,
	int originalLowerBound,
	std::vector<bool>* bitSetPtr) {

	bool* deviceBitSetPtr;
	cudaMalloc((void**)&deviceBitSetPtr, bitSetPtr->size() * sizeof(bool));
	cudaMemcpy(deviceBitSetPtr, bitSetPtr, bitSetPtr->size() * sizeof(bool), cudaMemcpyHostToDevice);

	filterVariableKernel << < 1, nKernel >> > (
		sum,
		lowerBoundSum,
		upperBoundSum,
		originalLowerBound,
		deviceBitSetPtr
		);

	cudaMemcpy(deviceBitSetPtr, bitSetPtr, bitSetPtr->size() * sizeof(bool), cudaMemcpyDeviceToHost);

}
