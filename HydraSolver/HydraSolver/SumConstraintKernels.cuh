#pragma once

#include "cuda_runtime.h"
#include "BitsetIntVariable.h"

__global__ void filterVariableKernel(
	int sum,
	int lowerBoundSum,
	int upperBoundSum,
	int offset,
	uint8_t* pBitset
);
