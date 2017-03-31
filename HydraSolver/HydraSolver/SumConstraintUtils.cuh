#pragma once

#include <vector>

void launchFilteringKernels(
	int nKernel,
	int sum,
	int lowerBoundSum,
	int upperBoundSum,
	int originalLowerBound,
	std::vector<uint8_t>* bitSetPtr);