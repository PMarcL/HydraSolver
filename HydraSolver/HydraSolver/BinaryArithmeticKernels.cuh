#pragma once

#include <cstdint>
#include "device_launch_parameters.h"

// Kernel for Bound filtering

__global__ void filterBoundPLUS_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value + *lb == *rhs) || (value + *ub == *rhs);
	} else {
		result[threadIdx.x] = (*lb + value == *rhs) || (*ub + value == *rhs);
	}
}

__global__ void filterBoundPLUS_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value + *lb != *rhs) || (value + *ub != *rhs);
	} else {
		result[threadIdx.x] = (*lb + value != *rhs) || (*ub + value != *rhs);
	}
}

__global__ void filterBoundPLUS_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value + *lb >= *rhs) || (value + *ub >= *rhs);
	} else {
		result[threadIdx.x] = (*lb + value >= *rhs) || (*ub + value >= *rhs);
	}
}

__global__ void filterBoundPLUS_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value + *lb > *rhs) || (value + *ub > *rhs);
	} else {
		result[threadIdx.x] = (*lb + value > *rhs) || (*ub + value > *rhs);
	}
}

__global__ void filterBoundPLUS_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value + *lb <= *rhs) || (value + *ub <= *rhs);
	} else {
		result[threadIdx.x] = (*lb + value <= *rhs) || (*ub + value <= *rhs);
	}
}

__global__ void filterBoundPLUS_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value + *lb < *rhs) || (value + *ub < *rhs);
	} else {
		result[threadIdx.x] = (*lb + value < *rhs) || (*ub + value < *rhs);
	}
}

__global__ void filterBoundMINUS_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value - *lb == *rhs) || (value - *ub == *rhs);
	} else {
		result[threadIdx.x] = (*lb - value == *rhs) || (*ub - value == *rhs);
	}
}

__global__ void filterBoundMINUS_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value - *lb != *rhs) || (value - *ub != *rhs);
	} else {
		result[threadIdx.x] = (*lb - value != *rhs) || (*ub - value != *rhs);
	}
}

__global__ void filterBoundMINUS_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value - *lb >= *rhs) || (value - *ub >= *rhs);
	} else {
		result[threadIdx.x] = (*lb - value >= *rhs) || (*ub - value >= *rhs);
	}
}

__global__ void filterBoundMINUS_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value - *lb > *rhs) || (value - *ub > *rhs);
	} else {
		result[threadIdx.x] = (*lb - value > *rhs) || (*ub - value > *rhs);
	}
}

__global__ void filterBoundMINUS_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value - *lb <= *rhs) || (value - *ub <= *rhs);
	} else {
		result[threadIdx.x] = (*lb - value <= *rhs) || (*ub - value <= *rhs);
	}
}

__global__ void filterBoundMINUS_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value - *lb < *rhs) || (value - *ub < *rhs);
	} else {
		result[threadIdx.x] = (*lb - value < *rhs) || (*ub - value < *rhs);
	}
}

__global__ void filterBoundMULTIPLIES_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value * *lb == *rhs) || (value * *ub == *rhs);
	} else {
		result[threadIdx.x] = (*lb * value == *rhs) || (*ub * value == *rhs);
	}
}

__global__ void filterBoundMULTIPLIES_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value * *lb != *rhs) || (value * *ub != *rhs);
	} else {
		result[threadIdx.x] = (*lb * value != *rhs) || (*ub * value != *rhs);
	}
}

__global__ void filterBoundMULTIPLIES_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value * *lb >= *rhs) || (value * *ub >= *rhs);
	} else {
		result[threadIdx.x] = (*lb * value >= *rhs) || (*ub * value >= *rhs);
	}
}

__global__ void filterBoundMULTIPLIES_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value * *lb > *rhs) || (value * *ub > *rhs);
	} else {
		result[threadIdx.x] = (*lb * value > *rhs) || (*ub * value > *rhs);
	}
}

__global__ void filterBoundMULTIPLIES_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value * *lb <= *rhs) || (value * *ub <= *rhs);
	} else {
		result[threadIdx.x] = (*lb * value <= *rhs) || (*ub * value <= *rhs);
	}
}

__global__ void filterBoundMULTIPLIES_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value * *lb < *rhs) || (value * *ub < *rhs);
	} else {
		result[threadIdx.x] = (*lb * value < *rhs) || (*ub * value < *rhs);
	}
}

__global__ void filterBoundDIVIDES_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value / *lb == *rhs) || (value / *ub == *rhs);
	} else {
		result[threadIdx.x] = (*lb / value == *rhs) || (*ub / value == *rhs);
	}
}

__global__ void filterBoundDIVIDES_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value / *lb != *rhs) || (value / *ub != *rhs);
	} else {
		result[threadIdx.x] = (*lb / value != *rhs) || (*ub / value != *rhs);
	}
}

__global__ void filterBoundDIVIDES_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value / *lb >= *rhs) || (value / *ub >= *rhs);
	} else {
		result[threadIdx.x] = (*lb / value >= *rhs) || (*ub / value >= *rhs);
	}
}

__global__ void filterBoundDIVIDES_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value / *lb > *rhs) || (value / *ub > *rhs);
	} else {
		result[threadIdx.x] = (*lb / value > *rhs) || (*ub / value > *rhs);
	}
}

__global__ void filterBoundDIVIDES_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value / *lb <= *rhs) || (value / *ub <= *rhs);
	} else {
		result[threadIdx.x] = (*lb / value <= *rhs) || (*ub / value <= *rhs);
	}
}

__global__ void filterBoundDIVIDES_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result, bool *valueIsFirst) {
	int value = *originalLowerBound + threadIdx.x;
	if (*valueIsFirst) {
		result[threadIdx.x] = (value / *lb < *rhs) || (value / *ub < *rhs);
	} else {
		result[threadIdx.x] = (*lb / value < *rhs) || (*ub / value < *rhs);
	}
}

// Kernel for Domain filtering

__global__ void sumMatrixRows(uint8_t * matrix, unsigned int * rowSize, uint8_t *result) {
	int row = threadIdx.x;
	result[row] = 0;
	for (unsigned int i = 0; i < *rowSize; i++) {
		result[row] += matrix[row * *rowSize + i];
	}
}

__global__ void filterDomainMINUS_NEQ(int *rhs, unsigned int *sizeVar1Bitset, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar2, uint8_t * matrix, bool *var1IsFirst) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * (*sizeVar1Bitset) + col;

	if (*var1IsFirst) {
		matrix[matrixIndex] = bitsetvar2[col] && valueVar1 - valueVar2 != *rhs;
	} else {
		matrix[matrixIndex] = bitsetvar2[col] && valueVar2 - valueVar1 != *rhs;
	}
}