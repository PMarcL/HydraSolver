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

__global__ void sumMatrixRows(uint8_t * matrix, unsigned int rowSize, uint8_t *result) {
	int row = threadIdx.x;
	result[row] = 0;
	for (unsigned int i = 0; i < rowSize; i++) {
		result[row] += matrix[row * rowSize + i];
	}
}

__global__ void sumMatrixCols(uint8_t * matrix, unsigned int rowSize, unsigned int colSize, uint8_t *result) {
	int col = threadIdx.x;
	result[col] = 0;
	for (unsigned int i = 0; i < colSize; i++) {
		result[col] += matrix[i * rowSize + col];
	}
}

__global__ void filterDomainDIVIDES_GT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 / valueVar2 > *rhs;
}

__global__ void filterDomainDIVIDES_LT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 / valueVar2 < *rhs;
}

__global__ void filterDomainDIVIDES_LEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 / valueVar2 <= *rhs;
}

__global__ void filterDomainDIVIDES_GEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 / valueVar2 >= *rhs;
}

__global__ void filterDomainDIVIDES_EQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 / valueVar2 == *rhs;
}

__global__ void filterDomainDIVIDES_NEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 / valueVar2 != *rhs;
}

__global__ void filterDomainMULTIPLIES_GT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 * valueVar2 > *rhs;
}

__global__ void filterDomainMULTIPLIES_LT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 * valueVar2 < *rhs;
}

__global__ void filterDomainMULTIPLIES_LEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 * valueVar2 <= *rhs;
}

__global__ void filterDomainMULTIPLIES_GEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 * valueVar2 >= *rhs;
}

__global__ void filterDomainMULTIPLIES_EQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 * valueVar2 == *rhs;
}

__global__ void filterDomainMULTIPLIES_NEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 * valueVar2 != *rhs;
}

__global__ void filterDomainPLUS_GT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 + valueVar2 > *rhs;
}

__global__ void filterDomainPLUS_LT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 + valueVar2 < *rhs;
}

__global__ void filterDomainPLUS_LEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 + valueVar2 <= *rhs;
}

__global__ void filterDomainPLUS_GEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 + valueVar2 >= *rhs;
}

__global__ void filterDomainPLUS_EQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 + valueVar2 == *rhs;
}

__global__ void filterDomainPLUS_NEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 + valueVar2 != *rhs;
}

__global__ void filterDomainMINUS_GT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 - valueVar2 > *rhs;
}

__global__ void filterDomainMINUS_LT(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 - valueVar2 < *rhs;
}

__global__ void filterDomainMINUS_LEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 - valueVar2 <= *rhs;
}

__global__ void filterDomainMINUS_GEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 - valueVar2 >= *rhs;
}

__global__ void filterDomainMINUS_EQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 - valueVar2 == *rhs;
}

__global__ void filterDomainMINUS_NEQ(int *rhs, int *originalLowerBoundVar1, int *originalLowerboundVar2,
	uint8_t *bitsetvar1, uint8_t *bitsetvar2, uint8_t * matrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int valueVar1 = *originalLowerBoundVar1 + threadIdx.y;
	int valueVar2 = *originalLowerboundVar2 + threadIdx.x;
	int matrixIndex = row * blockDim.x + col;

	matrix[matrixIndex] = bitsetvar1[row] && bitsetvar2[col] && valueVar1 - valueVar2 != *rhs;
}