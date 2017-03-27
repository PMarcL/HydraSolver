#pragma once

#include <cstdint>
#include <stdio.h>
#include "device_launch_parameters.h"

__global__ void filterBoundPLUS_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value + *lb == *rhs) || (value + *ub == *rhs);
}

__global__ void filterBoundPLUS_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value + *lb != *rhs) || (value + *ub != *rhs);
}

__global__ void filterBoundPLUS_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value + *lb >= *rhs) || (value + *ub >= *rhs);
}

__global__ void filterBoundPLUS_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value + *lb > *rhs) || (value + *ub > *rhs);
}

__global__ void filterBoundPLUS_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value + *lb <= *rhs) || (value + *ub <= *rhs);
}

__global__ void filterBoundPLUS_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value + *lb < *rhs) || (value + *ub < *rhs);
}

__global__ void filterBoundMINUS_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value - *lb == *rhs) || (value - *ub == *rhs);
}

__global__ void filterBoundMINUS_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value - *lb != *rhs) || (value - *ub != *rhs);
}

__global__ void filterBoundMINUS_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value - *lb >= *rhs) || (value - *ub >= *rhs);
}

__global__ void filterBoundMINUS_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value - *lb > *rhs) || (value - *ub > *rhs);
}

__global__ void filterBoundMINUS_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value - *lb <= *rhs) || (value - *ub <= *rhs);
}

__global__ void filterBoundMINUS_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value - *lb < *rhs) || (value - *ub < *rhs);
}

__global__ void filterBoundMULTIPLIES_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value * *lb == *rhs) || (value * *ub == *rhs);
}

__global__ void filterBoundMULTIPLIES_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value * *lb != *rhs) || (value * *ub != *rhs);
}

__global__ void filterBoundMULTIPLIES_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	printf("Lowerbound : %d\n", *originalLowerBound);
	printf("Value : %d\n", value);
	printf("Lowerbound : %d\n", *lb);
	printf("Upperbound : %d\n", *ub);
	result[threadIdx.x] = (value * *lb >= *rhs) || (value * *ub >= *rhs);
}

__global__ void filterBoundMULTIPLIES_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value * *lb > *rhs) || (value * *ub > *rhs);
}

__global__ void filterBoundMULTIPLIES_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value * *lb <= *rhs) || (value * *ub <= *rhs);
}

__global__ void filterBoundMULTIPLIES_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value * *lb < *rhs) || (value * *ub < *rhs);
}

__global__ void filterBoundDIVIDES_EQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value / *lb == *rhs) || (value / *ub == *rhs);
}

__global__ void filterBoundDIVIDES_NEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value / *lb != *rhs) || (value / *ub != *rhs);
}

__global__ void filterBoundDIVIDES_GEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value / *lb >= *rhs) || (value / *ub >= *rhs);
}

__global__ void filterBoundDIVIDES_GT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value / *lb > *rhs) || (value / *ub > *rhs);
}

__global__ void filterBoundDIVIDES_LEQ(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value / *lb <= *rhs) || (value / *ub <= *rhs);
}

__global__ void filterBoundDIVIDES_LT(int *rhs, int *lb, int *ub, int *originalLowerBound, uint8_t *result) {
	int value = *originalLowerBound + threadIdx.x;
	result[threadIdx.x] = (value / *lb < *rhs) || (value / *ub < *rhs);
}