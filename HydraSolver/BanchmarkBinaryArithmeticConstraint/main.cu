#include "BinaryArithmeticConstraint.h"
#include "BitsetIntVariable.h"
#include "TimeLogger.h"

using namespace hydra;
using namespace std;

const int NUMBER_OF_ITERATIONS = 1000;

int main() {
	TimeLogger loggerDomainGPU("Banchmark-BinaryArithmeticConstraint-domain-GPU");
	TimeLogger loggerDomainCPU("Banchmark-BinaryArithmeticConstraint-domain-CPU");
	TimeLogger loggerBoundsGPU("Banchmark-BinaryArithmeticConstraint-bounds-GPU");
	TimeLogger loggerBoundsCPU("Banchmark-BinaryArithmeticConstraint-bounds-CPU");

	for (auto i = 0; i < NUMBER_OF_ITERATIONS; i++) {
		auto currentCardinality = 10 + i;
		// GPU domain filtering
		BitsetIntVariable v1("v1", 1, currentCardinality);
		BitsetIntVariable v2("v2", 1, currentCardinality);
		BinaryArithmeticConstraint constraintDomainGPU(&v1, &v2, currentCardinality / 2 * 7, MULTIPLIES, EQ);
		constraintDomainGPU.setGPUFilteringActive();

		loggerDomainGPU.tic();
		constraintDomainGPU.filterDomains();
		loggerDomainGPU.toc(currentCardinality);

		// GPU bounds filtering
		v1 = BitsetIntVariable("v1", 1, currentCardinality);
		v2 = BitsetIntVariable("v2", 1, currentCardinality);
		BinaryArithmeticConstraint constraintBoundsGPU(&v1, &v2, currentCardinality / 2 * 7, MULTIPLIES, EQ);
		constraintBoundsGPU.setGPUFilteringActive();

		loggerBoundsGPU.tic();
		constraintBoundsGPU.filterBounds();
		loggerBoundsGPU.toc(currentCardinality);

		// CPU domain filtering
		v1 = BitsetIntVariable("v1", 1, currentCardinality);
		v2 = BitsetIntVariable("v2", 1, currentCardinality);
		BinaryArithmeticConstraint constraintDomainCPU(&v1, &v2, currentCardinality / 2 * 7, MULTIPLIES, EQ);

		loggerDomainCPU.tic();
		constraintDomainCPU.filterDomains();
		loggerDomainCPU.toc(currentCardinality);

		// CPU bounds filtering
		v1 = BitsetIntVariable("v1", 1, currentCardinality);
		v2 = BitsetIntVariable("v2", 1, currentCardinality);
		BinaryArithmeticConstraint constraintBoundsCPU(&v1, &v2, currentCardinality / 2 * 7, MULTIPLIES, EQ);

		loggerBoundsCPU.tic();
		constraintDomainCPU.filterBounds();
		loggerBoundsCPU.toc(currentCardinality);
	}

	return 0;
}

