#include "SumConstraint.h"
#include "IntVariable.h"

using namespace std;

namespace hydra {

	SumConstraint::SumConstraint(const vector<IntVariable*>& variables, int sum) : variables(variables), sum(sum) {
	}


	SumConstraint::~SumConstraint() {
	}

	void SumConstraint::filter() {

	}

	void SumConstraint::filterBounds() {

	}

	void SumConstraint::filterDomains() {

	}

	void SumConstraint::CPUdomainFilteringAlgorithm() {

	}


} // namespace hydra
