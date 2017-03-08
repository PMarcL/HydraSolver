#include "Solution.h"
#include "Variable.h"
#include <string>
#include <iostream>

using namespace std;

namespace hydra {

	Solution::Solution(const vector<Variable*>& variables, bool isConsistent) : isAConsistentSolution(isConsistent), variables(variables) {
	}

	Solution::~Solution() {
	}

	bool Solution::isConsistent() const {
		return isAConsistentSolution;
	}

	void Solution::printSolution() const {
		cout << "Solution :" << endl;
		for (auto var : variables) {
			cout << var->getName() << " = " << var->getFormattedDomain() << endl;
		}
	}


} // namespace hydra
