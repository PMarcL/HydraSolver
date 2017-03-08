#include "Solution.h"
#include "Variable.h"
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

namespace hydra {

	Solution::Solution() : isAConsistentSolution(false) {
	}

	Solution::Solution(const Solution& solution) : isAConsistentSolution(solution.isAConsistentSolution), variables(solution.variables) {
	}

	Solution::Solution(const vector<Variable*>& variables, bool isConsistent) : isAConsistentSolution(isConsistent), variables(variables) {
	}

	Solution::~Solution() {
	}

	bool Solution::isConsistent() const {
		return isAConsistentSolution;
	}

	string Solution::getFormattedSolution() const {
		ostringstream os;

		os << "Solution :" << endl;
		for (auto var : variables) {
			os << var->getName() << " = " << var->getFormattedDomain() << endl;
		}

		return os.str();
	}

	Solution& Solution::operator=(const Solution& solution) {
		isAConsistentSolution = solution.isConsistent();
		variables = solution.variables;
		return *this;
	}

} // namespace hydra
