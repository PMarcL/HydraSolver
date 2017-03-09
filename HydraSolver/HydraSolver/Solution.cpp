#include "Solution.h"
#include "Variable.h"
#include "Model.h"
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

namespace hydra {

	Solution::Solution() : isAConsistentSolution(false), computingTime(0), nbOfBacktracks(0), model(nullptr) {
	}

	Solution::Solution(const Solution& solution) : isAConsistentSolution(solution.isAConsistentSolution), variables(solution.variables),
		computingTime(solution.computingTime), nbOfBacktracks(solution.nbOfBacktracks), model(solution.model) {
	}

	Solution::Solution(const vector<Variable*>& variables, bool isConsistent, Model* model) : isAConsistentSolution(isConsistent), variables(variables),
		computingTime(0), nbOfBacktracks(0), model(model) {
	}

	Solution::~Solution() {
	}

	bool Solution::isConsistent() const {
		return isAConsistentSolution;
	}

	string Solution::getFormattedSolution() const {
		ostringstream os;
		os << "- Model - " << model->getName() << endl;
		os << "\tVariables : " << model->getNumberOfVariables() << endl;
		os << "\tConstraints : " << model->getNumberOfConstraints() << endl;

		os << "- Complete search - ";
		if (!isConsistent()) {
			os << "no solution found.";
			return os.str();
		}

		os << "solution found." << endl;
		os << "\tResolution time : " << computingTime / 1000 << "." << computingTime % 1000 << "s" << endl;
		os << "\tBacktracks : " << nbOfBacktracks << endl;

		os << "Solution :" << endl;
		for (auto i = 0; i < variables.size() - 1; i++) {
			auto var = variables[i];
			os << "\t" << var->getName() << " = " << var->getFormattedDomain() << ", ";
		}
		os << "\t" << variables[variables.size() - 1]->getName() << " = " << variables[variables.size() - 1]->getFormattedDomain() << endl;
		return os.str();
	}

	void Solution::setComputingtime(long long executionTime) {
		computingTime = executionTime;
	}

	void Solution::setNumberOfBacktracks(int backtracks) {
		nbOfBacktracks = backtracks;
	}

	Solution& Solution::operator=(const Solution& solution) {
		isAConsistentSolution = solution.isConsistent();
		variables = solution.variables;
		nbOfBacktracks = solution.nbOfBacktracks;
		computingTime = solution.computingTime;
		model = solution.model;
		return *this;
	}

} // namespace hydra
