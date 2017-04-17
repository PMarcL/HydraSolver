#include "Solution.h"
#include "Variable.h"
#include "Model.h"
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

namespace hydra {

	Solution::Solution() : isAConsistentSolution(false), computingTime(0), nbOfBacktracks(0), nbOfRestarts(0), model(nullptr) {
	}

	Solution::Solution(const Solution& solution) : isAConsistentSolution(solution.isAConsistentSolution), variables(solution.variables),
		computingTime(solution.computingTime), nbOfBacktracks(solution.nbOfBacktracks), nbOfRestarts(solution.nbOfRestarts), model(solution.model) {
	}

	Solution::Solution(const vector<Variable*>& variables, bool isConsistent, Model* model) : isAConsistentSolution(isConsistent), variables(variables),
		computingTime(0), nbOfBacktracks(0), nbOfRestarts(0), model(model) {
	}

	Solution::~Solution() {
	}

	bool Solution::isConsistent() const {
		return isAConsistentSolution;
	}

	int Solution::getNumberOfBacktracks() const {
		return nbOfBacktracks;
	}

	int Solution::getNumberOfRestarts() const {
		return nbOfRestarts;
	}


	string Solution::getFormattedSolution() const {
		ostringstream os;
		os << "- Model - " << model->getName() << endl;
		os << "\tVariables : " << model->getNumberOfVariables() << endl;
		os << "\tConstraints : " << model->getNumberOfConstraints() << endl;

		os << "- Complete search - ";
		if (!isConsistent()) {
			os << "no solution found." << endl;
		} else {
			os << "solution found." << endl;
		}

		os << "\tResolution time : " << computingTime / 1000 << "." << setfill('0') << setw(3) << computingTime % 1000 << "s" << endl;
		os << "\tBacktracks : " << nbOfBacktracks << endl;
		os << "\tRestarts : " << nbOfRestarts << endl;

		if (!isConsistent()) {
			return os.str();
		}

		os << "Solution :" << endl;
		for (size_t i = 0; i < variables.size() - 1; i++) {
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

	void Solution::setNumberOfRestarts(int restarts) {
		nbOfRestarts = restarts;
	}

	Solution& Solution::operator=(const Solution& solution) {
		isAConsistentSolution = solution.isConsistent();
		variables = solution.variables;
		nbOfBacktracks = solution.nbOfBacktracks;
		computingTime = solution.computingTime;
		nbOfRestarts = solution.nbOfRestarts;
		model = solution.model;
		return *this;
	}

} // namespace hydra
