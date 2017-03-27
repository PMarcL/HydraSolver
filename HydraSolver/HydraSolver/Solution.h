#pragma once

#include <vector>
#include <chrono>

namespace hydra {

	class Variable;
	class Model;

	class Solution {
	public:
		Solution();
		Solution(const std::vector<Variable*>& variables, bool isConsistent, Model* model);
		Solution(const Solution& solution);
		~Solution();

		bool isConsistent() const;
		std::string getFormattedSolution() const;
		void setComputingtime(long long executionTime);
		void setNumberOfBacktracks(int backtracks);
		void setNumberOfRestarts(int restarts);

		Solution& operator=(const Solution& solution);
	private:
		bool isAConsistentSolution;
		std::vector<Variable*> variables;
		long long computingTime;
		int nbOfBacktracks;
		int nbOfRestarts;
		Model* model;
	};

} // namespace hydra
