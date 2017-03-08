#pragma once

#include <vector>

namespace hydra {

	class Variable;

	class Solution {
	public:
		Solution();
		Solution(const std::vector<Variable*>& variables, bool isConsistent);
		Solution(const Solution& solution);
		~Solution();

		bool isConsistent() const;
		std::string getFormattedSolution() const;

		Solution& operator=(const Solution& solution);
	private:
		bool isAConsistentSolution;
		std::vector<Variable*> variables;
	};

} // namespace hydra
