#pragma once

#include <vector>

namespace hydra {

	class Variable;

	class Solution {
	public:
		Solution(const std::vector<Variable*>& variables, bool isConsistent);
		~Solution();

		bool isConsistent() const;
		void printSolution() const;

	private:
		bool isAConsistentSolution;
		std::vector<Variable*> variables;
	};

} // namespace hydra
