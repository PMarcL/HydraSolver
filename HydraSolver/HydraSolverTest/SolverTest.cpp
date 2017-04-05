#include "stdafx.h"
#include "CppUnitTest.h"
#include "Solver.h"
#include "Model.h"
#include "SumConstraint.h"
#include "FixedIntVariable.h"
#include "BitsetIntVariable.h"

using namespace hydra;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {
	TEST_CLASS(SolverTest) {
public:

	TEST_METHOD(SolverTest_solutionShouldAddUpTo9) {
		auto expectedSum = 9;
		auto var1 = new FixedIntVariable("var1", 4);
		auto var2 = new FixedIntVariable("var2", 5);
		Model model("Solver test - sum with fixed ints");
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, expectedSum);
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTest_solutionShouldAddUpTo9 results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFiltering_solutionShouldAddUpTo7) {
		auto expectedSum = 7;
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		Model model("Solver test - sum with bitsets");
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, expectedSum);
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFiltering_solutionShouldAddUpTo7 results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFiltering_solutionShouldAddUpTo7_usingGPU) {
		auto expectedSum = 7;
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		Model model("Solver test - sum with bitsets");
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, expectedSum);
		sumConstraint->setGPUFilteringActive();
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		solver.setLocalConsistencyConfig(hydra::BOUND_CONSISTENCY);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFiltering_solutionShouldAddUpTo7 results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFilteringAndRandomHeuristic_solutionShouldAddUpTo21) {
		auto expectedSum = 21;
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		auto var3 = new BitsetIntVariable("var2", 1, 2);
		auto var4 = new BitsetIntVariable("var2", 10, 15);
		Model model("Solver test - sum with random heuristic");
		model.addVariableArray({ var1, var2, var3, var4 });
		auto sumConstraint = new SumConstraint({ var1, var2, var3, var4 }, expectedSum);
		model.postConstraint(sumConstraint);

		Solver solver(&model, RANDOM);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFilteringAndRandomHeuristic_solutionShouldAddUpTo21 results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFilteringAndRandomHeuristic_solutionShouldAddUpTo21_usingGPU) {
		auto expectedSum = 21;
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		auto var3 = new BitsetIntVariable("var2", 1, 2);
		auto var4 = new BitsetIntVariable("var2", 10, 15);
		Model model("Solver test - sum with random heuristic");
		model.addVariableArray({ var1, var2, var3, var4 });
		auto sumConstraint = new SumConstraint({ var1, var2, var3, var4 }, expectedSum);
		sumConstraint->setGPUFilteringActive();
		model.postConstraint(sumConstraint);

		Solver solver(&model, RANDOM);
		solver.setLocalConsistencyConfig(BOUND_CONSISTENCY);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFilteringAndRandomHeuristic_solutionShouldAddUpTo21 results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFiltering_NoSolution) {
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		Model model("Solver test - no solution");
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, 13);
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsFalse(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFiltering_NoSolution results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFiltering_NoSolution_usingGPU) {
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		Model model("Solver test - no solution");
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, 13);
		sumConstraint->setGPUFilteringActive();
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		solver.setLocalConsistencyConfig(BOUND_CONSISTENCY);
		auto solution = solver.findSolution();

		Assert::IsFalse(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFiltering_NoSolution results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithMultipleConstraints) {
		auto expectedSum1 = 21;
		auto expectedSum2 = 4;
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		auto var3 = new BitsetIntVariable("var2", 1, 2);
		auto var4 = new BitsetIntVariable("var2", 10, 15);
		Model model("Solver test - multiple constraints");
		model.addVariableArray({ var1, var2, var3, var4 });
		auto sumConstraint = new SumConstraint({ var1, var2, var3, var4 }, expectedSum1);
		model.postConstraint(sumConstraint);
		auto sumConstraint2 = new SumConstraint({ var1, var2 }, expectedSum2);
		model.postConstraint(sumConstraint2);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithMultipleConstraints results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithMultipleConstraints_usingGPU) {
		auto expectedSum1 = 21;
		auto expectedSum2 = 4;
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		auto var3 = new BitsetIntVariable("var2", 1, 2);
		auto var4 = new BitsetIntVariable("var2", 10, 15);
		Model model("Solver test - multiple constraints");
		model.addVariableArray({ var1, var2, var3, var4 });
		auto sumConstraint = new SumConstraint({ var1, var2, var3, var4 }, expectedSum1);
		sumConstraint->setGPUFilteringActive();
		model.postConstraint(sumConstraint);
		auto sumConstraint2 = new SumConstraint({ var1, var2 }, expectedSum2);
		sumConstraint->setGPUFilteringActive();
		model.postConstraint(sumConstraint2);

		Solver solver(&model);
		solver.setLocalConsistencyConfig(BOUND_CONSISTENCY);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithMultipleConstraints results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	};
}