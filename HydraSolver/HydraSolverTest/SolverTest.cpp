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

	TEST_METHOD(SolverTest_simpleCase) {
		auto var1 = new FixedIntVariable("var1", 3);
		auto var2 = new FixedIntVariable("var2", 4);
		Model model;
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, 7);
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTest_simpleCase results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFiltering_simpleCase) {
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		Model model;
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, 7);
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFiltering_simpleCase results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFilteringAndRandomHeuristic_simpleCase) {
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		auto var3 = new BitsetIntVariable("var2", 1, 2);
		auto var4 = new BitsetIntVariable("var2", 10, 15);
		Model model;
		model.addVariableArray({ var1, var2, var3, var4 });
		auto sumConstraint = new SumConstraint({ var1, var2, var3, var4 }, 21);
		model.postConstraint(sumConstraint);

		Solver solver(&model, RANDOM);
		auto solution = solver.findSolution();

		Assert::IsTrue(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFilteringAndRandomHeuristic_simpleCase results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	TEST_METHOD(SolverTestWithFilteringNoSolution_simpleCase) {
		auto var1 = new BitsetIntVariable("var1", 1, 4);
		auto var2 = new BitsetIntVariable("var2", 3, 8);
		Model model;
		model.addVariableArray({ var1, var2 });
		auto sumConstraint = new SumConstraint({ var1, var2 }, 13);
		model.postConstraint(sumConstraint);

		Solver solver(&model);
		auto solution = solver.findSolution();

		Assert::IsFalse(solution.isConsistent());

		Logger::WriteMessage("SolverTestWithFilteringNoSolution_simpleCase results :");
		Logger::WriteMessage(solution.getFormattedSolution().c_str());
	}

	};
}