#include "stdafx.h"
#include "CppUnitTest.h"
#include "Solver.h"
#include "Model.h"
#include "SumConstraint.h"
#include "FixedIntVariable.h"

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

	};
}