#include "stdafx.h"
#include "CppUnitTest.h"
#include "Solution.h"
#include "VariableImpl.h"
#include "Model.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(SolutionTest) {
public:
	TEST_METHOD(ShouldBeConsistentGivenAConsistentSolution) {
		Model m("test");
		Solution solution({}, true, &m);
		Assert::IsTrue(solution.isConsistent());
	}

	TEST_METHOD(ShouldNotBeConsistentGivenAnInconsistentSolution) {
		Model m("test");
		Solution solution({}, false, &m);
		Assert::IsFalse(solution.isConsistent());
	}

	TEST_METHOD(ShouldPrintAllVariableOnGetFormattedSolution) {
		Model m("test");
		VariableImpl var1;
		VariableImpl var2;
		VariableImpl var3;
		Solution solution({ &var1, &var2, &var3 }, true, &m);

		solution.getFormattedSolution();

		Assert::IsTrue(var1.formattedDomainWasCalled);
		Assert::IsTrue(var2.formattedDomainWasCalled);
		Assert::IsTrue(var3.formattedDomainWasCalled);
	}
	};
}