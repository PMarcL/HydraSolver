#include "stdafx.h"
#include "CppUnitTest.h"
#include "Solution.h"
#include "VariableImpl.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(SolutionTest) {
public:
	TEST_METHOD(ShouldBeConsistentGivenAConsistentSolution) {
		Solution solution({}, true);
		Assert::IsTrue(solution.isConsistent());
	}

	TEST_METHOD(ShouldNotBeConsistentGivenAnInconsistentSolution) {
		Solution solution({}, false);
		Assert::IsFalse(solution.isConsistent());
	}

	TEST_METHOD(ShouldPrintAllVariableOnPrintSolution) {
		VariableImpl var1;
		VariableImpl var2;
		VariableImpl var3;
		Solution solution({ &var1, &var2, &var3 }, true);

		solution.printSolution();

		Assert::IsTrue(var1.formattedDomainWasCalled);
		Assert::IsTrue(var2.formattedDomainWasCalled);
		Assert::IsTrue(var3.formattedDomainWasCalled);
	}
	};
}