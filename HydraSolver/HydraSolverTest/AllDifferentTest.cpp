#include "stdafx.h"
#include "CppUnitTest.h"
#include "AllDifferent.h"
#include "BitsetIntVariable.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(AllDifferentTest) {
public:
	TEST_METHOD(ShouldContainGivenVariables) {
		BitsetIntVariable var1("test", 1, 10);
		BitsetIntVariable var2("test", 1, 10);
		AllDifferent allDiff({ &var1, &var2 });

		Assert::IsTrue(allDiff.containsVariable(&var1));
		Assert::IsTrue(allDiff.containsVariable(&var2));
	}

	TEST_METHOD(ShouldNotContainOtherVariables) {
		BitsetIntVariable var1("test", 1, 10);
		BitsetIntVariable var2("test", 1, 10);
		AllDifferent allDiff({ &var1 });

		Assert::IsFalse(allDiff.containsVariable(&var2));
	}

	};
}