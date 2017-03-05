#include "stdafx.h"
#include "CppUnitTest.h"
#include "FixedIntVariable.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(FixedIntVariableTest) {
public:
	TEST_METHOD(ShouldHaveGivenNameAndGivenValue) {
		string expectedName = "test";
		auto expectedValue = 42;

		FixedIntVariable fixedInt(expectedName, expectedValue);

		Assert::AreEqual(expectedName, fixedInt.getName());
		Assert::IsTrue(fixedInt.containsValue(expectedValue));
	}

	TEST_METHOD(LowerBoundAndUpperBoundShouldBeTheFixedValue) {
		auto expectedValue = 42;

		FixedIntVariable fixedInt("test", expectedValue);

		Assert::AreEqual(expectedValue, fixedInt.getLowerBound());
		Assert::AreEqual(expectedValue, fixedInt.getUpperBound());
	}

	};
}