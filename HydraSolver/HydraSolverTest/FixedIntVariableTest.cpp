#include "stdafx.h"
#include "CppUnitTest.h"
#include "FixedIntVariable.h"
#include "IllegalVariableOperationException.h"

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

	TEST_METHOD(ShouldThrowIllegalOperationWhenTryingToFilterValue) {
		auto func = [] {
			FixedIntVariable fixedInt("test", 42);
			fixedInt.filterValue(10);
		};

		Assert::ExpectException<IllegalVariableOperationException, void>(func);
	}

	TEST_METHOD(ShouldThrowIllegalOperationWhenTryingToFilterLowerBound) {
		auto func = [] {
			FixedIntVariable fixedInt("test", 42);
			fixedInt.filterLowerBound(10);
		};

		Assert::ExpectException<IllegalVariableOperationException, void>(func);
	}

	TEST_METHOD(ShouldThrowIllegalOperationWhenTryingToFilterUpperBound) {
		auto func = [] {
			FixedIntVariable fixedInt("test", 42);
			fixedInt.filterUpperBound(10);
		};

		Assert::ExpectException<IllegalVariableOperationException, void>(func);
	}

	TEST_METHOD(ShouldHaveACardinalityOfOne) {
		FixedIntVariable fixedInt("test", 42);
		Assert::AreEqual(1, fixedInt.cardinality());
	}

	TEST_METHOD(IteratorShouldAlwaysReturnTheFixedValue) {
		auto expectedValue = 42;
		FixedIntVariable fixedInt("test", expectedValue);

		auto iterator = fixedInt.iterator();

		Assert::AreEqual(expectedValue, iterator->next());
		Assert::AreEqual(expectedValue, iterator->previous());

		delete iterator;
	}
	};
}