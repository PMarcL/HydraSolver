#include "stdafx.h"
#include "CppUnitTest.h"
#include "SumConstraint.h"
#include "BitsetIntVariable.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(SumConstraintTest) {
public:
	TEST_METHOD(ShouldContainGivenVariables) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		SumConstraint sumConstraint({ &var1, &var2 }, 4);

		Assert::IsTrue(sumConstraint.containsVariable(&var1));
		Assert::IsTrue(sumConstraint.containsVariable(&var2));
	}

	TEST_METHOD(ShouldNotContainOtherVariables) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 2);
		SumConstraint sumConstraint({ &var1 }, 4);

		Assert::IsFalse(sumConstraint.containsVariable(&var2));
	}

	TEST_METHOD(ShouldFilterDomainsWhenFilterIsCalled_simpleCase) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filter();

		// 1 should be filtered from var2, var1 should be unchanged
		Assert::IsFalse(var2.containsValue(1));
		Assert::AreEqual(2, var1.cardinality());
	}

	TEST_METHOD(ShouldFilterDomainsWhenFilterDomainsIsCalled_simpleCase) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var2", 0, 4);
		BitsetIntVariable var4("var2", 0, 5);
		auto sum = 14;

		SumConstraint sumConstraint({ &var1, &var2, &var3, &var4 }, sum);

		sumConstraint.filterDomains();

		// should be filter all values except for upper bounds
		Assert::AreEqual(1, var1.cardinality());
		Assert::IsTrue(var1.containsValue(2));

		Assert::AreEqual(1, var2.cardinality());
		Assert::IsTrue(var2.containsValue(3));

		Assert::AreEqual(1, var3.cardinality());
		Assert::IsTrue(var3.containsValue(4));

		Assert::AreEqual(1, var4.cardinality());
		Assert::IsTrue(var4.containsValue(5));
	}

	TEST_METHOD(ShouldFilterValuesOnFilterBounds_simpleCase) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filterBounds();

		// 0 should be filtered from var1 and 1 should be filtered from var2
		Assert::IsFalse(var1.containsValue(0));
		Assert::IsFalse(var2.containsValue(1));
		Assert::AreEqual(2, var1.cardinality());
		Assert::AreEqual(2, var2.cardinality());
	}

	TEST_METHOD(ShouldFilterValuesOnFilterBounds_simpleCase2) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var2", 0, 4);
		BitsetIntVariable var4("var2", 0, 5);
		auto sum = 14;

		SumConstraint sumConstraint({ &var1, &var2, &var3, &var4 }, sum);

		sumConstraint.filterBounds();

		// should be filter all values except for upper bounds
		Assert::AreEqual(1, var1.cardinality());
		Assert::IsTrue(var1.containsValue(2));

		Assert::AreEqual(1, var2.cardinality());
		Assert::IsTrue(var2.containsValue(3));

		Assert::AreEqual(1, var3.cardinality());
		Assert::IsTrue(var3.containsValue(4));

		Assert::AreEqual(1, var4.cardinality());
		Assert::IsTrue(var4.containsValue(5));
	}

	};
}