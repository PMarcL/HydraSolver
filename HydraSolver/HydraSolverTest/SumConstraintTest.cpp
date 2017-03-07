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

	TEST_METHOD(ShouldFilterDomainsWhenFilterIsCalled_simpleCase2) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var2", 0, 4);
		BitsetIntVariable var4("var2", 0, 5);
		auto sum = 14;

		SumConstraint sumConstraint({ &var1, &var2, &var3, &var4 }, sum);

		sumConstraint.filter();

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