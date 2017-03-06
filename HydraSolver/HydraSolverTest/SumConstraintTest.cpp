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
		int sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, 4);

		sumConstraint.filter();

		// 1 should be filtered from var2, var1 should be unchanged
		//Assert::IsFalse(var2.containsValue(1));
		//Assert::AreEqual(2, var1.cardinality());
	}

	};
}