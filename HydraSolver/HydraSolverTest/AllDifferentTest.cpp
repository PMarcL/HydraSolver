#include "stdafx.h"
#include "CppUnitTest.h"
#include "AllDifferent.h"
#include "BitsetIntVariable.h"
#include "FixedIntVariable.h"

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

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringWithConsistentVariables) {
		BitsetIntVariable var1("test", 1, 10);
		BitsetIntVariable var2("test", 1, 10);
		AllDifferent allDiff({ &var1, &var2 });

		allDiff.filter();

		Assert::IsTrue(allDiff.isSatisfied());
	}

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringWithConsistentVariables_2) {
		BitsetIntVariable var1("var1", 11, 13);
		var1.filterValue(12);
		BitsetIntVariable var2("var2", 9, 13);
		var2.filterValue(11);
		var2.filterValue(12);
		BitsetIntVariable var3("var3", 9, 13);
		var3.filterValue(10);
		var3.filterValue(12);
		BitsetIntVariable var4("var4", 11, 13);
		var4.filterValue(12);
		AllDifferent allDiff({ &var1, &var2, &var3, &var4 });

		auto filteredVariables = allDiff.filter();

		size_t expectedFilteredSize = 2;
		Assert::IsTrue(allDiff.isSatisfied());
		Assert::AreEqual(expectedFilteredSize, filteredVariables.size());
	}

	TEST_METHOD(ShouldNotBeSatisfiedAfterFilteringWithInconsistentVariables) {
		FixedIntVariable var1("test", 1);
		FixedIntVariable var2("test", 1);
		AllDifferent allDiff({ &var1, &var2 });

		allDiff.filter();

		Assert::IsFalse(allDiff.isSatisfied());
	}

	};
}