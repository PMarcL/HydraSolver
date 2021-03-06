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

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringBoundsWithInconsistentVariables) {
		BitsetIntVariable var1("test", 2, 3);
		BitsetIntVariable var2("test", 3, 4);
		BitsetIntVariable var3("test", 3, 4);
		AllDifferent allDiff({ &var1, &var2, &var3 });

		allDiff.filterBounds();

		Assert::IsTrue(allDiff.isSatisfied());
	}

	TEST_METHOD(ShouldNotBeSatisfiedAfterFilteringBoundsWithInconsistentVariables) {
		BitsetIntVariable var01("test", 1, 2);
		BitsetIntVariable var02("test", 1, 2);
		BitsetIntVariable var1("test", 2, 3);
		BitsetIntVariable var2("test", 3, 4);
		BitsetIntVariable var3("test", 3, 4);
		AllDifferent allDiff({ &var01, &var02, &var1, &var2, &var3 });

		allDiff.filterBounds();

		Assert::IsFalse(allDiff.isSatisfied());
	}

	TEST_METHOD(ShouldNotContainVariableAfterReplacementAndShouldContainReplacement) {
		FixedIntVariable var1("test", 1);
		FixedIntVariable var2("test", 2);
		FixedIntVariable var3("test", 3);
		AllDifferent allDiff({ &var1, &var2 });

		allDiff.replaceVariable(&var1, &var3);

		Assert::IsTrue(allDiff.containsVariable(&var3));
		Assert::IsFalse(allDiff.containsVariable(&var1));
	}

	TEST_METHOD(CloneShouldContainTheSameVariablesAsOriginal) {
		FixedIntVariable var1("test", 1);
		FixedIntVariable var2("test", 2);
		AllDifferent allDiff({ &var1, &var2 });

		auto clone = allDiff.clone();

		Assert::IsTrue(clone->containsVariable(&var1));
		Assert::IsTrue(clone->containsVariable(&var2));

		delete clone;
	}

	};
}