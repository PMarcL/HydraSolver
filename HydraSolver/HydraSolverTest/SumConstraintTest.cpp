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

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringConsistentVariables) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filter();

		Assert::IsTrue(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringConsistentVariablesDomains) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 2);
		auto sum = 3;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filterDomains();

		Assert::IsTrue(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringConsistentVariablesBounds) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 2);
		auto sum = 3;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filterBounds();

		Assert::IsTrue(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldBeSatisfiedAfterFilteringConsistentVariablesBounds_usingGPU) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 2);
		auto sum = 3;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);
		sumConstraint.setGPUFilteringActive();

		sumConstraint.filterBounds();

		Assert::IsTrue(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldNotBeSatisfiedAfterFilteringInconsistentVariables) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 6;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filter();

		Assert::IsFalse(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldNotBeSatisfiedAfterFilteringInconsistentVariablesDomains) {
		BitsetIntVariable var1("var1", 2, 3);
		BitsetIntVariable var2("var2", 1, 4);
		auto sum = 8;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filterDomains();

		Assert::IsFalse(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldNotBeSatisfiedAfterFilteringInconsistentVariablesBounds) {
		BitsetIntVariable var1("var1", 2, 5);
		BitsetIntVariable var2("var2", 1, 4);
		auto sum = 12;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		sumConstraint.filterBounds();

		Assert::IsFalse(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldNotBeSatisfiedAfterFilteringInconsistentVariablesBounds_usingGPU) {
		BitsetIntVariable var1("var1", 2, 5);
		BitsetIntVariable var2("var2", 1, 4);
		auto sum = 12;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);
		sumConstraint.setGPUFilteringActive();

		sumConstraint.filterBounds();

		Assert::IsFalse(sumConstraint.isSatisfied());
	}

	TEST_METHOD(ShouldReturnFilteredVariablesWhenFilteringDomain) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		auto modifiedVariable = sumConstraint.filter();

		Assert::IsTrue(find(modifiedVariable.begin(), modifiedVariable.end(), &var2) != modifiedVariable.end());
		Assert::IsTrue(find(modifiedVariable.begin(), modifiedVariable.end(), &var1) == modifiedVariable.end());
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

	TEST_METHOD(ShouldFilterValuesOnFilterBounds_simpleCase_usingGPU) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);
		sumConstraint.setGPUFilteringActive();

		sumConstraint.filterBounds();

		// 0 should be filtered from var1 and 1 should be filtered from var2
		Assert::IsFalse(var1.containsValue(0));
		Assert::IsFalse(var2.containsValue(1));
		Assert::AreEqual(2, var1.cardinality());
		Assert::AreEqual(2, var2.cardinality());
	}

	TEST_METHOD(ShouldReturnModifiedVariablesWhenFilteringBounds) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);

		auto modifiedVariables = sumConstraint.filterBounds();

		Assert::IsTrue(find(modifiedVariables.begin(), modifiedVariables.end(), &var1) != modifiedVariables.end());
		Assert::IsTrue(find(modifiedVariables.begin(), modifiedVariables.end(), &var2) != modifiedVariables.end());
	}

	TEST_METHOD(ShouldReturnModifiedVariablesWhenFilteringBounds_usingGPU) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 1, 3);
		auto sum = 4;

		SumConstraint sumConstraint({ &var1, &var2 }, sum);
		sumConstraint.setGPUFilteringActive();

		auto modifiedVariables = sumConstraint.filterBounds();

		Assert::IsTrue(find(modifiedVariables.begin(), modifiedVariables.end(), &var1) != modifiedVariables.end());
		Assert::IsTrue(find(modifiedVariables.begin(), modifiedVariables.end(), &var2) != modifiedVariables.end());
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

	TEST_METHOD(ShouldFilterValuesOnFilterBounds_simpleCase2_usingGPU) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var2", 0, 4);
		BitsetIntVariable var4("var2", 0, 5);
		auto sum = 14;

		SumConstraint sumConstraint({ &var1, &var2, &var3, &var4 }, sum);
		sumConstraint.setGPUFilteringActive();

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

	TEST_METHOD(ShouldNotContainVariableAfterReplacementAndShouldContainReplacement) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var2", 0, 4);

		SumConstraint sumConstraint({ &var1, &var2 }, 4);

		sumConstraint.replaceVariable(&var1, &var3);

		Assert::IsTrue(sumConstraint.containsVariable(&var3));
		Assert::IsFalse(sumConstraint.containsVariable(&var1));
	}

	TEST_METHOD(CloneShouldContainTheSameVariablesAsTheOriginal) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		SumConstraint sumConstraint({ &var1, &var2 }, 4);

		auto clone = sumConstraint.clone();

		Assert::IsTrue(clone->containsVariable(&var1));
		Assert::IsTrue(clone->containsVariable(&var2));

		delete clone;
	}

	};
}