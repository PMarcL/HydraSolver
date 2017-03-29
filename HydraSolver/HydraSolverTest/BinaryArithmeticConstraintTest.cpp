#include "stdafx.h"
#include "CppUnitTest.h"
#include "BinaryArithmeticConstraint.h"
#include "BitsetIntVariable.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(BinaryArithmeticConstraintTest) {
public:
	TEST_METHOD(ShouldContainGivenVariables) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, PLUS, EQ);

		Assert::IsTrue(constraint.containsVariable(&var1));
		Assert::IsTrue(constraint.containsVariable(&var2));
	}

	TEST_METHOD(ShouldReplaceGivenVariableAndNotContainReplacedVariable) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var3", 5, 6);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, PLUS, EQ);

		Assert::IsFalse(constraint.containsVariable(&var3));
	}

	TEST_METHOD(ShouldNotContainOtherVariable) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BitsetIntVariable var3("var3", 5, 6);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, PLUS, EQ);

		constraint.replaceVariable(&var2, &var3);

		Assert::IsTrue(constraint.containsVariable(&var3));
		Assert::IsFalse(constraint.containsVariable(&var2));
	}

	TEST_METHOD(CloneShouldContainSameVariableAsOriginal) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, PLUS, EQ);

		auto clone = constraint.clone();

		Assert::IsTrue(clone->containsVariable(&var1));
		Assert::IsTrue(clone->containsVariable(&var2));
	}

	TEST_METHOD(CloneShouldContainTheSameVariablesAsTheOriginal) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, PLUS, EQ);

		auto clone = constraint.clone();

		Assert::IsTrue(clone->containsVariable(&var1));
		Assert::IsTrue(clone->containsVariable(&var2));

		delete clone;
	}

	TEST_METHOD(ShouldFilterConsistentVariablesAndBeSatisfied) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, PLUS, EQ);

		auto filteredVariables = constraint.filterDomains();
		size_t expectedSize = 2;

		Assert::IsTrue(var1.cardinality() == 1);
		Assert::IsTrue(var1.containsValue(2));
		Assert::IsTrue(var2.cardinality() == 1);
		Assert::IsTrue(var2.containsValue(3));
		Assert::IsTrue((constraint.isSatisfied()));
		Assert::AreEqual(expectedSize, filteredVariables.size());
	}

	TEST_METHOD(ShouldFilterInconsistentVariablesAndNotBeSatisfied) {
		BitsetIntVariable var1("var1", 0, 2);
		BitsetIntVariable var2("var2", 0, 3);
		BinaryArithmeticConstraint constraint(&var1, &var2, 5, MINUS, EQ);

		constraint.filterDomains();

		Assert::IsFalse(constraint.isSatisfied());
	}

	TEST_METHOD(ShouldFilterBoundsAndBeSatisfiedForConsistentVariables) {
		BitsetIntVariable var1("var1", 3, 6);
		BitsetIntVariable var2("var2", 2, 7);
		BinaryArithmeticConstraint constraint(&var1, &var2, 30, MULTIPLIES, GEQ);

		auto filteredVariables = constraint.filterBounds();
		size_t expectedSize = 2;

		Assert::AreEqual(expectedSize, filteredVariables.size());
		Assert::IsTrue(constraint.isSatisfied());
		Assert::AreEqual(5, var1.getLowerBound());
		Assert::AreEqual(5, var2.getLowerBound());
	}

	TEST_METHOD(ShouldFilterBoundsAndBeSatisfiedForConsistentVariables_UsingGPU) {
		BitsetIntVariable var1("var1", 3, 6);
		BitsetIntVariable var2("var2", 2, 7);
		BinaryArithmeticConstraint constraint(&var1, &var2, 30, MULTIPLIES, GEQ);
		constraint.setGPUFilteringActive();

		auto filteredVariables = constraint.filterBounds();
		size_t expectedSize = 2;

		Assert::AreEqual(expectedSize, filteredVariables.size());
		Assert::IsTrue(constraint.isSatisfied());
		Assert::AreEqual(5, var1.getLowerBound());
		Assert::AreEqual(5, var2.getLowerBound());
	}

	TEST_METHOD(ShouldFilterBoundsAndNotBeSatisfiedForInconsistentVariables) {
		BitsetIntVariable var1("var1", 3, 6);
		BitsetIntVariable var2("var2", 2, 7);
		BinaryArithmeticConstraint constraint(&var1, &var2, 50, MULTIPLIES, GT);

		constraint.filterBounds();

		Assert::IsFalse(constraint.isSatisfied());
	}

	};
}