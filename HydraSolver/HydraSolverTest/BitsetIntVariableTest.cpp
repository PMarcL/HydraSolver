#include "stdafx.h"
#include "CppUnitTest.h"
#include "BitsetIntVariable.h"
#include "IllegalVariableOperationException.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(BitsetIntVariableTest) {
public:
	TEST_METHOD(ShouldThrowIllegalOperationOnCreationWithInconsistentBounds) {
		auto func = [] {
			BitsetIntVariable bitset("test", 10, 1);
		};
		Assert::ExpectException<IllegalVariableOperationException, void>(func);
	}

	TEST_METHOD(ShouldHaveGivenNameAndGivenBounds) {
		string expectedName = "test";
		auto lowerBound = 1;
		auto upperBound = 10;

		BitsetIntVariable bitset(expectedName, lowerBound, upperBound);

		Assert::AreEqual(expectedName, bitset.getName());
		Assert::AreEqual(lowerBound, bitset.getLowerBound());
		Assert::AreEqual(upperBound, bitset.getUpperBound());
	}

	TEST_METHOD(ShouldHaveCardinalityEqualToTheDifferenceBetweenUpperAndLowerBoundPlusOne) {
		auto expectedCardinality = 10;
		BitsetIntVariable bitset("test", 1, 10);
		Assert::AreEqual(expectedCardinality, bitset.cardinality());
	}

	TEST_METHOD(ShouldContainValueBetweenBoundsAfterCreation) {
		BitsetIntVariable bitset("test", 1, 10);
		Assert::IsTrue(bitset.containsValue(5));
	}

	TEST_METHOD(ShouldNotContainValueOutsideBoundAfterCreation) {
		BitsetIntVariable bitset("test", 1, 10);
		Assert::IsFalse(bitset.containsValue(0));
		Assert::IsFalse(bitset.containsValue(11));
	}

	TEST_METHOD(ShouldFilterLowerBoundWhenGivenAGreaterValueThanCurrentLowerBound) {
		auto expectedLowerBound = 5;
		BitsetIntVariable bitset("test", 1, 10);
		bitset.filterLowerBound(expectedLowerBound);
		Assert::AreEqual(expectedLowerBound, bitset.getLowerBound());
	}

	TEST_METHOD(ShouldRemoveValuesLowerThanNewLowerBoundWhenFilteringLowerBound) {
		BitsetIntVariable bitset("test", 1, 10);
		auto expectedLowerBound = 5;
		auto originalCardinality = bitset.cardinality();

		bitset.filterLowerBound(expectedLowerBound);

		Assert::IsFalse(bitset.containsValue(1));
		Assert::IsFalse(bitset.containsValue(2));
		Assert::IsFalse(bitset.containsValue(3));
		Assert::IsFalse(bitset.containsValue(4));
		Assert::AreEqual(originalCardinality - 4, bitset.cardinality());
	}

	TEST_METHOD(ShouldThrowIllegalOperationWhenFilteringLowerBoundWithValueLowerThanCurrentLowerBound) {
		auto func = [] {
			BitsetIntVariable bitset("test", 5, 10);
			bitset.filterLowerBound(1);
		};
		Assert::ExpectException<IllegalVariableOperationException, void>(func);
	}

	TEST_METHOD(ShouldEmptyItsDomainWhenFilteringLowerBoundWithValueGreaterThanUpperBound) {
		auto expectedCardinality = 0;
		BitsetIntVariable bitset("test", 1, 10);
		bitset.filterLowerBound(11);
		Assert::AreEqual(expectedCardinality, bitset.cardinality());
	}

	TEST_METHOD(ShouldFilterUpperBoundWhenGivenAValueLowerThanCurrentUpperBound) {
		auto expectedUpperBound = 5;
		BitsetIntVariable bitset("test", 1, 10);
		bitset.filterUpperBound(5);
		Assert::AreEqual(expectedUpperBound, bitset.getUpperBound());
	}

	TEST_METHOD(ShouldRemoveValuesGreaterThanNewUpperBoundWhenFilteringUpperBound) {
		BitsetIntVariable bitset("test", 1, 10);
		auto expectedUpperBound = 5;
		auto originalCardinality = bitset.cardinality();

		bitset.filterUpperBound(expectedUpperBound);

		Assert::IsFalse(bitset.containsValue(6));
		Assert::IsFalse(bitset.containsValue(7));
		Assert::IsFalse(bitset.containsValue(8));
		Assert::IsFalse(bitset.containsValue(9));
		Assert::IsFalse(bitset.containsValue(10));
		Assert::AreEqual(originalCardinality - 5, bitset.cardinality());
	}

	TEST_METHOD(ShouldThrowIllegalOperationWhenFilteringUpperBoundWithValueGreaterThanCurrentUpperBound) {
		auto func = [] {
			BitsetIntVariable bitset("test", 1, 10);
			bitset.filterUpperBound(11);
		};
		Assert::ExpectException<IllegalVariableOperationException, void>(func);
	}

	TEST_METHOD(ShouldEmptyItsDomainWhenFilteringUpperBoundWithValueLowerThanLowerBound) {
		auto expectedCardinality = 0;
		BitsetIntVariable bitset("test", 5, 10);
		bitset.filterUpperBound(1);
		Assert::AreEqual(expectedCardinality, bitset.cardinality());
	}

	TEST_METHOD(ShouldFilterValueInBounds) {
		auto expectedCardinality = 9;
		BitsetIntVariable bitset("test", 1, 10);
		bitset.filterValue(5);
		Assert::AreEqual(expectedCardinality, bitset.cardinality());
		Assert::IsFalse(bitset.containsValue(5));
	}

	TEST_METHOD(ShouldUpdateLowerBoundWhenFilteringItWithFilterValue) {
		auto expectedLowerBound = 2;
		BitsetIntVariable bitset("test", 1, 10);
		bitset.filterValue(1);
		Assert::AreEqual(expectedLowerBound, bitset.getLowerBound());
	}

	TEST_METHOD(ShouldUpdateLowerBoundWhenFilteringItWithFilterValueAndPushPop) {
		auto expectedLowerBound = 1;
		BitsetIntVariable bitset("test", expectedLowerBound, 10);
		bitset.filterValue(expectedLowerBound);

		bitset.pushCurrentState();
		Assert::AreEqual(2, bitset.getLowerBound());

		bitset.popState();
		Assert::AreEqual(expectedLowerBound, bitset.getLowerBound());
	}

	TEST_METHOD(ShouldUpdateUpperBoundWhenFilteringItWithFilterValue) {
		auto expectedUpperBound = 9;
		BitsetIntVariable bitset("test", 1, 10);
		bitset.filterValue(10);
		Assert::AreEqual(expectedUpperBound, bitset.getUpperBound());
	}

	TEST_METHOD(ShouldUpdateUpperBoundWhenFilteringItWithFilterValueAndPushPop) {
		auto expectedUpperBound = 10;
		BitsetIntVariable bitset("test", 1, expectedUpperBound);
		bitset.filterValue(expectedUpperBound);

		bitset.pushCurrentState();
		Assert::AreEqual(9, bitset.getUpperBound());

		bitset.popState();
		Assert::AreEqual(expectedUpperBound, bitset.getUpperBound());
	}

	TEST_METHOD(ShouldThrowIllegalOperationWhenTryingToFilterValueOutsideBound) {
		auto filterValueGreaterThanUpperBound = [] {
			BitsetIntVariable bitset("test", 1, 10);
			bitset.filterValue(11);
		};

		auto filterValueLowerThanLowerBound = [] {
			BitsetIntVariable bitset("test", 0, 10);
			bitset.filterValue(11);
		};

		Assert::ExpectException<IllegalVariableOperationException, void>(filterValueGreaterThanUpperBound);
		Assert::ExpectException<IllegalVariableOperationException, void>(filterValueLowerThanLowerBound);
	}

	TEST_METHOD(ShouldHaveACardinalityOfOneWhenInstantiated) {
		BitsetIntVariable bitset("test", 1, 10);
		bitset.instantiate();
		Assert::AreEqual(1, bitset.cardinality());
	}

	TEST_METHOD(AfterInstantiateShouldReturnLowerBound) {
		auto expectedValue = 4;
		BitsetIntVariable bitset("test", expectedValue, 10);
		bitset.instantiate();
		Assert::AreEqual(expectedValue, bitset.getInstantiatedValue());
	}

	TEST_METHOD(ShouldReturnToOriginalStateAfterPushAndPop) {
		BitsetIntVariable bitset("test", 1, 10);
		auto originalCardinality = bitset.cardinality();

		bitset.filterValue(5);
		bitset.pushCurrentState();

		Assert::AreEqual(originalCardinality - 1, bitset.cardinality());
		Assert::IsFalse(bitset.containsValue(5));

		bitset.popState();

		Assert::AreEqual(originalCardinality, bitset.cardinality());
		Assert::IsTrue(bitset.containsValue(5));
	}

	TEST_METHOD(ShouldReinsertAllValuesAfterFilteringLowerBoundPushAndPop) {
		BitsetIntVariable bitset("test", 1, 10);
		auto originalCardinality = bitset.cardinality();

		bitset.filterLowerBound(5);
		bitset.pushCurrentState();

		Assert::AreEqual(originalCardinality - 4, bitset.cardinality());
		Assert::IsFalse(bitset.containsValue(4));
		Assert::IsFalse(bitset.containsValue(3));
		Assert::IsFalse(bitset.containsValue(2));
		Assert::IsFalse(bitset.containsValue(1));

		bitset.popState();

		Assert::AreEqual(originalCardinality, bitset.cardinality());
		Assert::IsTrue(bitset.containsValue(4));
		Assert::IsTrue(bitset.containsValue(3));
		Assert::IsTrue(bitset.containsValue(2));
		Assert::IsTrue(bitset.containsValue(1));
	}

	TEST_METHOD(ShouldReinsertAllValuesAfterFilteringUpperBoundPushAndPop) {
		BitsetIntVariable bitset("test", 1, 10);
		auto originalCardinality = bitset.cardinality();

		bitset.filterUpperBound(5);
		bitset.pushCurrentState();

		Assert::AreEqual(originalCardinality - 5, bitset.cardinality());
		Assert::IsFalse(bitset.containsValue(6));
		Assert::IsFalse(bitset.containsValue(7));
		Assert::IsFalse(bitset.containsValue(8));
		Assert::IsFalse(bitset.containsValue(9));
		Assert::IsFalse(bitset.containsValue(10));

		bitset.popState();

		Assert::AreEqual(originalCardinality, bitset.cardinality());
		Assert::IsTrue(bitset.containsValue(6));
		Assert::IsTrue(bitset.containsValue(7));
		Assert::IsTrue(bitset.containsValue(8));
		Assert::IsTrue(bitset.containsValue(9));
		Assert::IsTrue(bitset.containsValue(10));
	}

	TEST_METHOD(ShouldReinsertAllValuesAfterMultipleActionsPushAndPop) {
		BitsetIntVariable bitset("test", 1, 10);
		auto originalCardinality = bitset.cardinality();

		bitset.filterUpperBound(9);
		bitset.filterValue(5);
		bitset.filterLowerBound(3);
		bitset.pushCurrentState();

		Assert::AreEqual(originalCardinality - 4, bitset.cardinality());
		Assert::IsFalse(bitset.containsValue(10));
		Assert::IsFalse(bitset.containsValue(5));
		Assert::IsFalse(bitset.containsValue(2));
		Assert::IsFalse(bitset.containsValue(1));

		bitset.popState();

		Assert::AreEqual(originalCardinality, bitset.cardinality());
		Assert::IsTrue(bitset.containsValue(10));
		Assert::IsTrue(bitset.containsValue(5));
		Assert::IsTrue(bitset.containsValue(2));
		Assert::IsTrue(bitset.containsValue(1));
	}

	TEST_METHOD(ShouldReinsertAllValuesAfterMultipleActionsAndMultiplePushAndPop) {
		BitsetIntVariable bitset("test", 1, 10);
		auto originalCardinality = bitset.cardinality();

		bitset.filterUpperBound(9);
		bitset.pushCurrentState();
		Assert::IsFalse(bitset.containsValue(10));

		bitset.filterValue(5);
		bitset.pushCurrentState();
		Assert::IsFalse(bitset.containsValue(5));

		bitset.filterLowerBound(3);
		bitset.pushCurrentState();
		Assert::IsFalse(bitset.containsValue(2));
		Assert::IsFalse(bitset.containsValue(1));

		Assert::AreEqual(originalCardinality - 4, bitset.cardinality());

		bitset.popState();
		Assert::IsTrue(bitset.containsValue(2));
		Assert::IsTrue(bitset.containsValue(1));

		bitset.popState();
		Assert::IsTrue(bitset.containsValue(5));

		bitset.popState();
		Assert::AreEqual(originalCardinality, bitset.cardinality());
		Assert::IsTrue(bitset.containsValue(10));
	}

	TEST_METHOD(IteratorNextShouldGiveValuesOfSetBits) {
		BitsetIntVariable bitset("test", 1, 5);
		bitset.filterValue(2);
		bitset.filterValue(4);

		auto iterator = bitset.iterator();

		Assert::AreEqual(1, iterator->next());
		Assert::AreEqual(3, iterator->next());
		Assert::AreEqual(5, iterator->next());

		delete iterator;
	}

	TEST_METHOD(IteratorPreviousShouldGiveValuesOfSetBits) {
		BitsetIntVariable bitset("test", 1, 5);
		bitset.filterValue(2);
		bitset.filterValue(4);

		auto iterator = bitset.iterator();

		Assert::AreEqual(1, iterator->previous());
		Assert::AreEqual(5, iterator->previous());
		Assert::AreEqual(3, iterator->previous());

		delete iterator;
	}

	TEST_METHOD(IteratorShouldWrapWhenNextIteratePastTheEnd) {
		BitsetIntVariable bitset("test", 5, 6);
		auto iterator = bitset.iterator();

		iterator->next();
		iterator->next();

		Assert::AreEqual(5, iterator->next());

		delete iterator;
	}

	TEST_METHOD(IteratorShouldWrapWhenPreviousIteratePastTheBeginning) {
		BitsetIntVariable bitset("test", 7, 8);
		auto iterator = bitset.iterator();

		iterator->previous();

		Assert::AreEqual(8, iterator->previous());

		delete iterator;
	}

	TEST_METHOD(IteratorShouldHaveValueAtCreationIfBitsetIsNotEmpty) {
		BitsetIntVariable bitset("test", 7, 8);
		auto iterator = bitset.iterator();
		Assert::IsTrue(iterator->hasNextValue());
	}

	TEST_METHOD(IteratorShouldNotHaveValueAfterIteratingOverAllValues) {
		BitsetIntVariable bitset("test", 7, 8);
		auto iterator = bitset.iterator();

		iterator->next();
		iterator->next();

		Assert::IsFalse(iterator->hasNextValue());
	}
	};
}