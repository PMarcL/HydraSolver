#include "stdafx.h"
#include "CppUnitTest.h"
#include "VariableSelector.h"
#include "BitsetIntVariable.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(VariableSelectorTest) {
public:
	TEST_METHOD(ShouldInstantiateSmallestDomainByDefault) {
		BitsetIntVariable smallestDomain("smallestDomain", 1, 5);
		BitsetIntVariable otherVar("test", 1, 10);
		VariableSelector varSelector;

		varSelector.instantiateVariable({ &smallestDomain, &otherVar });

		Assert::AreEqual(1, smallestDomain.cardinality());
		Assert::AreEqual(10, otherVar.cardinality());
	}

	TEST_METHOD(ShouldInstantiateRandomVariableGivenRandomHeuristic) {
		BitsetIntVariable var1("test", 1, 5);
		BitsetIntVariable var2("test", 1, 10);
		BitsetIntVariable var3("test", 1, 10);
		VariableSelector varSelector(RANDOM);

		varSelector.instantiateVariable({ &var1, &var2, &var3 });

		Assert::IsTrue(var1.cardinality() == 1 || var2.cardinality() == 1 || var3.cardinality() == 1);
	}

	};
}