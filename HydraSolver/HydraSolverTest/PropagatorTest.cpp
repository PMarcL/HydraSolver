#include "stdafx.h"
#include "CppUnitTest.h"
#include "Propagator.h"
#include "BitsetIntVariable.h"
#include "SumConstraint.h"
#include "ConstraintImpl.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(PropagatorTest) {
public:
	TEST_METHOD(ShouldOnlyCallFilterOnPropagateWithDefaultConfig) {
		ConstraintImpl constraint;
		Propagator propagator({ &constraint });

		propagator.propagate();

		Assert::IsTrue(constraint.filterWasCalled);
		Assert::IsFalse(constraint.filterDomainWasCalled);
		Assert::IsFalse(constraint.filterBoundsWasCalled);
	}

	TEST_METHOD(ShouldOnlyCallFilterDomainOnPropagateWithDomainFilteringConfig) {
		ConstraintImpl constraint;
		Propagator propagator({ &constraint }, DOMAIN_CONSISTENCY);

		propagator.propagate();

		Assert::IsFalse(constraint.filterWasCalled);
		Assert::IsTrue(constraint.filterDomainWasCalled);
		Assert::IsFalse(constraint.filterBoundsWasCalled);
	}

	TEST_METHOD(ShouldOnlyCallFilterBoundsOnPropagateWithDomainFilteringConfig) {
		ConstraintImpl constraint;
		Propagator propagator({ &constraint }, BOUND_CONSISTENCY);

		propagator.propagate();

		Assert::IsFalse(constraint.filterWasCalled);
		Assert::IsFalse(constraint.filterDomainWasCalled);
		Assert::IsTrue(constraint.filterBoundsWasCalled);
	}

	TEST_METHOD(GivenMultipleConstraintsShouldCallFilterOnAllOfThem) {
		ConstraintImpl c1;
		ConstraintImpl c2;
		ConstraintImpl c3;
		Propagator propagator({ &c1, &c2, &c3 });

		propagator.propagate();

		Assert::IsTrue(c1.filterWasCalled);
		Assert::IsTrue(c2.filterWasCalled);
		Assert::IsTrue(c3.filterWasCalled);
	}

	TEST_METHOD(GivenConsistentStatePropagationShouldReturnLocalConsistancyWithDefaultConfig) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		SumConstraint sumConstraint({ &var1, &var2 }, 4);

		Propagator propagator({ &sumConstraint });
		auto result = propagator.propagate();

		Assert::IsTrue(result == LOCAL_CONSISTENCY);
	}

	TEST_METHOD(GivenInconsistentStatePropagationShouldReturnInconsistentStateWithDefaultConfig) {
		BitsetIntVariable var1("var1", 1, 2);
		BitsetIntVariable var2("var2", 1, 3);
		SumConstraint sumConstraint({ &var1, &var2 }, 6);

		Propagator propagator({ &sumConstraint });
		auto result = propagator.propagate();

		Assert::IsTrue(result == INCONSISTENT_STATE);
	}

	TEST_METHOD(GivenConsistentStatePropagationShouldReturnLocalConsistancyWithDomainConsistencyConfig) {
		BitsetIntVariable var1("var1", 4, 12);
		BitsetIntVariable var2("var2", 1, 4);
		SumConstraint sumConstraint({ &var1, &var2 }, 8);

		Propagator propagator({ &sumConstraint }, DOMAIN_CONSISTENCY);
		auto result = propagator.propagate();

		Assert::IsTrue(result == LOCAL_CONSISTENCY);
	}

	TEST_METHOD(GivenInconsistentStatePropagationShouldReturnInconsistentStateWithDomainConsistencyConfig) {
		BitsetIntVariable var1("var1", 4, 5);
		BitsetIntVariable var2("var2", 1, 3);
		SumConstraint sumConstraint({ &var1, &var2 }, 2);

		Propagator propagator({ &sumConstraint }, DOMAIN_CONSISTENCY);
		auto result = propagator.propagate();

		Assert::IsTrue(result == INCONSISTENT_STATE);
	}

	TEST_METHOD(GivenConsistentStatePropagationShouldReturnLocalConsistancyWithBoundConsistencyConfig) {
		BitsetIntVariable var1("var1", 4, 12);
		BitsetIntVariable var2("var2", 1, 4);
		SumConstraint sumConstraint({ &var1, &var2 }, 8);

		Propagator propagator({ &sumConstraint }, BOUND_CONSISTENCY);
		auto result = propagator.propagate();

		Assert::IsTrue(result == LOCAL_CONSISTENCY);
	}

	TEST_METHOD(GivenInconsistentStatePropagationShouldReturnInconsistentStateWithBoundConsistencyConfig) {
		BitsetIntVariable var1("var1", 4, 5);
		BitsetIntVariable var2("var2", 1, 3);
		SumConstraint sumConstraint({ &var1, &var2 }, 2);

		Propagator propagator({ &sumConstraint }, BOUND_CONSISTENCY);
		auto result = propagator.propagate();

		Assert::IsTrue(result == INCONSISTENT_STATE);
	}

	TEST_METHOD(GivenMultipleConsistentConstraintsShouldFilterAllOfThemAndReturnLocalConsistency) {
		BitsetIntVariable var1("var1", 4, 6);
		BitsetIntVariable var2("var2", 1, 3);
		BitsetIntVariable var3("var2", 5, 10);
		BitsetIntVariable var4("var2", 3, 8);
		SumConstraint sumConstraint({ &var1, &var2, &var3, &var4 }, 19);

		Propagator propagator({ &sumConstraint });
		auto result = propagator.propagate();

		Assert::IsTrue(result == LOCAL_CONSISTENCY);
	}

	TEST_METHOD(GivenMultipleInconsistentConstraintsShouldFilterAllOfThemAndReturnInconsistentState) {
		BitsetIntVariable var1("var1", 4, 6);
		BitsetIntVariable var2("var2", 1, 3);
		BitsetIntVariable var3("var2", 5, 10);
		BitsetIntVariable var4("var2", 3, 8);
		SumConstraint sumConstraint({ &var1, &var2, &var3, &var4 }, 10);

		Propagator propagator({ &sumConstraint });
		auto result = propagator.propagate();

		Assert::IsTrue(result == INCONSISTENT_STATE);
	}
	};
}