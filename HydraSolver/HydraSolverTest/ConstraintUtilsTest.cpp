#include "stdafx.h"
#include "CppUnitTest.h"
#include "Model.h"
#include "ConstraintUtils.h"
#include "SumConstraint.h"
#include "AllDifferent.h"
#include "VariableImpl.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(ConstraintUtilsTest) {
public:
	TEST_METHOD(CreateSumConstraintShouldReturnAValidSumWithoutThrowing) {
		size_t expectedNumberOfConstraint = 1;
		Model m;

		auto sum = CreateSumConstraint({ new VariableImpl, new VariableImpl }, 4);
		m.postConstraint(sum);

		Assert::AreEqual(expectedNumberOfConstraint, m.getNumberOfConstraints());
	}

	TEST_METHOD(CreateAllDifferentConstraintShouldReturnAValidAllDifferentWithoutThrowing) {
		size_t expectedNumberOfConstraint = 1;
		Model m;

		auto allDiff = CreateAllDifferentConstraint({ new VariableImpl, new VariableImpl });
		m.postConstraint(allDiff);

		Assert::AreEqual(expectedNumberOfConstraint, m.getNumberOfConstraints());
	}

	};
}