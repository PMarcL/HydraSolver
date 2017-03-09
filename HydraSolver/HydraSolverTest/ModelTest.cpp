#include "stdafx.h"
#include "CppUnitTest.h"
#include "Model.h"
#include "ConstraintImpl.h"
#include "VariableImpl.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(ModelTest) {
public:
	TEST_METHOD(ShouldHaveGivenName) {
		string expectedName = "test";
		Model m(expectedName);
		Assert::AreEqual(expectedName, m.getName());
	}

	TEST_METHOD(ShouldNotHaveConstraintsOnCreation) {
		Model m;
		Assert::IsTrue(m.getConstraints().empty());
	}

	TEST_METHOD(ShouldBeAbleToPostConstraint) {
		Model m;
		m.postConstraint(new ConstraintImpl);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, m.getNumberOfConstraints());
	}

	TEST_METHOD(ShouldBeAbleToPostMultipleConstraints) {
		Model m;
		m.postConstraints({ new ConstraintImpl, new ConstraintImpl, new ConstraintImpl });
		size_t expectedSize = 3;
		Assert::AreEqual(expectedSize, m.getNumberOfConstraints());
	}

	TEST_METHOD(ShouldNotHaveVariableOnCreation) {
		Model m;
		Assert::IsTrue(m.getVariables().empty());
	}

	TEST_METHOD(ShouldBeAbleToAddVariable) {
		Model m;
		m.addVariable(new VariableImpl);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, m.getNumberOfVariables());
	}

	TEST_METHOD(ShouldBeAbleToAddVariableArray) {
		Model m;
		m.addVariableArray({ new VariableImpl, new VariableImpl, new VariableImpl });
		size_t expectedSize = 3;
		Assert::AreEqual(expectedSize, m.getNumberOfVariables());
	}

	TEST_METHOD(ShouldInstantiateAndAddVariableOnCreateVariableWithValue) {
		Model m;
		auto var = m.createIntVar("testVar", 12);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, m.getNumberOfVariables());
		Assert::IsNotNull(var);
	}

	TEST_METHOD(ShouldInstantiateAndAddVariableOnCreateVariableWithBounds) {
		Model m;
		auto var = m.createIntVar("testVar", 1, 10);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, m.getNumberOfVariables());
		Assert::IsNotNull(var);
	}

	TEST_METHOD(ShouldInstantiateAndAddVariableArrayOnCreateVariableArray) {
		Model m;
		size_t expectedSize = 10;
		auto vars = m.createIntVarArray("Test", expectedSize, 4, 8);
		Assert::AreEqual(expectedSize, m.getNumberOfVariables());
		Assert::AreEqual(expectedSize, vars.size());

		for (auto var : vars) {
			Assert::IsNotNull(var);
		}
	}

	TEST_METHOD(ShouldInstantiateAndAddVariableMatrixOnCreateVariableMatrix) {
		Model m;
		size_t expectedRowNumber = 10;
		size_t expectedColNumber = 15;

		auto vars = m.createIntVarMatrix("Test", expectedRowNumber, expectedColNumber, 5, 15);

		Assert::AreEqual(expectedRowNumber * expectedColNumber, m.getNumberOfVariables());
		Assert::AreEqual(expectedRowNumber, vars.size());
		Assert::AreEqual(expectedColNumber, vars[0].size());

		for (auto row : vars) {
			for (auto var : row) {
				Assert::IsNotNull(var);
			}
		}
	}

	};
}