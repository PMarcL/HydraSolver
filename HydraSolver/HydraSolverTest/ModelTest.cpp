#include "stdafx.h"
#include "CppUnitTest.h"
#include "Model.h"
#include "Constraint.h"
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
		m.postConstraint(new Constraint);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, m.getConstraints().size());
	}

	TEST_METHOD(ShouldBeAbleToPostMultipleConstraints) {
		Model m;
		m.postConstraints({ new Constraint, new Constraint, new Constraint });
		size_t expectedSize = 3;
		Assert::AreEqual(expectedSize, m.getConstraints().size());
	}

	TEST_METHOD(ShouldNotHaveVariableOnCreation) {
		Model m;
		Assert::IsTrue(m.getVariables().empty());
	}

	TEST_METHOD(ShouldBeAbleToAddVariable) {
		Model m;
		m.addVariable(new VariableImpl);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, m.getVariables().size());
	}

	};
}