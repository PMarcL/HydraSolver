#include "stdafx.h"
#include "CppUnitTest.h"
#include "VariableEnvironment.h"
#include "VariableImpl.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {
	TEST_CLASS(VariableEnvironmentTest) {
public:
	TEST_METHOD(ShouldNotHaveVariableOnCreation) {
		VariableEnvironment env;
		Assert::IsTrue(env.getVariables().empty());
	}

	TEST_METHOD(ShouldBeAbletoAddVariable) {
		VariableImpl var;
		VariableEnvironment env;
		env.addVariable(&var);
		size_t expectedSize = 1;
		Assert::AreEqual(expectedSize, env.getVariables().size());
	}

	TEST_METHOD(ShouldBeAbleToAddVariableArray) {
		VariableImpl var1;
		VariableImpl var2;
		VariableImpl var3;
		VariableEnvironment env;

		env.addVariableArray({ &var1, &var2, &var3 });

		size_t expectedSize = 3;
		Assert::AreEqual(expectedSize, env.getVariables().size());
	}

	TEST_METHOD(OnPushEnvironmentShouldCallPushOnAllVariables) {
		VariableImpl var1;
		VariableImpl var2;
		VariableImpl var3;
		VariableEnvironment env;

		env.addVariableArray({ &var1, &var2, &var3 });
		env.push();

		Assert::IsTrue(var1.pushWasCalled);
		Assert::IsTrue(var2.pushWasCalled);
		Assert::IsTrue(var3.pushWasCalled);
	}

	TEST_METHOD(OnPopEnvironmentShouldCallPopOnAllVariables) {
		VariableImpl var1;
		VariableImpl var2;
		VariableImpl var3;
		VariableEnvironment env;

		env.addVariableArray({ &var1, &var2, &var3 });
		env.pop();

		Assert::IsTrue(var1.popWasCalled);
		Assert::IsTrue(var2.popWasCalled);
		Assert::IsTrue(var3.popWasCalled);
	}

	};
}