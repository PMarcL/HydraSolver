#include "stdafx.h"
#include "CppUnitTest.h"
#include "VariableImpl.h"

using namespace std;
using namespace hydra;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {
	TEST_CLASS(VariableTest) {
public:

	TEST_METHOD(ShouldHaveGivenName) {
		string expectedName = "test";
		VariableImpl v(expectedName);
		Assert::AreEqual(expectedName, v.getName());
	}

	};
}