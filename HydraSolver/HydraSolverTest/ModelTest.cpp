#include "stdafx.h"
#include "CppUnitTest.h"
#include "Model.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {
	TEST_CLASS(ModelTest) {
public:

	TEST_METHOD(GetTestShouldReturnTest) {
		Model model(4);
		Assert::AreEqual(4, model.getTest());
	}

	};
}