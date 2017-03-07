#include "stdafx.h"
#include "CppUnitTest.h"
#include "VariableImpl.h"
#include "VariableObserverImpl.h"

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

	TEST_METHOD(ShouldNotifyObserverOnDomainChanged) {
		VariableObserverImpl observer;
		VariableImpl variable;
		variable.addObserver(&observer);

		variable.notifyObservers();

		Assert::IsTrue(observer.wasNotified());
	}

	};
}