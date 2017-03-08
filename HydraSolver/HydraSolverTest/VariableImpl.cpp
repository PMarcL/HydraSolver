#include "stdafx.h"
#include "VariableImpl.h"

using namespace std;

VariableImpl::VariableImpl(const string& name) : hydra::Variable(name), pushWasCalled(false), popWasCalled(false) {
}

VariableImpl::~VariableImpl() {
}

void VariableImpl::pushCurrentState() {
	pushWasCalled = true;
}

void VariableImpl::popState() {
	popWasCalled = true;
}

int VariableImpl::cardinality() const {
	return 0;
}

void VariableImpl::instantiate() {
}
