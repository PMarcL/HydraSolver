#include "stdafx.h"
#include "VariableImpl.h"

using namespace std;

VariableImpl::VariableImpl(const string& name) : hydra::Variable(name), pushWasCalled(false), popWasCalled(false), formattedDomainWasCalled(false) {
}

VariableImpl::~VariableImpl() {
}

string VariableImpl::getFormattedDomain() const {
	// this is just a hack to change the value of the formattedDomainWasCalled field to true in a const method
	auto ptr = (bool*)(&formattedDomainWasCalled);
	*ptr = true;

	return "test";
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
