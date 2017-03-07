#include "stdafx.h"
#include "VariableImpl.h"

using namespace std;

VariableImpl::VariableImpl(const string& name) : hydra::Variable(name) {
}

VariableImpl::~VariableImpl() {
}

void VariableImpl::pushCurrentState() {
}

void VariableImpl::popState() {
}

int VariableImpl::cardinality() const {
	return 0;
}

void VariableImpl::instantiate() {
}

void VariableImpl::notifyObservers() const {
	notifyDomainChanged();
}
