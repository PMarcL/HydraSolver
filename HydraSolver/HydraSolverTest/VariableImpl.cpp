#include "stdafx.h"
#include "VariableImpl.h"

using namespace std;

VariableImpl::VariableImpl(const string& name) : hydra::Variable(name) {
}

VariableImpl::~VariableImpl() {
}

void VariableImpl::pushCurrentState() {
	return;
}

void VariableImpl::popState() {
	return;
}

