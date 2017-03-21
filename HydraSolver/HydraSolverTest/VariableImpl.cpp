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

int VariableImpl::getInstantiatedValue() const {
	return 0;
}

void VariableImpl::filterValue(int value) {
}

void VariableImpl::filterLowerBound(int newLowerBound) {
}

void VariableImpl::filterUpperBound(int newUpperBound) {
}

int VariableImpl::getLowerBound() const {
	return 0;
}

bool VariableImpl::containsValue(int value) const {
	return false;
}

int VariableImpl::getUpperBound() const {
	return 0;
}

hydra::IntVariableIterator* VariableImpl::iterator() {
	return nullptr;
}

hydra::Variable* VariableImpl::clone() const {
	return new VariableImpl;
}


