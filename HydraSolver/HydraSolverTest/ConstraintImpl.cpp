#include "stdafx.h"
#include "ConstraintImpl.h"

using namespace hydra;
using namespace std;

ConstraintImpl::ConstraintImpl() : filterWasCalled(false), filterDomainWasCalled(false), filterBoundsWasCalled(false) {
}

ConstraintImpl::~ConstraintImpl() {
}

bool ConstraintImpl::containsVariable(hydra::Variable*) const {
	return false;
}

vector<hydra::Variable*> ConstraintImpl::filter() {
	filterWasCalled = true;
	return vector<hydra::Variable*>();
}

vector<hydra::Variable*> ConstraintImpl::filterBounds() {
	filterBoundsWasCalled = true;
	return vector<hydra::Variable*>();
}

vector<hydra::Variable*> ConstraintImpl::filterDomains() {
	filterDomainWasCalled = true;
	return vector<hydra::Variable*>();
}

bool ConstraintImpl::isSatisfied() const {
	return true;
}

void ConstraintImpl::replaceVariable(hydra::Variable* varToReplace, hydra::Variable* replacement) {
}

Constraint* ConstraintImpl::clone() const {
	return new ConstraintImpl;
}
