#include "stdafx.h"
#include "ConstraintImpl.h"

using namespace hydra;
using namespace std;

ConstraintImpl::ConstraintImpl() {
}

ConstraintImpl::~ConstraintImpl() {
}

bool ConstraintImpl::containsVariable(hydra::Variable*) const {
	return false;
}

vector<hydra::Variable*> ConstraintImpl::filter() {
	return vector<hydra::Variable*>();
}

vector<hydra::Variable*> ConstraintImpl::filterBounds() {
	return vector<hydra::Variable*>();
}

vector<hydra::Variable*> ConstraintImpl::filterDomains() {
	return vector<hydra::Variable*>();
}

bool ConstraintImpl::isSatisfied() const {
	return false;
}

