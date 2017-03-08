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


void ConstraintImpl::filter() {
}

void ConstraintImpl::filterBounds() {
}

void ConstraintImpl::filterDomains() {
}
