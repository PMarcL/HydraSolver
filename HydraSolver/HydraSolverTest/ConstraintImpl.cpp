#include "stdafx.h"
#include "ConstraintImpl.h"

using namespace hydra;
using namespace std;

ConstraintImpl::ConstraintImpl() {
}


ConstraintImpl::~ConstraintImpl() {
}

vector<Variable*> ConstraintImpl::getVariables() const {
	return vector<Variable*>();
}

void ConstraintImpl::filter() {
}

void ConstraintImpl::filterBounds() {
}

void ConstraintImpl::filterDomains() {
}
