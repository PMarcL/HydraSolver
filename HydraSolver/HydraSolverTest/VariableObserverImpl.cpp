#include "stdafx.h"
#include "VariableObserverImpl.h"

VariableObserverImpl::VariableObserverImpl() : notified(false) {
}

VariableObserverImpl::~VariableObserverImpl() {
}

void VariableObserverImpl::domainChanged() {
	notified = true;
}

bool VariableObserverImpl::wasNotified() const {
	return notified;
}

