#pragma once

#include "VariableObserver.h"

class VariableObserverImpl : public hydra::VariableObserver {
public:
	VariableObserverImpl();
	~VariableObserverImpl();

	void domainChanged() override;
	bool wasNotified() const;

private:
	bool notified;
};

