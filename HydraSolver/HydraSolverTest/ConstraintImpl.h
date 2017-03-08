#pragma once

#include "Constraint.h"

class ConstraintImpl : public hydra::Constraint {
public:
	ConstraintImpl();
	~ConstraintImpl();

	bool containsVariable(hydra::Variable* variable) const override;
	void filter() override;
	void filterDomains() override;
	void filterBounds() override;
};

