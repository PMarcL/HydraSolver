#pragma once
#include "Constraint.h"

class ConstraintImpl : public hydra::Constraint {
public:
	ConstraintImpl();
	~ConstraintImpl();

	void filter() override;
	void filterDomains() override;
	void filterBounds() override;
};

