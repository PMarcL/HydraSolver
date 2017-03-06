#pragma once
#include "Constraint.h"

class ConstraintImpl : public hydra::Constraint {
public:
	ConstraintImpl();
	~ConstraintImpl();

	std::vector<hydra::Variable*> getVariables() const override;
	void filter() override;
	void filterDomains() override;
	void filterBounds() override;
};

