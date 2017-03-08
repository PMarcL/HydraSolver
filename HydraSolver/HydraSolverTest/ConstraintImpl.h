#pragma once

#include "Constraint.h"

class ConstraintImpl : public hydra::Constraint {
public:
	ConstraintImpl();
	~ConstraintImpl();

	bool containsVariable(hydra::Variable* variable) const override;
	std::vector<hydra::Variable*> filter() override;
	std::vector<hydra::Variable*> filterDomains() override;
	std::vector<hydra::Variable*> filterBounds() override;
};

