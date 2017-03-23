#pragma once

#include "Constraint.h"

class ConstraintImpl : public hydra::Constraint {
public:
	ConstraintImpl();
	~ConstraintImpl();

	std::vector<hydra::Variable*> filter() override;
	std::vector<hydra::Variable*> filterDomains() override;
	std::vector<hydra::Variable*> filterBounds() override;
	bool isSatisfied() const override;
	void replaceVariable(hydra::Variable* varToReplace, hydra::Variable* replacement) override;
	Constraint* clone() const override;

	bool filterWasCalled;
	bool filterDomainWasCalled;
	bool filterBoundsWasCalled;
};

