#pragma once

#include "Variable.h"

class VariableImpl : public hydra::Variable {
public:
	explicit VariableImpl(const std::string& name = "test");
	~VariableImpl();

	std::string getFormattedDomain() const override;
	void pushCurrentState() override;
	void popState() override;
	int cardinality() const override;
	void instantiate() override;

	bool pushWasCalled;
	bool popWasCalled;
	bool formattedDomainWasCalled;
};

