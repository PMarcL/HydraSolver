#pragma once

#include "Variable.h"

class VariableImpl : public hydra::Variable {
public:
	explicit VariableImpl(const std::string& name = "test");
	~VariableImpl();

	void pushCurrentState() override;
	void popState() override;
};

