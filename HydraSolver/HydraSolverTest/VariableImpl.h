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
	int getInstantiatedValue() const override;
	void filterValue(int value) override;
	void filterLowerBound(int newLowerBound) override;
	void filterUpperBound(int newUpperBound) override;
	int getLowerBound() const override;
	int getUpperBound() const override;
	bool containsValue(int value) const override;
	hydra::IntVariableIterator* iterator() override;

	bool pushWasCalled;
	bool popWasCalled;
	bool formattedDomainWasCalled;
};

