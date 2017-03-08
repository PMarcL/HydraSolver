#pragma once

#include <vector>
#include "IntVariableIterator.h"

namespace hydra {

	class Variable {
	public:
		explicit Variable(const std::string& name = "Var-");
		virtual ~Variable();

		std::string getName() const;

		virtual std::string getFormattedDomain() const = 0;
		virtual void pushCurrentState() = 0;
		virtual void popState() = 0;
		virtual int cardinality() const = 0;
		virtual void instantiate() = 0;
		virtual int getInstantiatedValue() const = 0;
		virtual void filterValue(int value) = 0;
		virtual void filterLowerBound(int newLowerBound) = 0;
		virtual void filterUpperBound(int newUpperBound) = 0;
		virtual int getLowerBound() const = 0;
		virtual int getUpperBound() const = 0;
		virtual bool containsValue(int value) const = 0;
		virtual IntVariableIterator* iterator() = 0;

	protected:
		std::string name;
	};

} // namespace hydra
