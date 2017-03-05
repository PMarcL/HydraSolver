#pragma once

#include <string>
#include "Variable.h"

namespace hydra {

	class IntVariable : public Variable {
	public:
		explicit IntVariable(const std::string& = "IntVariable-");
		virtual ~IntVariable();

		virtual void filterValue(int value) = 0;
		virtual void filterLowerBound(int newLowerBound) = 0;
		virtual void filterUpperBound(int newUpperBound) = 0;
		virtual int getLowerBound() const = 0;
		virtual int getUpperBound() const = 0;
		virtual bool containsValue(int value) const = 0;
	};

} // namespace hydra
