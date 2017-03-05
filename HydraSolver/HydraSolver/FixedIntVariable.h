#pragma once

#include "IntVariable.h"

namespace hydra {

	class FixedIntVariable : public IntVariable {
	public:
		FixedIntVariable(const std::string& name, int value);
		~FixedIntVariable();

		void pushCurrentState() override;
		void popState() override;

		void filterValue(int value) override;
		void filterLowerBound(int newLowerBound) override;
		void filterUpperBound(int newUpperBound) override;
		int getLowerBound() const override;
		int getUpperBound() const override;
		bool containsValue(int value) const override;

	private:
		static std::string getErrorDescriptionForMehtod(const std::string& methodName);

		int value;
	};

} // namespace hydra

