#pragma once

#include "IntVariable.h"

namespace hydra {

	class FixedIntVariable : public IntVariable {
	public:
		FixedIntVariable(const std::string& name, int value);
		~FixedIntVariable();

		void pushCurrentState() override;
		void popState() override;
		int cardinality() const override;

		void filterValue(int value) override;
		void filterLowerBound(int newLowerBound) override;
		void filterUpperBound(int newUpperBound) override;
		int getLowerBound() const override;
		int getUpperBound() const override;
		bool containsValue(int value) const override;
		IntVariableIterator* iterator() override;

	private:
		class FixedIntIterator : public IntVariableIterator {
		public:
			explicit FixedIntIterator(int value) : value(value) {}

			int next() override {
				return value;
			}

			int previous() override {
				return value;
			}

		private:
			int value;
		};

		std::string getErrorDescriptionForMehtod(const std::string& methodName) const;

		int value;
	};

} // namespace hydra

