#pragma once

#include "IntVariable.h"

namespace hydra {

	class FixedIntVariable : public IntVariable {
	public:
		FixedIntVariable(const std::string& name, int value);
		~FixedIntVariable();

		std::string getFormattedDomain() const override;
		void pushCurrentState() override;
		void popState() override;
		int cardinality() const override;

		void filterValue(int value) override;
		void filterLowerBound(int newLowerBound) override;
		void filterUpperBound(int newUpperBound) override;
		int getLowerBound() const override;
		int getUpperBound() const override;
		bool containsValue(int value) const override;
		void instantiate() override;
		IntVariableIterator* iterator() override;

	private:
		class FixedIntIterator : public IntVariableIterator {
		public:
			explicit FixedIntIterator(int value) : value(value), counter(0) {}

			int next() override {
				counter++;
				return value;
			}

			int previous() override {
				return value;
			}

			bool hasNextValue() const override {
				return counter > 0;
			}

		private:
			int value;
			int counter;
		};

		std::string getErrorDescriptionForMehtod(const std::string& methodName) const;

		int value;
	};

} // namespace hydra

