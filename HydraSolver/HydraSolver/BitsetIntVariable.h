#pragma once

#include <vector>
#include <stack>
#include "IntVariable.h"

namespace hydra {
	class BitsetIntVariable : public IntVariable {
	public:
		BitsetIntVariable(const std::string& name, int lowerBound, int upperBound);

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
		class BitsetIterator : public IntVariableIterator {
		public:
			BitsetIterator(std::vector<bool>* bitset, int originalLowerBound);
			int next() override;
			int previous() override;

		private:
			int offset;
			int originalLowerBound;
			std::vector<bool>* bitset;
		};

		enum FilterActions {
			LOWER_BOUND_CHANGED,
			UPPER_BOUND_CHANGED,
			VALUE_REMOVED
		};

		struct BitsetAction {
			BitsetAction(int value, FilterActions action) : value(value), action(action) {
			}

			int value;
			FilterActions action;
		};

		void reinsertValues();

		std::stack<std::vector<BitsetAction>> statesStack;
		std::vector<BitsetAction> currentRemovedValues;
		std::vector<bool> bitset;
		int currentLowerBound;
		int currentUpperBound;
		int originalLowerBound; // needed to find index in the bitset after filtering the original lower bound
	};
} // namespace hydra