#pragma once

#include <vector>
#include <stack>
#include "Variable.h"
#include "SumConstraint.h"

namespace hydra {
	class BitsetIntVariable : public Variable {
	public:
		BitsetIntVariable(const std::string& name, int lowerBound, int upperBound);

		int getOriginalLowerBound() const;
		size_t getOriginalSize() const;
		const std::vector<uint8_t>& getBitset() const;
		bool mergeBitset(uint8_t *bitset);

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
		std::vector<uint8_t>* getBitSet();
		bool containsValue(int value) const override;
		IntVariableIterator* iterator() override;
		Variable* clone() const override;

	private:
		class BitsetIterator : public IntVariableIterator {
		public:
			BitsetIterator(std::vector<uint8_t>* bitset, int originalLowerBound, int originalCardinality);
			int next() override;
			int previous() override;
			bool hasNextValue() const override;

		private:
			int offset;
			int counter;
			int cardinalityAtCreation;
			int originalLowerBound;
			std::vector<uint8_t>* bitset;
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
		void updateLowerBound();
		void updateUpperBound();
		friend class SumConstraint; // needed for low level gpu access

		std::stack<std::vector<BitsetAction>> statesStack;
		std::vector<BitsetAction> currentRemovedValues;
		std::vector<uint8_t> bitset;
		int currentLowerBound;
		int currentUpperBound;
		int originalLowerBound; // needed to find index in the bitset after filtering the original lower bound

	};
} // namespace hydra
