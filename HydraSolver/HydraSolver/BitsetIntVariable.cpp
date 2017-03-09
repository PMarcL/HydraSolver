#include "BitsetIntVariable.h"
#include "IllegalVariableOperationException.h"
#include <string>

using namespace std;

namespace hydra {

	BitsetIntVariable::BitsetIntVariable(const string& name, int lowerBound, int upperBound) :
		Variable(name), statesStack(), currentRemovedValues(), bitset(upperBound - lowerBound + 1, true), currentLowerBound(lowerBound),
		currentUpperBound(upperBound), originalLowerBound(lowerBound) {
	}

	string BitsetIntVariable::getFormattedDomain() const {
		vector<int> valuesToPrint;
		for (size_t i = 0; i < bitset.size(); i++) {
			if (bitset[i]) {
				valuesToPrint.push_back(originalLowerBound + i);
			}
		}

		string formattedDomain = "{ ";
		for (size_t i = 0; i < valuesToPrint.size() - 1; i++) {
			formattedDomain += to_string(valuesToPrint[i]) + ", ";
		}
		formattedDomain += to_string(valuesToPrint[valuesToPrint.size() - 1]) + " }";
		return formattedDomain;
	}

	void BitsetIntVariable::pushCurrentState() {
		statesStack.push(vector<BitsetAction>(currentRemovedValues));
		currentRemovedValues = vector<BitsetAction>();
	}

	void BitsetIntVariable::popState() {
		currentRemovedValues = statesStack.top();
		statesStack.pop();
		reinsertValues();
	}

	void BitsetIntVariable::reinsertValues() {
		for (auto action : currentRemovedValues) {
			switch (action.action) {
			case VALUE_REMOVED:
				bitset[action.value - originalLowerBound] = true;
				if (action.value < currentLowerBound) {
					updateLowerBound();
				}
				if (action.value > currentUpperBound) {
					updateUpperBound();
				}
				break;
			case LOWER_BOUND_CHANGED:
				for (auto i = action.value - originalLowerBound; i < currentLowerBound - originalLowerBound; i++) {
					bitset[i] = true;
				}
				currentLowerBound = action.value;
				break;
			case UPPER_BOUND_CHANGED:
				for (auto i = currentUpperBound - originalLowerBound + 1; i <= action.value - originalLowerBound; i++) {
					bitset[i] = true;
				}
				currentUpperBound = action.value;
				break;
			default:
				break;
			}
		}
		currentRemovedValues = vector<BitsetAction>();
	}

	int BitsetIntVariable::cardinality() const {
		return count(bitset.begin(), bitset.end(), true);
	}

	void BitsetIntVariable::instantiate() {
		filterUpperBound(currentLowerBound);
	}

	int BitsetIntVariable::getInstantiatedValue() const {
		return currentLowerBound;
	}

	void BitsetIntVariable::filterValue(int value) {
		auto index = value - originalLowerBound;

		if (index < 0) {
			IllegalVariableOperationException e;
			e.setDescription("filterValue was called on a BitsetVariable (" + name + ") with a value lower than current lower bound.");
			throw e;
		}
		if (index >= bitset.size()) {
			IllegalVariableOperationException e;
			e.setDescription("filterValue was called on a BitsetVariable (" + name + ") with a value greater than current upper bound.");
			throw e;
		}

		bitset[index] = false;
		currentRemovedValues.push_back(BitsetAction(value, VALUE_REMOVED));

		if (value == currentLowerBound) {
			updateLowerBound();
		}

		if (value == currentUpperBound) {
			updateUpperBound();
		}
	}

	void BitsetIntVariable::updateLowerBound() {
		size_t index = 0;
		while (index < bitset.size() && !bitset[index]) {
			index++;
		}

		if (index < bitset.size()) {
			currentLowerBound = originalLowerBound + index;
		}
	}

	void BitsetIntVariable::updateUpperBound() {
		auto index = bitset.size() - 1;
		while (index > 0 && !bitset[index]) {
			index--;
		}

		if (index > 0) {
			currentUpperBound = originalLowerBound + index;
		}
	}

	void BitsetIntVariable::filterLowerBound(int newLowerBound) {
		if (newLowerBound < currentLowerBound) {
			IllegalVariableOperationException e;
			e.setDescription("filterLowerBound was called on a BitsetVariable (" + name + ") with a value lower than current lower bound.");
			throw e;
		}

		if (newLowerBound > currentUpperBound) {
			bitset.assign(bitset.size(), false);
		} else {
			currentRemovedValues.push_back(BitsetAction(currentLowerBound, LOWER_BOUND_CHANGED));

			for (auto i = 0; i < newLowerBound - originalLowerBound; i++) {
				bitset[i] = false;
			}

			currentLowerBound = newLowerBound;
		}
	}

	void BitsetIntVariable::filterUpperBound(int newUpperBound) {
		if (newUpperBound > currentUpperBound) {
			IllegalVariableOperationException e;
			e.setDescription("filterUpperBound was called on a BitsetVariable with a value greater than current upper bound.");
			throw e;
		}

		if (newUpperBound < currentLowerBound) {
			bitset.assign(bitset.size(), false);
		} else {
			currentRemovedValues.push_back(BitsetAction(currentUpperBound, UPPER_BOUND_CHANGED));

			for (auto i = newUpperBound - originalLowerBound + 1; i <= currentUpperBound - originalLowerBound; i++) {
				bitset[i] = false;
			}

			currentUpperBound = newUpperBound;
		}
	}

	int BitsetIntVariable::getLowerBound() const {
		return currentLowerBound;
	}

	int BitsetIntVariable::getUpperBound() const {
		return currentUpperBound;
	}

	bool BitsetIntVariable::containsValue(int value) const {
		return value >= currentLowerBound && value <= currentUpperBound && bitset[value - originalLowerBound];
	}

	IntVariableIterator* BitsetIntVariable::iterator() {
		return new BitsetIterator(&bitset, originalLowerBound, cardinality());
	}

	BitsetIntVariable::BitsetIterator::BitsetIterator(std::vector<bool>* bitset, int originalLowerBound, int originalCardinality) :
		offset(0), counter(0), cardinalityAtCreation(originalCardinality), originalLowerBound(originalLowerBound), bitset(bitset) {
		while (!(*bitset)[offset]) {
			offset++;
		}
	}

	int BitsetIntVariable::BitsetIterator::next() {
		auto value = originalLowerBound + offset;
		counter++;
		offset = (offset + 1) % bitset->size();
		while (!(*bitset)[offset]) {
			offset = (offset + 1) % bitset->size();
		}
		return value;
	}

	int BitsetIntVariable::BitsetIterator::previous() {
		auto value = originalLowerBound + offset;
		offset--;
		if (offset < 0) {
			offset += bitset->size();
		}
		while (!(*bitset)[offset]) {
			offset--;
			if (offset < 0) {
				offset += bitset->size();
			}
		}
		return value;
	}

	bool BitsetIntVariable::BitsetIterator::hasNextValue() const {
		return counter < cardinalityAtCreation;
	}

} // namespace hydra
