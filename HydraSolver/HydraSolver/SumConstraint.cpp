#include "SumConstraint.h"
#include "Variable.h"
#include "IntVariableIterator.h"
#include <list>

using namespace std;

namespace hydra {

	SumConstraint::SumConstraint(const vector<Variable*>& variables, int sum) : variables(variables), sum(sum) {
	}

	SumConstraint::~SumConstraint() {
	}

	bool SumConstraint::containsVariable(Variable* variable) const {
		for (auto var : variables) {
			if (var == variable) {
				return true;
			}
		}
		return false;
	}

	vector<Variable*> SumConstraint::filter() {
		return CPUDomainFilteringAlgorithm();
	}

	vector<Variable*> SumConstraint::filterDomains() {
		return CPUDomainFilteringAlgorithm();
	}

	vector<Variable*> SumConstraint::filterBounds() {
		return CPUBoundsFilteringAlgorithm();
	}

	bool SumConstraint::isSatisfied() const {
		return satisfied;
	}

	vector<Variable*> SumConstraint::CPUBoundsFilteringAlgorithm() {
		vector<Variable*> modifiedVariables;
		satisfied = true;
		for (size_t i = 0; i < variables.size(); i++) {

			auto iterator = variables[i]->iterator();
			while (iterator->hasNextValue()) {
				auto currentValue = iterator->next();
				auto lowerBoundSum = currentValue;
				auto upperBoundSum = currentValue;

				for (size_t j = 0; j < variables.size(); j++) {
					if (j == i) {
						continue;
					}

					lowerBoundSum += variables[j]->getLowerBound();
					upperBoundSum += variables[j]->getUpperBound();
				}

				if (sum < lowerBoundSum || sum > upperBoundSum) {
					variables[i]->filterValue(currentValue);
					if (modifiedVariables.empty() || modifiedVariables[modifiedVariables.size() - 1] != variables[i]) {
						modifiedVariables.push_back(variables[i]);
					}
				}
			}
			satisfied = satisfied && variables[i]->cardinality() != 0;
			delete iterator;
		}
		return modifiedVariables;
	}

	// implementation of Trick algorithm
	struct TrickNode;

	struct TrickArc {
		TrickArc(int varValue, TrickNode* from, TrickNode* to) : varValue(varValue), from(from), to(to) {}
		int varValue;
		TrickNode* from;
		TrickNode* to;
	};

	struct TrickNode {
		explicit TrickNode(int value) : value(value), parent(nullptr) {}
		int value;
		TrickArc* parent;
		vector<TrickArc*> children;
	};

	vector<Variable*> SumConstraint::CPUDomainFilteringAlgorithm() {
		list<TrickNode*> nodeQueue;
		auto initialNode = new TrickNode(0);
		nodeQueue.push_back(initialNode);

		// building the graph
		for (auto variable : variables) {
			auto nodeToProcess = nodeQueue.size();
			for (size_t i = 0; i < nodeToProcess; i++) {
				auto currentNode = nodeQueue.front();
				nodeQueue.pop_front();

				auto iterator = variable->iterator();
				for (auto v = 0; v < variable->cardinality(); v++) {
					auto currentValue = iterator->next();
					auto child = new TrickNode(currentNode->value + currentValue);

					auto arc = new TrickArc(currentValue, currentNode, child);
					currentNode->children.push_back(arc);
					child->parent = arc;

					nodeQueue.push_back(child);
				}
				delete iterator;
			}
		}

		// nodeQueue here contains all the nodes at the last level
		// remove any node different than the expected sum
		for (auto it = nodeQueue.begin(); it != nodeQueue.end();) {
			if ((*it)->value != sum) {
				nodeQueue.erase(it++);
				if (it == nodeQueue.end()) {
					break;
				}
			} else {
				++it;
			}
		}

		// if nodeQueue is empty here, the constraint is unsatisfiable, otherwise it is
		satisfied = !nodeQueue.empty();

		vector<Variable*> filteredVariables;
		// Traversing the graph backward level by level, filtering values along the way
		for (auto variable = variables.rbegin(); variable != variables.rend(); ++variable) {
			vector<int> valuesToKeep;
			auto nodeToProcess = nodeQueue.size();

			for (size_t i = 0; i < nodeToProcess; i++) {
				auto currentNode = nodeQueue.front();
				nodeQueue.pop_front();

				valuesToKeep.push_back(currentNode->parent->varValue);
				nodeQueue.push_back(currentNode->parent->from);
			}

			auto iterator = (*variable)->iterator();
			auto originalCardinality = (*variable)->cardinality();
			for (auto v = 0; v < originalCardinality; v++) {
				auto currentValue = iterator->next();
				if (find(valuesToKeep.begin(), valuesToKeep.end(), currentValue) == valuesToKeep.end()) {
					(*variable)->filterValue(currentValue);

					if (filteredVariables.empty() || filteredVariables[filteredVariables.size() - 1] != *variable) {
						filteredVariables.push_back(*variable);
					}
				}
			}
			delete iterator;
		}

		// Traversing the graph forward to delete all the nodes
		nodeQueue.erase(nodeQueue.begin(), nodeQueue.end());
		nodeQueue.push_back(initialNode);
		while (!nodeQueue.empty()) {
			auto currentNode = nodeQueue.front();
			nodeQueue.pop_front();

			for (auto child : currentNode->children) {
				nodeQueue.push_back(child->to);
			}

			delete currentNode;
		}

		return filteredVariables;
	}

} // namespace hydra
