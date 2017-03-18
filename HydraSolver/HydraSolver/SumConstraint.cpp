#include "SumConstraint.h"
#include "Variable.h"
#include "IntVariableIterator.h"
#include <list>
#include <unordered_set>
#include <unordered_map>

using namespace std;

namespace hydra {

	SumConstraint::SumConstraint(const vector<Variable*>& variables, int sum) : variables(variables), sum(sum) {
	}

	SumConstraint::~SumConstraint() {
	}

	bool SumConstraint::containsVariable(Variable* variable) const {
		return find(variables.begin(), variables.end(), variable) != variables.end();
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
		auto lowerBoundSum = 0;
		auto upperBoundSum = 0;

		for (size_t i = 0; i < variables.size(); i++) {
			lowerBoundSum += variables[i]->getLowerBound();
			upperBoundSum += variables[i]->getUpperBound();
		}

		for (size_t i = 0; i < variables.size(); i++) {
			lowerBoundSum -= variables[i]->getLowerBound();
			upperBoundSum -= variables[i]->getUpperBound();

			auto iterator = variables[i]->iterator();
			while (iterator->hasNextValue()) {
				auto currentValue = iterator->next();
				lowerBoundSum += currentValue;
				upperBoundSum += currentValue;

				if (sum < lowerBoundSum || sum > upperBoundSum) {
					variables[i]->filterValue(currentValue);
					if (modifiedVariables.empty() || modifiedVariables[modifiedVariables.size() - 1] != variables[i]) {
						modifiedVariables.push_back(variables[i]);
					}
				}

				lowerBoundSum -= currentValue;
				upperBoundSum -= currentValue;
			}
			satisfied = satisfied && variables[i]->cardinality() != 0;
			delete iterator;

			lowerBoundSum += variables[i]->getLowerBound();
			upperBoundSum += variables[i]->getUpperBound();
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
			unordered_map<int, TrickNode*> nodesAtThisLevel; // used to prevent creating multiple nodes for the same value at every level.
			for (size_t i = 0; i < nodeToProcess; i++) {
				auto currentNode = nodeQueue.front();
				nodeQueue.pop_front();

				auto iterator = variable->iterator();
				for (auto v = 0; v < variable->cardinality(); v++) {
					auto currentValue = iterator->next();
					TrickNode *child;
					auto iteratorToNode = nodesAtThisLevel.find(currentValue);

					if (iteratorToNode == nodesAtThisLevel.end()) {
						child = new TrickNode(currentNode->value + currentValue);
					} else {
						child = (*iteratorToNode).second;
					}


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
			unordered_set<int> valuesToKeep;
			auto nodeToProcess = nodeQueue.size();

			for (size_t i = 0; i < nodeToProcess; i++) {
				auto currentNode = nodeQueue.front();
				nodeQueue.pop_front();

				valuesToKeep.insert(currentNode->parent->varValue);
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
