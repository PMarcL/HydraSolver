#include "SumConstraint.h"
#include "IntVariable.h"
#include <algorithm>
#include <list>
#include <queue>

using namespace std;

namespace hydra {

	SumConstraint::SumConstraint(const vector<IntVariable*>& variables, int sum) : variables(variables), sum(sum) {
	}


	SumConstraint::~SumConstraint() {
	}

	void SumConstraint::filter() {
		CPUdomainFilteringAlgorithm();
	}

	void SumConstraint::filterBounds() {

	}

	void SumConstraint::filterDomains() {
		CPUdomainFilteringAlgorithm();
	}

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
		vector<TrickArc*> childs;
	};

	// implementation of Trick algorithm
	void SumConstraint::CPUdomainFilteringAlgorithm() {
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
					currentNode->childs.push_back(arc);
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

			for (auto child : currentNode->childs) {
				nodeQueue.push_back(child->to);
			}

			delete currentNode;
		}
	}


} // namespace hydra
