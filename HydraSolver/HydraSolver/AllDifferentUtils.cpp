#include "AllDifferentUtils.h"
#include "Variable.h"
#include <unordered_map>
#include <algorithm>

using namespace std;

namespace hydra {
	vector<AllDiffEdge*> findPathFromSourceToSink(const vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target);
	AllDiffEdge* findOrCreateInvertedEdge(AllDiffEdge* edge);

	bool edgeResidualCapacityCompare(AllDiffEdge *edge1, AllDiffEdge *edge2) {
		return edge1->residualCapacity < edge2->residualCapacity;
	}

	void FordFulkersonAlgorithm(const vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target) {
		auto currentPath = findPathFromSourceToSink(nodes, source, target);
		while (currentPath.size() > 0) {
			auto minResidualCapacity = (*min_element(currentPath.begin(), currentPath.end(), [](AllDiffEdge *edge1, AllDiffEdge *edge2) {
				return edge1->residualCapacity < edge2->residualCapacity;
			}))->residualCapacity;

			for (auto edge : currentPath) {
				edge->flow += minResidualCapacity;
				edge->residualCapacity = edge->initialCapacity - edge->flow;

				auto invertedEdge = findOrCreateInvertedEdge(edge);
				invertedEdge->flow -= minResidualCapacity;
				invertedEdge->residualCapacity = invertedEdge->initialCapacity - invertedEdge->flow;
			}

			currentPath = findPathFromSourceToSink(nodes, source, target);
		}
	}

	vector<AllDiffEdge*> findPathFromSourceToSink(const vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target) {
		for (auto node : nodes) {
			node->visited = false;
			node->parent = nullptr;
		}

		list<AllDiffNode*> nodeStack;
		nodeStack.push_back(source);
		while (!nodeStack.empty()) {
			auto currentNode = nodeStack.back();
			currentNode->visited = true;
			nodeStack.pop_back();

			if (currentNode->type == SINK) {
				break;
			}

			for (auto edge : currentNode->adjencyList) {
				if (!edge->to->visited && edge->residualCapacity > 0) {
					edge->to->parent = edge;
					nodeStack.push_back(edge->to);
				}
			}
		}

		if (target->parent == nullptr) {
			return vector<AllDiffEdge*>();
		}

		vector<AllDiffEdge*> path;
		auto currentNode = target;
		while (currentNode->type != SOURCE) {
			path.push_back(currentNode->parent);
			currentNode = currentNode->parent->from;
		}
		return path;
	}

	AllDiffEdge* findOrCreateInvertedEdge(AllDiffEdge* edge) {
		auto node = edge->to;
		AllDiffEdge* invertedEdge = nullptr;
		for (auto e : node->adjencyList) {
			if (e->to == edge->from) {
				invertedEdge = e;
			}
		}

		if (invertedEdge == nullptr) {
			invertedEdge = new AllDiffEdge(0, node, edge->from);
			node->adjencyList.push_back(invertedEdge);
		}
		return invertedEdge;
	}

	void visitNodeForwardDirection(AllDiffNode* node, list<AllDiffNode*>& visitedNodes);
	void reverseGraph(const vector<AllDiffNode*>& nodes);
	void visitNodeBackwardDirection(AllDiffNode* node, unordered_set<AllDiffNode*>& connectedNodes);

	vector<unordered_set<AllDiffNode*> > KosarajuAlgorithm(const vector<AllDiffNode*>& nodes) {
		for (auto node : nodes) {
			node->visited = false;
		}

		list<AllDiffNode*> visitedNodes;
		for (auto node : nodes) {
			if (node->visited) {
				continue;
			}
			visitNodeForwardDirection(node, visitedNodes);
		}

		reverseGraph(nodes);

		for (auto node : nodes) {
			node->visited = false;
		}

		vector<unordered_set<AllDiffNode*> > stronglyConnectedComponents;
		for (auto node : visitedNodes) {
			if (node->visited) {
				continue;
			}
			unordered_set<AllDiffNode*> connectedNodes;
			visitNodeBackwardDirection(node, connectedNodes);
			stronglyConnectedComponents.push_back(connectedNodes);
		}

		reverseGraph(nodes);

		return stronglyConnectedComponents;
	}

	void visitNodeForwardDirection(AllDiffNode* node, list<AllDiffNode*>& visitedNodes) {
		node->visited = true;
		for (auto edge : node->adjencyList) {
			if (!edge->to->visited && edge->residualCapacity > 0) {
				visitNodeForwardDirection(edge->to, visitedNodes);
			}
		}
		visitedNodes.push_front(node);
	}

	void reverseGraph(const vector<AllDiffNode*>& nodes) {
		for (auto node : nodes) {
			for (auto edge : node->adjencyList) {
				edge->reversed = false;
			}
		}

		for (auto node : nodes) {
			for (auto it = node->adjencyList.begin(); it != node->adjencyList.end();) {
				auto currentEdge = *it;
				if (!currentEdge->reversed) {
					node->adjencyList.erase(it++);

					auto temp = currentEdge->to;
					currentEdge->to = currentEdge->from;
					currentEdge->from = temp;

					currentEdge->from->adjencyList.push_back(currentEdge);
					currentEdge->reversed = true;
				} else {
					++it;
				}
			}
		}
	}

	void visitNodeBackwardDirection(AllDiffNode* node, unordered_set<AllDiffNode*>& connectedNodes) {
		node->visited = true;
		for (auto edge : node->adjencyList) {
			if (!edge->to->visited && edge->residualCapacity > 0) {
				visitNodeBackwardDirection(edge->to, connectedNodes);
			}
		}
		connectedNodes.insert(node);
	}

	bool ReginAlgorithm(const vector<Variable*>& vars, unordered_set<Variable*>& filteredVariable) {
		auto source = new AllDiffNode(SOURCE);
		auto sink = new AllDiffNode(SINK);
		vector<AllDiffNode*> nodes;
		unordered_map<int, AllDiffNode*> valueNodes;

		// build the graph, instantiate a node for every variable and for every possible value
		for (auto var : vars) {
			auto varNode = new AllDiffNode(VARIABLE);
			nodes.push_back(varNode);
			varNode->var = var;

			auto edgeSourceToVar = new AllDiffEdge(1, source, varNode);
			source->adjencyList.push_back(edgeSourceToVar);

			auto iterator = var->iterator();
			for (auto i = 0; i < var->cardinality(); i++) {
				auto currentValue = iterator->next();
				auto valueNodeIt = valueNodes.find(currentValue);
				AllDiffNode *valueNode;

				if (valueNodeIt == valueNodes.end()) {
					valueNode = new AllDiffNode(VALUE);
					valueNode->value = currentValue;
					valueNodes.emplace(currentValue, valueNode);

					auto edgeValueToSink = new AllDiffEdge(1, valueNode, sink);
					valueNode->adjencyList.push_back(edgeValueToSink);
				} else {
					valueNode = (*valueNodeIt).second;
				}

				auto edgeVarToValue = new AllDiffEdge(1, varNode, valueNode);
				varNode->adjencyList.push_back(edgeVarToValue);
			}
			delete iterator;
		}

		for (auto value : valueNodes) {
			nodes.push_back(value.second);
		}
		nodes.push_back(source);
		nodes.push_back(sink);

		// at this point, all variable nodes are in the first n entry of the nodes vector (n being the number of variables)
		FordFulkersonAlgorithm(nodes, source, sink);

		// if the maximum flow of the graph is not equal to the number of variable, the all different constraint is not satisfiable
		size_t maxFlow = 0;
		for (auto value : valueNodes) {
			for (auto edge : value.second->adjencyList) {
				if (edge->to->type == SINK) {
					maxFlow += edge->flow;
				}
			}
		}
		if (maxFlow != vars.size()) {
			deleteGraph(nodes);
			return false;
		}

		auto connectedComponents = KosarajuAlgorithm(nodes);

		for (size_t i = 0; i < vars.size(); i++) {
			auto currentNode = nodes[i];
			for (auto edge : currentNode->adjencyList) {
				if (edge->flow == 0) {
					unordered_set<AllDiffNode*> connectedNodes;
					for (auto component : connectedComponents) {
						if (component.find(currentNode) != component.end()) {
							connectedNodes = component;
							break;
						}
					}

					// if the value node is not in the same strongly connected nodes set, filter the value from the variable
					if (connectedNodes.find(edge->to) == connectedNodes.end()) {
						currentNode->var->filterValue(edge->to->value);
						if (filteredVariable.find(currentNode->var) == filteredVariable.end()) {
							filteredVariable.insert(currentNode->var);
						}
					}
				}
			}
		}

		deleteGraph(nodes);
		return true;
	}

	void deleteGraph(const vector<AllDiffNode*>& nodes) {
		for (auto node : nodes) {
			for (auto edge : node->adjencyList) {
				delete edge;
			}
			delete node;
		}
	}
}
