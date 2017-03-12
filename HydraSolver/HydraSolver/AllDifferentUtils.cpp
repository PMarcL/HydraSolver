#include "AllDifferentUtils.h"

using namespace std;

namespace hydra {
	vector<AllDiffEdge*> findPathFromSourceToSink(const vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target);
	AllDiffEdge* findOrCreateInvertedEdge(AllDiffEdge* edge);

	void FordFulkersonAlgorithm(const vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target) {
		auto currentPath = findPathFromSourceToSink(nodes, source, target);
		while (currentPath.size() > 0) {
			auto minResidualCapacity = currentPath[0]->residualCapacity;
			for (auto edge : currentPath) {
				if (edge->residualCapacity < minResidualCapacity) {
					minResidualCapacity = edge->residualCapacity;
				}
			}
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
			if (!edge->to->visited) {
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
			if (!edge->to->visited) {
				visitNodeBackwardDirection(edge->to, connectedNodes);
			}
		}
		connectedNodes.insert(node);
	}

	void ReginAlgorithm(const vector<Variable*>& vars) {
		// TODO : implement and test this function
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
