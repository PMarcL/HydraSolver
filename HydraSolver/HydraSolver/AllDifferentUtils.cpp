#include "AllDifferentUtils.h"
#include <list>

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

	void ReginAlgorithm(const vector<Variable*>& vars) {
		// TODO : implement and test this function
	}
}
