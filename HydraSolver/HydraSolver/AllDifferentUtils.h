#pragma once

#include <vector>
#include <list>
#include <unordered_set>

namespace hydra {

	class Variable;

	enum AllDiffNodeType {
		SOURCE,
		VARIABLE,
		VALUE,
		SINK
	};

	struct AllDiffNode;

	struct AllDiffEdge {
		AllDiffEdge(int capacity, AllDiffNode* from, AllDiffNode* to) : initialCapacity(capacity), residualCapacity(capacity), flow(0),
			reversed(false), from(from), to(to) {
		}

		int initialCapacity;
		int residualCapacity;
		int flow;
		bool reversed;
		AllDiffNode* from;
		AllDiffNode* to;
	};

	struct AllDiffNode {
		explicit AllDiffNode(AllDiffNodeType type) : type(type), var(nullptr), value(-1), visited(false), parent(nullptr) {}

		AllDiffNodeType type;
		Variable* var;
		int value;
		std::list<AllDiffEdge*> adjencyList;
		bool visited;
		AllDiffEdge* parent;
	};

	/*
	 * Calculates the maximum flow for the graph and updates all flow value of each edge.
	 */
	void FordFulkersonAlgorithm(const std::vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target);

	/*
	 * Returns a vector of set of strongly connected components of the given graph.
	 */
	std::vector<std::unordered_set<AllDiffNode*> > KosarajuAlgorithm(const std::vector<AllDiffNode*>& nodes);

	/*
	 * Verifies if the all different constraint can be satisfied with the given variable.
	 * Adds every filtered variable to the filteredVariable vector and returns true if the constraint is satisfiable.
	 */
	bool ReginAlgorithm(const std::vector<Variable*>& vars, std::unordered_set<Variable*>& filteredVariable);

	/*
	 * Deletes all nodes and edges of the given graph.
	 */
	void deleteGraph(const std::vector<AllDiffNode*>& nodes);
}
