#pragma once

#include <vector>

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
		AllDiffEdge(int capacity, AllDiffNode* from, AllDiffNode* to) : initialCapacity(capacity), residualCapacity(capacity), flow(0), from(from), to(to) {}

		int initialCapacity;
		int residualCapacity;
		int flow;
		AllDiffNode* from;
		AllDiffNode* to;
	};

	struct AllDiffNode {
		explicit AllDiffNode(AllDiffNodeType type) : type(type), var(nullptr), value(-1), visited(false), parent(nullptr) {}

		AllDiffNodeType type;
		Variable* var;
		int value;
		std::vector<AllDiffEdge*> adjencyList;
		bool visited;
		AllDiffEdge* parent;
	};

	void FordFulkersonAlgorithm(const std::vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target);
	void ReginAlgorithm(const std::vector<Variable*>& vars);
}
