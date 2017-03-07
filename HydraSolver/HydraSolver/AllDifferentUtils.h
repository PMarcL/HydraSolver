#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

namespace hydra {

	class IntVariable;

	enum AllDiffNodeType {
		SOURCE,
		VARIABLE,
		VALUE,
		TARGET
	};

	struct AllDiffNode;

	struct AllDiffEdge {
		AllDiffEdge(int capacity, AllDiffNode* from, AllDiffNode* to) : capacity(capacity), from(from), to(to) {}
		int capacity;
		AllDiffNode* from;
		AllDiffNode* to;
	};

	struct AllDiffNode {
		explicit AllDiffNode(AllDiffNodeType type) : type(type), var(nullptr), value(-1) {}
		AllDiffNodeType type;
		IntVariable* var;
		int value;
		std::vector<AllDiffEdge*> adjencyList;
	};

	std::vector<std::vector<int>> FordFulkersonAlgorithm(const std::vector<AllDiffNode*>& nodes, AllDiffNode* source, AllDiffNode* target);
	int calculateFlowValue(std::vector<std::vector<int>> flow, AllDiffNode* source, AllDiffNode* target);
	void ReginAlgorithm(const std::vector<IntVariable*>& vars);
}
