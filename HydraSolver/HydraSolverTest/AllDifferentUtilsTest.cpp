#include "stdafx.h"
#include "CppUnitTest.h"
#include "AllDifferentUtils.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {

	TEST_CLASS(AllDifferentUtilsTest) {
public:
	TEST_METHOD(FordFulkersonAlgorithmShouldReturnValidFlowValue_simpleCase) {
		auto expectedFlowValue = 7;
		vector<AllDiffNode*> nodes;
		auto node1 = new AllDiffNode(SOURCE);
		auto node2 = new AllDiffNode(VARIABLE);
		auto node3 = new AllDiffNode(VARIABLE);
		auto node4 = new AllDiffNode(VARIABLE);
		auto node5 = new AllDiffNode(VARIABLE);
		auto node6 = new AllDiffNode(TARGET);

		auto edge12 = new AllDiffEdge(3, node1, node2);
		auto edge13 = new AllDiffEdge(7, node1, node3);
		node1->adjencyList.push_back(edge12);
		node1->adjencyList.push_back(edge13);

		auto edge24 = new AllDiffEdge(2, node2, node4);
		auto edge25 = new AllDiffEdge(2, node2, node5);
		node2->adjencyList.push_back(edge24);
		node2->adjencyList.push_back(edge25);

		auto edge35 = new AllDiffEdge(4, node3, node5);
		node3->adjencyList.push_back(edge35);

		auto edge46 = new AllDiffEdge(3, node4, node6);
		node4->adjencyList.push_back(edge46);

		auto edge56 = new AllDiffEdge(5, node5, node6);
		node5->adjencyList.push_back(edge56);

		nodes.push_back(node1);
		nodes.push_back(node2);
		nodes.push_back(node3);
		nodes.push_back(node4);
		nodes.push_back(node5);
		nodes.push_back(node6);

		auto flowVector = FordFulkersonAlgorithm(nodes, node1, node6);
		auto result = calculateFlowValue(flowVector, node1, node2);
		// TODO : uncomment to test the FordFulkersonAlgorithm function
		//Assert::AreEqual(expectedFlowValue, result);

		for (auto node : nodes) {
			for (auto edge : node->adjencyList) {
				delete edge;
			}
			delete node;
		}
	}

	};
}